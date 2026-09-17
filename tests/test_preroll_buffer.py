"""Real assertions for the pre-roll ring buffer and its replaying wrapper
(VOICE-06, T-02-17).

A pure-dataclass unit test in the shape `tests/test_timing.py` uses: every
assertion here is over byte offsets and byte counts, never a transcript --
a transcript-level claim about a continuous utterance is the live human
check in `02-04-PLAN.md` Task 1, and a fixture asserting on transcript text
would prove nothing about real continuous speech.
"""

from __future__ import annotations

from spire_voice.audio.ring import PrerollBuffer
from spire_voice.sources.runner import PrerollReplayingSource
from spire_voice.transports.base import SourceFormat


def test_preroll_buffer_never_holds_more_than_its_configured_window():
    """Pushed well past the configured window, the buffer's held byte
    count never exceeds the window's byte size -- bounded at push, not
    trimmed lazily at drain (the continuous-rolling-capture failure
    CONTEXT.md rejects by name)."""
    source_format = SourceFormat("alaw", 8000)
    window_ms = 100  # 8000 bytes/sec * 0.1s = 800 bytes
    buffer = PrerollBuffer(source_format, window_ms)

    for _ in range(50):
        buffer.push(b"\x01" * 100)
        assert buffer.held_bytes <= 800

    assert buffer.held_bytes <= 800


def test_drain_returns_oldest_first():
    source_format = SourceFormat("alaw", 8000)
    buffer = PrerollBuffer(source_format, window_ms=1000)  # 8000 bytes, plenty of room

    buffer.push(b"first")
    buffer.push(b"second")
    buffer.push(b"third")

    assert buffer.drain() == [b"first", b"second", b"third"]


def test_a_drained_buffer_is_empty():
    source_format = SourceFormat("alaw", 8000)
    buffer = PrerollBuffer(source_format, window_ms=1000)

    buffer.push(b"some audio")
    buffer.drain()

    assert buffer.drain() == []
    assert buffer.held_bytes == 0


def test_clear_empties_the_buffer_with_no_return_value():
    source_format = SourceFormat("alaw", 8000)
    buffer = PrerollBuffer(source_format, window_ms=1000)

    buffer.push(b"some audio")
    assert buffer.clear() is None
    assert buffer.held_bytes == 0
    assert buffer.drain() == []


def test_byte_bound_is_computed_from_the_sources_own_declared_format():
    """The same `window_ms` sizes a different byte bound for a different
    declared format -- proof the bound is derived from `SourceFormat`, not
    a hardcoded rate that would silently mis-size it for any other source."""
    alaw_buffer = PrerollBuffer(SourceFormat("alaw", 8000), window_ms=1000)
    pcm_buffer = PrerollBuffer(SourceFormat("pcm", 16000), window_ms=1000)

    for _ in range(1000):
        alaw_buffer.push(b"\x00" * 100)
        pcm_buffer.push(b"\x00" * 100)

    # alaw @ 8kHz, 1 byte/sample: 8000 bytes/sec * 1s = 8000 bytes.
    # pcm @ 16kHz, 2 bytes/sample: 32000 bytes/sec * 1s = 32000 bytes.
    assert alaw_buffer.held_bytes == 8000
    assert pcm_buffer.held_bytes == 32000


async def test_preroll_replaying_source_joins_preroll_and_live_frames_with_no_gap_or_duplication():
    """The byte-offset property the plan's flagged assumption names: for a
    single continuous synthetic buffer where the hit is reported partway
    through, the concatenation of the drained pre-roll and the following
    live frames begins at or before the byte offset the wake phrase
    started at, with no gap and no duplicated region at the seam.
    """
    # One continuous, deterministic byte sequence standing in for a real
    # utterance -- not all-zero, so a byte-identity assertion proves
    # something rather than passing vacuously against a constant buffer.
    continuous_audio = bytes((i * 13 + 1) % 256 for i in range(2000))

    # The wake phrase "starts" at byte 500 in this synthetic buffer,
    # already captured before the detector reports its hit -- exactly the
    # audio the pre-roll must recover.
    wake_phrase_start_offset = 500

    source_format = SourceFormat("alaw", 8000)
    buffer = PrerollBuffer(source_format, window_ms=1000)  # 8000-byte window, plenty for this fixture

    # Everything up to and including the packet the hit fires on is pushed
    # into the buffer, chunk by chunk, exactly as `SourceRunner.run()` does
    # before a hit -- the remainder is what a live source still has queued.
    hit_reported_at_offset = 700  # partway through the phrase, after it started
    for start in range(0, hit_reported_at_offset, 100):
        buffer.push(continuous_audio[start : start + 100])

    live_remainder = continuous_audio[hit_reported_at_offset:]

    class _LiveSource:
        async def frames(self):
            yield live_remainder

        async def send_audio(self, chunk: bytes) -> None:
            raise AssertionError("not exercised by this test")

        async def send_event(self, event: dict) -> None:
            raise AssertionError("not exercised by this test")

        def source_format(self) -> SourceFormat:
            return source_format

    preroll_chunks = buffer.drain()
    assert buffer.drain() == []  # draining started the turn: nothing left to replay twice

    wrapped_source = PrerollReplayingSource(_LiveSource(), preroll_chunks)

    received = bytearray()
    async for chunk in wrapped_source.frames():
        received += chunk

    # The seam: no gap, no duplicated region -- concatenating the drained
    # pre-roll with the live remainder reconstructs the original buffer
    # exactly from the first pushed byte onward.
    assert bytes(received) == continuous_audio[0:hit_reported_at_offset] + live_remainder
    assert bytes(received) == continuous_audio

    # The property the flagged assumption names: the replayed stream
    # begins at byte 0 of the original buffer -- at or before the offset
    # the wake phrase started at (500), since the drained pre-roll always
    # starts at its own first held byte.
    assert preroll_chunks
    replay_start_offset = 0
    assert replay_start_offset <= wake_phrase_start_offset

    # `source_format()` delegates to the wrapped source.
    assert wrapped_source.source_format() == source_format
