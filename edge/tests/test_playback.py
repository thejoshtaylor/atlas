"""RED for Playback (10-09-PLAN.md Task 2, per the orchestrator directive:
Playback never opens its own RawOutputStream -- it writes through a fake
`enqueue` sink, the same seam Capture.enqueue_playback provides in
production). No real audio device anywhere in this file."""

from __future__ import annotations

import logging

import pytest

from atlas_edge.playback import Playback


def _fake_clock(values):
    it = iter(values)

    def _clock():
        return next(it)

    return _clock


@pytest.mark.asyncio
async def test_mono_samples_are_duplicated_into_both_channels() -> None:
    sunk: "list[bytes]" = []
    playback = Playback(sunk.append, clock=_fake_clock([0.0] * 10))

    # Two 2-byte PCM16 "samples": a=0x0102, b=0x0304.
    mono = bytes([0x01, 0x02, 0x03, 0x04])
    await playback.write(mono)

    assert b"".join(sunk) == bytes([0x01, 0x02, 0x01, 0x02, 0x03, 0x04, 0x03, 0x04])


@pytest.mark.asyncio
async def test_an_odd_trailing_byte_is_kept_for_the_next_write() -> None:
    sunk: "list[bytes]" = []
    playback = Playback(sunk.append, clock=_fake_clock([0.0] * 10))

    await playback.write(bytes([0x01, 0x02, 0x03]))  # one full sample + 1 leftover byte
    assert b"".join(sunk) == bytes([0x01, 0x02, 0x01, 0x02])

    sunk.clear()
    await playback.write(bytes([0x04]))  # completes the leftover sample: [0x03, 0x04]
    assert b"".join(sunk) == bytes([0x03, 0x04, 0x03, 0x04])


@pytest.mark.asyncio
async def test_a_buffer_past_max_buffer_s_drops_the_newest_audio_and_counts_it() -> None:
    sunk: "list[bytes]" = []
    # A frozen clock -- no decay between writes -- so the buffer only grows.
    playback = Playback(
        sunk.append,
        sample_rate=16000,
        channels=2,
        max_buffer_s=0.001,  # tiny cap: 16000*2*2*0.001 = 64 bytes
        clock=_fake_clock([0.0] * 100),
    )

    mono_chunk = bytes(range(64)) * 4  # 256 mono bytes -> 512 stereo bytes, far over cap
    await playback.write(mono_chunk)

    assert playback.dropped_bytes > 0
    assert len(b"".join(sunk)) <= 64
    # The bytes that reach the sink are the tail of the incoming chunk's stereo form.
    stereo = b"".join(mono_chunk[i : i + 2] * 2 for i in range(0, len(mono_chunk), 2))
    assert b"".join(sunk) == stereo[-len(b"".join(sunk)) :]
    assert playback.dropped_bytes == len(stereo) - len(b"".join(sunk))


@pytest.mark.asyncio
async def test_an_overflow_logs_one_warning_per_episode(caplog: pytest.LogCaptureFixture) -> None:
    now = [0.0]
    playback = Playback(
        lambda data: None,
        sample_rate=16000,
        channels=2,
        max_buffer_s=0.001,  # 64 bytes
        clock=lambda: now[0],
    )
    chunk = bytes(256)

    with caplog.at_level(logging.DEBUG, logger="atlas_edge.playback"):
        for _ in range(50):
            await playback.write(chunk)
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert len(warnings) == 1
        assert "newest" in warnings[0].getMessage()

        # The buffer drains, and the next overflow is a new episode.
        now[0] = 10.0
        await playback.write(chunk)
        await playback.write(chunk)

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 2
    assert playback.dropped_bytes > 0


@pytest.mark.asyncio
async def test_the_buffer_estimate_decays_over_elapsed_time() -> None:
    sunk: "list[bytes]" = []
    bytes_per_second = 16000 * 2 * 2
    max_buffer_s = 1.0
    playback = Playback(
        sunk.append,
        sample_rate=16000,
        channels=2,
        max_buffer_s=max_buffer_s,
        # First write establishes last_update_at at t=0. Second write is at
        # t=2.0s later -- long enough to fully drain any prior estimate.
        clock=_fake_clock([0.0, 2.0]),
    )

    small = bytes([0x00, 0x01])  # 2 mono bytes -> 4 stereo bytes
    await playback.write(small)
    await playback.write(small)

    assert playback.dropped_bytes == 0


@pytest.mark.asyncio
async def test_start_and_stop_are_harmless_no_ops() -> None:
    playback = Playback(lambda data: None)
    playback.start()
    playback.stop()


@pytest.mark.asyncio
async def test_pending_s_is_zero_before_any_write_then_decays_to_zero() -> None:
    playback = Playback(lambda data: None, clock=_fake_clock([0.0, 0.0, 0.25, 5.0]))

    assert playback.pending_s() == 0.0

    # 1 s of mono audio at 16 kHz: 32000 bytes.
    await playback.write(bytes(32000))

    assert abs(playback.pending_s() - 0.75) < 0.01
    assert playback.pending_s() == 0.0


@pytest.mark.asyncio
async def test_clear_drops_the_estimate_to_the_kept_bytes_and_a_later_write_works() -> None:
    sunk: "list[bytes]" = []
    # 16000 Hz x 2 channels x 2 bytes = 64000 bytes per second.
    playback = Playback(sunk.append, clock=_fake_clock([0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0]))

    await playback.write(bytes(32000))  # 64000 stereo bytes: 1.0 s
    await playback.write(bytes([0x01]))  # an odd leftover byte is held
    playback.clear(6400)  # keep 0.1 s

    assert playback.pending_s() == pytest.approx(0.1)

    await playback.write(bytes([0x02, 0x03]))  # the old leftover must not misalign this
    assert sunk[-1] == bytes([0x02, 0x03, 0x02, 0x03])


def test_clear_with_no_kept_bytes_empties_the_estimate() -> None:
    playback = Playback(lambda _data: None, clock=_fake_clock([0.0] * 10))
    playback.clear()
    assert playback.pending_s() == 0.0
