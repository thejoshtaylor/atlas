"""`CameraAudioSource`: proves the zero-transcode claim by byte identity and
a zero decode count, not by reading the code (VOICE-04, SRC-02, PROV-07).

Every case that needs a real container drives a genuine A-law fixture file
through the real class -- only `open_container` is swapped for a local
file, the same substitution `tests/test_room_tracer.py` uses, never a
faked `CameraAudioSource` itself.

The reconnect case (SRC-04) belongs to plan 02-07 and stays the Wave-0
scaffold below, deliberately still red.
"""

from __future__ import annotations

from typing import Any

import av

from spire_voice.config import CameraConfig
from spire_voice.transports.base import SourceFormat
from spire_voice.transports.camera import CameraAudioSource

# A deterministic, non-repeating byte sequence, long enough to demux into
# several packets -- a constant buffer would let a byte-identity assertion
# pass vacuously.
_RAW_ALAW_BYTES = bytes((i * 11 + 5) % 256 for i in range(4000))


class _RecordingSpeaker:
    """A minimal `_SpeakerSink`: records every chunk `send_audio()` forwards."""

    def __init__(self) -> None:
        self.received: list[bytes] = []

    async def write(self, chunk: bytes) -> None:
        self.received.append(chunk)


def _make_open_container(fixture_path: str):
    def _open(_config: CameraConfig) -> Any:
        return av.open(
            fixture_path,
            format="alaw",
            options={"sample_rate": "8000", "ar": "8000", "ac": "1"},
        )

    return _open


def _build_camera_source(tmp_path, *, raw_bytes: bytes = _RAW_ALAW_BYTES) -> tuple[CameraAudioSource, _RecordingSpeaker]:
    fixture_path = str(tmp_path / "camera_fixture.alaw")
    with open(fixture_path, "wb") as fh:
        fh.write(raw_bytes)

    config = CameraConfig(rtsp_url="rtsp://redacted@camera.invalid/stream1", encoding="alaw", sample_rate=8000)
    speaker = _RecordingSpeaker()
    source = CameraAudioSource(config, speaker, open_container=_make_open_container(fixture_path))
    return source, speaker


async def test_camera_source_satisfies_audio_source_end_to_end(tmp_path):
    """SRC-02: the real class satisfies every member of `AudioSource`
    against fake RTSP packet bytes, the same way `test_transports.py`
    proves the two browser transports do."""
    source, speaker = _build_camera_source(tmp_path)

    assert hasattr(source, "frames")
    assert hasattr(source, "send_audio")
    assert hasattr(source, "send_event")
    assert hasattr(source, "source_format")

    source.start()
    frames = [chunk async for chunk in source.frames()]
    assert b"".join(frames) == _RAW_ALAW_BYTES

    await source.send_audio(b"reply-chunk")
    assert speaker.received == [b"reply-chunk"]

    # A no-op that must not raise -- a camera has nowhere to render an
    # event, but a silent failure here would still be a bug.
    await source.send_event({"type": "reply.text", "text": "the light is on"})

    await source.close()


async def test_camera_source_declares_its_own_alaw_format():
    """The camera's declared format is its own, not the browsers' 16 kHz
    PCM16 -- `XaiStt.build_url()` reads this rather than assuming a format
    (RESEARCH.md Pattern 1)."""
    config = CameraConfig(rtsp_url="rtsp://redacted@camera.invalid/stream1", encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(config, _RecordingSpeaker())

    assert source.source_format() == SourceFormat("alaw", 8000)


async def test_camera_source_yields_bit_identical_alaw_no_transcode(tmp_path):
    """VOICE-04 / PROV-07, the load-bearing case: byte identity alone would
    also pass against a decode-and-re-encode round trip that happened to be
    lossless, so this also asserts a zero decode count on the raw path --
    and, on the same chunks, a non-zero count once they are also handed to
    the detector-only path. The two assertions together are what
    distinguishes "the bytes happen to match" from "the bytes were never
    decoded".
    """
    source, _speaker = _build_camera_source(tmp_path)
    source.start()

    frames = [chunk async for chunk in source.frames()]
    assert b"".join(frames) == _RAW_ALAW_BYTES
    assert source.detector_decode_calls == 0

    for chunk in frames:
        source.decode_for_detector(chunk)
    assert source.detector_decode_calls == len(frames)

    await source.close()


async def test_camera_source_detector_copy_is_16khz_mono_pcm16(tmp_path):
    """RESEARCH.md Pattern 3: the detector-only copy is 16 kHz mono
    signed-16-bit, sliced to its valid sample count -- the over-allocation
    trap `webrtc.py`'s own resample comment documents, now on a second
    path."""
    source, _speaker = _build_camera_source(tmp_path)
    source.start()

    frames = [chunk async for chunk in source.frames()]
    detector_bytes = bytearray()
    for chunk in frames:
        detector_bytes += source.decode_for_detector(chunk)
    detector_bytes = bytes(detector_bytes)

    # 8 kHz mono A-law -> 16 kHz mono signed-16-bit doubles the sample
    # count; a resampler's own filter warm-up latency keeps this from
    # landing on an exact figure, the same slack
    # `test_transports.py::test_webrtc_resamples_non_16khz_mono_frames`
    # tolerates for the identical reason.
    expected_samples = len(_RAW_ALAW_BYTES) * 2
    actual_samples = len(detector_bytes) // 2
    assert abs(actual_samples - expected_samples) <= 64

    await source.close()


def test_camera_source_reconnects_after_a_dropped_connection():
    """SRC-04: a dropped RTSP connection must reconnect on its own -- neither
    PyAV nor FFmpeg do this automatically. Deliberately left red by plan
    02-03 (which owns the four tests above); turned green by plan 02-07,
    which adds the supervised reconnect loop.
    """
    raise AssertionError("plan 02-07 turns this green: the reconnect supervisor does not exist yet")
