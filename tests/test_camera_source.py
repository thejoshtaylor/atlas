"""`CameraAudioSource`: proves the zero-transcode claim by byte identity and
a zero decode count, not by reading the code (VOICE-04, SRC-02, PROV-07).

Every case that needs a real container drives a genuine A-law fixture file
through the real class -- only `open_container` is swapped for a local
file, the same substitution `tests/test_room_tracer.py` uses, never a
faked `CameraAudioSource` itself.

The reconnect cases (SRC-04, plan 02-07) use fully synthetic fake
containers instead, in the conftest fake convention `test_speaker_fifo.py`'s
own `_FakeSpawner` already establishes: what these prove is the reconnect
supervisor's own single-flight/already-connected/shutdown bookkeeping, not
PyAV's own demux behavior, which the four tests above already cover against
a real container.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from types import SimpleNamespace
from typing import Any

import av
import httpx

from spire_voice.config import CameraConfig, SpeakerConfig
from spire_voice.speaker.ffmpeg_supervisor import FfmpegSupervisor
from spire_voice.transports.base import SourceFormat
from spire_voice.transports.camera import CameraAudioSource

# A deterministic, non-repeating byte sequence, long enough to demux into
# several packets -- a constant buffer would let a byte-identity assertion
# pass vacuously.
_RAW_ALAW_BYTES = bytes((i * 11 + 5) % 256 for i in range(4000))

_CAMERA_URL = "rtsp://redacted@camera.invalid/stream1"


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

    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
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
    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
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


# ---------------------------------------------------------------------------
# SRC-04: the reconnect supervisor.
#
# A clean, exception-free end of `container.demux()` (the four fixture-file
# cases above, once the file's bytes run out) is not a drop, and the
# supervisor never retries it -- it behaves exactly as the pre-02-07,
# single-attempt shape did, which is what lets those four tests' own
# `[chunk async for chunk in source.frames()]` terminate on its own. Only an
# actual exception -- the open call failing, or `demux()` raising mid-read --
# counts as a drop worth retrying (T-02-28). Every fake below is built
# around that distinction.
# ---------------------------------------------------------------------------


async def _instant_sleep(_seconds: float) -> None:
    """An injectable backoff with no real delay: these tests prove retry
    ordering and bookkeeping, not real wall-clock timing (the plan's own
    injectable-clock idiom, applied to the backoff wait rather than to a
    boundary calculation)."""
    return None


class _ControllableSleep:
    """An injectable `sleep` a test can hold open, one call at a time, with
    no real delay at all -- not even `asyncio.sleep(0)`. Used for the one
    case that genuinely needs to catch the supervisor mid-backoff: closing
    the source while it awaits this cancels the `asyncio.Event.wait()`
    below immediately, the same real cancellation a production backoff
    wait would receive.
    """

    def __init__(self) -> None:
        self.calls: list[float] = []
        self._pending: asyncio.Event | None = None

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        event = asyncio.Event()
        self._pending = event
        await event.wait()

    def release_one(self) -> None:
        assert self._pending is not None, "no backoff wait is currently pending"
        pending, self._pending = self._pending, None
        pending.set()


def _fake_audio_stream() -> Any:
    """Just enough of a real PyAV stream's shape for
    `CameraAudioSource._build_detector_codec_context` -- a string codec
    name is accepted by `av.CodecContext.create` exactly as a real `Codec`
    object is (verified this session), so this needs no real opened
    container at all."""
    return SimpleNamespace(
        codec_context=SimpleNamespace(codec="pcm_alaw", sample_rate=8000, format="s16", layout="mono")
    )


class _ScriptedContainerFactory:
    """Counts calls; replays a scripted list of outcomes. Each outcome is
    either an `Exception` (raised, simulating an open failure), a callable
    (invoked with the config, for a real `av.open`-backed factory such as
    `_make_open_container`'s own return value), or anything else (returned
    as-is, for a fake container object). Calling past the end of the
    script repeats the last outcome, so a test does not have to predict
    exactly how many times the supervisor's own loop will call this before
    the test gets around to closing the source.
    """

    def __init__(self, outcomes: list) -> None:
        self._outcomes = list(outcomes)
        self.call_count = 0

    def __call__(self, config: CameraConfig) -> Any:
        index = min(self.call_count, len(self._outcomes) - 1)
        outcome = self._outcomes[index]
        self.call_count += 1
        if isinstance(outcome, BaseException):
            raise outcome
        if callable(outcome):
            return outcome(config)
        return outcome


class _SlowOpenFactory:
    """Counts calls; the first call blocks on a real `threading.Event`
    before returning `container` -- simulating an open attempt that is
    still in flight, the exact window `reconnect()`'s single-flight guard
    must hold shut against a second request. Real thread blocking, not a
    stand-in: `_read_loop` genuinely runs in a worker thread
    (`asyncio.to_thread`), so this call really does suspend there while
    the event loop keeps running everything else.
    """

    def __init__(self, container: Any) -> None:
        self._container = container
        self._release = threading.Event()
        self.call_count = 0

    def __call__(self, _config: CameraConfig) -> Any:
        self.call_count += 1
        self._release.wait()
        return self._container

    def release(self) -> None:
        self._release.set()


class _HoldOpenContainer:
    """A fake container: `demux()` yields `packet_count` scripted packets,
    then blocks its generator on a real `threading.Event` until released
    or closed -- "still connected, still draining" for as long as a test
    needs, not merely fast-forwarded past.
    """

    def __init__(self, packet_count: int = 1) -> None:
        self.streams = SimpleNamespace(audio=[_fake_audio_stream()])
        self._packet_count = packet_count
        self._hold = threading.Event()
        self.closed = False

    def demux(self, _stream: Any):
        for i in range(self._packet_count):
            packet = av.Packet(bytes([i % 256, (i + 1) % 256]))
            packet.dts = i
            yield packet
        self._hold.wait()

    def release(self) -> None:
        self._hold.set()

    def close(self) -> None:
        self.closed = True
        self._hold.set()


class _ImmediateDropContainer:
    """A fake container that opens fine but drops the instant its stream
    is read -- distinct from an open failure (`boom` below): this is
    "connected, then immediately gone", the shape a real mid-stream drop
    takes.
    """

    def __init__(self) -> None:
        self.streams = SimpleNamespace(audio=[_fake_audio_stream()])
        self.closed = False

    def demux(self, _stream: Any):
        raise ConnectionError("simulated mid-stream drop")
        yield  # pragma: no cover -- unreachable; keeps this a generator function

    def close(self) -> None:
        self.closed = True


async def _wait_for(predicate, *, attempts: int = 500, interval: float = 0.005) -> None:
    for _ in range(attempts):
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError(f"condition never became true after {attempts * interval:.2f}s")


async def test_camera_source_reconnects_after_a_dropped_connection_and_logs_the_attempt(caplog):
    """SRC-04's core claim: an open failure is retried, and the retry that
    follows is visible in the log naming what ended the previous attempt
    -- a source stuck retrying must never look like a quiet room
    (T-02-28).

    Uses a fake second container, not a real fixture file: the point here
    is the supervisor's own retry/log bookkeeping, already proven byte-
    faithful by the four fixture-backed tests above. A real `av` container
    closed from this test's own thread while its worker thread might still
    be mid-`demux()` is a genuine concurrent-access hazard against PyAV's C
    extension (a segfault, reproduced this session) -- the fakes below
    hold their "connection" open on a plain `threading.Event` instead,
    which this test releases before ever calling `close()`.
    """
    boom = RuntimeError("simulated open failure")
    second_attempt = _HoldOpenContainer(packet_count=1)
    factory = _ScriptedContainerFactory([boom, second_attempt])
    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(
        config, _RecordingSpeaker(), open_container=factory, backoff_s=0.0, sleep=_instant_sleep
    )

    with caplog.at_level(logging.WARNING, logger="spire_voice.transports.camera"):
        source.start()
        frames_iter = source.frames()
        first_chunk = await asyncio.wait_for(frames_iter.__anext__(), timeout=2.0)

    assert first_chunk
    assert factory.call_count == 2
    assert any("RuntimeError" in record.getMessage() for record in caplog.records)
    # The module's own hard rule: never the RTSP URL, never an exception's
    # raw message text, in any log line this retry loop emits.
    assert not any(_CAMERA_URL in record.getMessage() for record in caplog.records)

    second_attempt.release()
    await source.close()


async def test_camera_source_reconnect_holds_a_single_attempt_at_a_time():
    """T-02-29: a second reconnect request while the first is still trying
    to open must not open a second connection -- asserted by counting
    factory calls, not by inspecting internal state."""
    container = _HoldOpenContainer(packet_count=1)
    factory = _SlowOpenFactory(container)
    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(
        config, _RecordingSpeaker(), open_container=factory, backoff_s=0.0, sleep=_instant_sleep
    )

    source.start()
    await _wait_for(lambda: factory.call_count == 1)

    # The first attempt is still blocked inside the open call itself. A
    # second request now must be a no-op: the guard is checked before the
    # factory is ever invoked a second time.
    await source.reconnect()
    assert factory.call_count == 1

    factory.release()
    await _wait_for(lambda: source._connected)
    container.release()
    await source.close()


async def test_reconnecting_an_already_connected_camera_source_is_a_no_op():
    """T-02-29's other half: once connected, a reconnect request is a
    no-op rather than a second connection -- a supervisor retry racing a
    link that recovered on its own must not leave two readers on one
    camera."""
    container = _HoldOpenContainer(packet_count=2)
    factory = _ScriptedContainerFactory([container])
    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(
        config, _RecordingSpeaker(), open_container=factory, backoff_s=0.0, sleep=_instant_sleep
    )

    source.start()
    frames_iter = source.frames()
    first_chunk = await asyncio.wait_for(frames_iter.__anext__(), timeout=2.0)
    assert first_chunk

    await source.reconnect()
    assert factory.call_count == 1

    container.release()
    await source.close()


async def test_camera_source_shutdown_during_reconnect_backoff_ends_the_loop():
    """T-02-30: a source torn down while it waits out its backoff ends
    that wait rather than reviving itself for one more attempt -- no
    further factory call after `close()`."""
    boom = RuntimeError("simulated open failure")
    factory = _ScriptedContainerFactory([boom])
    sleep = _ControllableSleep()
    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(config, _RecordingSpeaker(), open_container=factory, backoff_s=5.0, sleep=sleep)

    source.start()
    await _wait_for(lambda: len(sleep.calls) == 1)
    assert factory.call_count == 1

    await source.close()

    assert factory.call_count == 1
    assert sleep.calls == [5.0]


async def test_camera_source_calls_on_reconnect_only_after_the_first_successful_connect():
    """The reconnect callback (Task 2's wiring point) fires for a genuine
    *re*-connect only -- never for the very first successful connect,
    which is not a recovery from anything."""
    boom = RuntimeError("simulated open failure")
    first_success = _ImmediateDropContainer()
    second_success = _HoldOpenContainer(packet_count=1)
    factory = _ScriptedContainerFactory([boom, first_success, second_success])
    reconnect_calls: list[int] = []

    async def _on_reconnect() -> None:
        reconnect_calls.append(1)

    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(
        config,
        _RecordingSpeaker(),
        open_container=factory,
        backoff_s=0.0,
        sleep=_instant_sleep,
        on_reconnect=_on_reconnect,
    )

    source.start()
    await _wait_for(lambda: factory.call_count >= 3)
    await _wait_for(lambda: len(reconnect_calls) >= 1)
    assert reconnect_calls == [1]

    second_success.release()
    await source.close()


async def test_camera_source_re_ensures_the_speaker_backchannel_on_reconnect():
    """Task 2: a reconnected camera gets its speaker back too. Wires a
    real `FfmpegSupervisor.handle_reconnect` as the camera's own
    `on_reconnect` callback -- proving the two supervisors actually
    connect, not just that each one's own half works in isolation."""

    class _RecordingBackchannel:
        def __init__(self) -> None:
            self.put_count = 0

        def __call__(self, request: httpx.Request) -> httpx.Response:
            if request.method == "PUT":
                self.put_count += 1
            return httpx.Response(200)

    backchannel = _RecordingBackchannel()
    http_client = httpx.AsyncClient(transport=httpx.MockTransport(backchannel))
    speaker_config = SpeakerConfig(ensure_url="http://go2rtc.invalid/api/streams/cam")
    ffmpeg_supervisor = FfmpegSupervisor(speaker_config, http_client=http_client)

    boom = RuntimeError("simulated open failure")
    first_success = _ImmediateDropContainer()
    second_success = _HoldOpenContainer(packet_count=1)
    factory = _ScriptedContainerFactory([boom, first_success, second_success])

    config = CameraConfig(rtsp_url=_CAMERA_URL, encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(
        config,
        _RecordingSpeaker(),
        open_container=factory,
        backoff_s=0.0,
        sleep=_instant_sleep,
        on_reconnect=ffmpeg_supervisor.handle_reconnect,
    )

    source.start()
    await _wait_for(lambda: backchannel.put_count >= 1)
    assert backchannel.put_count == 1

    second_success.release()
    await source.close()
    await http_client.aclose()
