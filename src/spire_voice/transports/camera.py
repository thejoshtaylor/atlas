"""The camera transport: RTSP in, undecoded A-law straight through to STT.

PyAV and FFmpeg own RTSP session handling and codec parsing here -- this
module never reimplements either, the same scoping posture `webrtc.py`
states for Opus/DTLS-SRTP.

`CameraAudioSource` is the third implementation of `AudioSource`
(transports/base.py). It never becomes a second pipeline: it satisfies the
same four-member protocol the two browser transports do, so `run_turn`
stays exactly as ignorant of this source as it is of them.

Two paths leave one `container.demux()` loop, and only one of them may ever
reach a transcription provider:

1. The **raw path** (`frames()`): the packet's own undecoded bytes, read at
   the packet level rather than the frame level (RESEARCH.md Pattern 2).
   This is what PROV-07 and VOICE-04 require -- the bytes handed onward are
   the bytes the camera sent, byte for byte, with no codec touching them a
   second time.
2. The **detector-only path** (`decode_for_detector()`): a *second*,
   independent decode of the same packet bytes, resampled to 16 kHz mono
   PCM16 for the wake engine (RESEARCH.md Pattern 3, generalizing
   `webrtc.py`'s own `_frame_to_pcm16`). This decoded copy exists for the
   detector alone and never substitutes for the raw path -- that is a
   stated invariant of this module, not an implementation detail: nothing
   in this file ever hands a decoded byte to `frames()`'s queue.

The RTSP URL (`CameraConfig.rtsp_url`) carries the camera's own credentials
embedded in it. No log line, exception message, or test fixture in this
module may ever include it or the raw text of an exception that might
repeat it back -- connection failures are logged by exception *type* only.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, AsyncIterator, Callable, Protocol

import av
from av.audio.resampler import AudioResampler

from spire_voice.config import CameraConfig
from spire_voice.transports.base import SourceFormat

logger = logging.getLogger("spire_voice.transports.camera")

_DETECTOR_FORMAT = "s16"
_DETECTOR_LAYOUT = "mono"
_DETECTOR_SAMPLE_RATE = 16000


class _SpeakerSink(Protocol):
    """The one thing `send_audio()` needs: an async `write()`.

    Structural, not `speaker.fifo_writer.FifoWriter` imported by name --
    this module stays ignorant of how reply audio actually reaches the
    camera speaker, the same discipline `providers/base.py`'s protocols
    already establish everywhere else in this codebase.
    """

    async def write(self, chunk: bytes) -> None: ...


def _open_rtsp_container(config: CameraConfig) -> Any:
    """Open the camera's RTSP audio stream over TCP with a socket timeout.

    The default `open_container` every real deployment uses. A test
    substitutes its own factory (see `tests/test_room_tracer.py` and
    `tests/test_camera_source.py`) to open a local fixture file instead --
    the class under test never changes, only how its container is opened.
    """
    return av.open(
        config.rtsp_url,
        options={"rtsp_transport": "tcp", "stimeout": "5000000"},
        timeout=5.0,
    )


class CameraAudioSource:
    """Satisfies `AudioSource` over one RTSP audio stream.

    Constructed once, holding a queue fed by one long-lived consumer task --
    never a task per packet (the same shape `WebrtcTransport` uses for its
    inbound track). `start()` schedules that task and returns immediately:
    the actual RTSP open can block for seconds or fail outright if the
    camera or network is unreachable, and doing that inline would block
    `lifespan` itself. A camera that never connects leaves this source
    idle and logs it (T-02-13, accepted risk) -- the rest of the
    application keeps working.
    """

    def __init__(
        self,
        config: CameraConfig,
        speaker: _SpeakerSink,
        *,
        open_container: Callable[[CameraConfig], Any] = _open_rtsp_container,
    ) -> None:
        self._config = config
        self._speaker = speaker
        self._open_container = open_container
        self._queue: asyncio.Queue[bytes | None] = asyncio.Queue()
        self._consumer_task: asyncio.Task[None] | None = None
        self._container: Any | None = None
        self._detector_codec_context: Any | None = None
        self._detector_resampler = AudioResampler(
            format=_DETECTOR_FORMAT, layout=_DETECTOR_LAYOUT, rate=_DETECTOR_SAMPLE_RATE
        )
        # Incremented once per `decode_for_detector()` call -- proof, for
        # `tests/test_camera_source.py`, that the raw path never decodes
        # while the detector path always does (PROV-07's stronger claim:
        # not just "the bytes match" but "no codec ran on that path").
        self.detector_decode_calls = 0

    def start(self) -> None:
        """Start the one long-lived RTSP consumer task. Never awaited by
        the caller -- see the class docstring."""
        loop = asyncio.get_running_loop()
        self._consumer_task = asyncio.create_task(self._run(loop))

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield the camera's own undecoded packet bytes until the stream ends."""
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        await self._speaker.write(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        """No-op: a camera has nowhere to render a partial transcript, a
        reply line, or a timing line. Logged at debug on purpose -- a
        silent no-op that looks unimplemented is worse than a documented
        one, during a live debugging session."""
        logger.debug("camera source send_event no-op: %s", event.get("type"))

    def source_format(self) -> SourceFormat:
        return SourceFormat(self._config.encoding, self._config.sample_rate)

    def decode_for_detector(self, raw_chunk: bytes) -> bytes:
        """Decode-and-resample `raw_chunk` to 16 kHz mono PCM16, for the
        wake detector only.

        This is a *second* decode of bytes `frames()` already yielded
        undecoded -- it never replaces or mutates the raw path (module
        invariant). Returns `b""` before the container has finished
        opening, or when this chunk's bytes complete no whole frame yet (a
        codec's own internal buffering can hold partial data across calls).
        """
        if self._detector_codec_context is None:
            return b""
        self.detector_decode_calls += 1
        packet = av.Packet(raw_chunk)
        out = bytearray()
        for frame in self._detector_codec_context.decode(packet):
            for resampled in self._detector_resampler.resample(frame):
                valid_length = (
                    resampled.samples * resampled.format.bytes * len(resampled.layout.channels)
                )
                out += bytes(resampled.planes[0])[:valid_length]
        return bytes(out)

    async def close(self) -> None:
        """Cancel the consumer task and release the container, best effort."""
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer_task
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.close()

    async def _run(self, loop: asyncio.AbstractEventLoop) -> None:
        try:
            await asyncio.to_thread(self._read_loop, loop)
        finally:
            loop.call_soon_threadsafe(self._queue.put_nowait, None)

    def _read_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Runs in a worker thread: `container.demux()` is a blocking
        call, and bridging it onto the event loop's queue happens through
        `call_soon_threadsafe` rather than blocking that loop directly."""
        try:
            container = self._open_container(self._config)
        except Exception as exc:  # noqa: BLE001 -- any open failure leaves this source idle (T-02-13), never crashes lifespan
            # Deliberately logs the exception's TYPE only, never its message
            # or `self._config.rtsp_url`: both may carry the camera's
            # embedded RTSP credentials (module docstring).
            logger.warning(
                "camera RTSP source failed to open (%s); leaving this source idle",
                type(exc).__name__,
            )
            return

        self._container = container
        try:
            audio_stream = container.streams.audio[0]
            self._detector_codec_context = self._build_detector_codec_context(audio_stream)
            for packet in container.demux(audio_stream):
                if packet.dts is None:
                    continue  # the end-of-stream flush packet, not audio
                raw = bytes(packet)
                loop.call_soon_threadsafe(self._queue.put_nowait, raw)
        except Exception as exc:  # noqa: BLE001 -- a mid-stream drop also leaves this source idle; plan 02-07 adds the reconnect supervisor
            logger.warning("camera RTSP source dropped mid-stream (%s)", type(exc).__name__)
        finally:
            with contextlib.suppress(Exception):
                container.close()

    def _build_detector_codec_context(self, audio_stream: Any) -> Any:
        """A second, independent decode context built from the stream's own
        codec (RESEARCH.md Pattern 3) -- `sample_rate`/`format`/`layout`
        must be set explicitly on a freshly created context or `decode()`
        raises immediately (`avcodec_open2` needs them; they are not
        inherited from the codec name alone, verified this session)."""
        codec_context = av.CodecContext.create(audio_stream.codec_context.codec, "r")
        codec_context.sample_rate = audio_stream.codec_context.sample_rate
        codec_context.format = audio_stream.codec_context.format
        codec_context.layout = audio_stream.codec_context.layout
        return codec_context
