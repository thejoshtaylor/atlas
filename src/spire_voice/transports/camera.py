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

**The reconnect supervisor (SRC-04, plan 02-07):** neither PyAV nor the
FFmpeg libraries beneath it retry a dropped RTSP session on their own
(RESEARCH.md Pattern 4). `start()` schedules `_supervise()`, an outer loop
around one connect-and-drain attempt at a time: an attempt that ends via an
exception -- the open itself failed, or the stream dropped mid-read -- is
logged and followed by a backoff wait, then another attempt, forever, until
`close()`. An attempt that ends with no exception at all is not a drop
(practically never happens on a live RTSP stream; it is what a genuinely
finite source -- a test fixture file -- looks like once it has nothing left
to give) and is never retried, the same single-attempt behavior this module
had before this plan. Two properties are held as explicit state on the instance rather
than as an emergent consequence of the loop shape, so a reader (and a test)
can see the guarantee directly: `reconnect()` is a no-op while already
connected (T-02-29's first half) and a no-op while another attempt is
already in flight (T-02-29's second half) -- a supervisor retry racing a
link that recovered on its own must never leave two readers consuming the
same camera's packets. `frames()` itself never learns any of this happened:
the end-of-iteration sentinel is queued exactly once, when the supervisor
loop itself ends at shutdown, never after an individual dropped attempt --
so a caller mid-read across a drop simply sees a gap in the audio, which is
the truth, rather than a premature end of stream it must not have expected
(the sentinel-per-attempt shape this replaces is exactly what let a
`frames()` reader wait forever for a second sentinel that a different
reader, active during a turn, had already consumed -- see 02-06-SUMMARY.md's
Deviations).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from typing import Any, AsyncIterator, Awaitable, Callable, Protocol

import av
from av.audio.resampler import AudioResampler

from spire_voice.config import CameraConfig
from spire_voice.providers.tts_xai import SinkFormat
from spire_voice.transports.base import SourceFormat

logger = logging.getLogger("spire_voice.transports.camera")

_DETECTOR_FORMAT = "s16"
_DETECTOR_LAYOUT = "mono"
_DETECTOR_SAMPLE_RATE = 16000

# 260922-gde: this is the RTSP (mic) reconnect, not the speaker's. The two
# used to share one number, but the speaker side has since grown a real
# reason to be slower -- the camera's talk port locks itself out after
# repeated failed auths, and a short backoff kept re-arming that lockout
# (`SpeakerConfig.respawn_backoff_s`, config.py, now 30s). The RTSP read
# side has no such lockout, so it stays fast: a dropped microphone stream
# should reconnect quickly, not wait out a lockout timer that does not
# apply to it. `app.py` must not pass `config.speaker.respawn_backoff_s`
# in for `backoff_s` here -- that would make the mic inherit the speaker's
# slower backoff for no reason. `CameraConfig` carries no backoff field of
# its own, so this module-level default is this source's only home for it.
_DEFAULT_BACKOFF_S = 2.0

# 260922-cts: the camera speaker's own playback pair, matching `TtsConfig`'s
# own default `codec`/`sample_rate` (config.py) -- used only when a caller
# builds a `CameraAudioSource` without a `sink=` argument at all (every test
# in `tests/test_camera_source.py` that predates this fix). `app.py`'s real
# construction call always passes the configured `config.tts.codec`/
# `config.tts.sample_rate` pair explicitly; this default exists so a test
# double, or a future caller with no opinion on the sink, still gets today's
# real production value rather than an unformed one.
_DEFAULT_SINK = SinkFormat(codec="alaw", sample_rate=8000)


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
    `lifespan` itself. A camera that never connects, or that drops mid-
    stream, no longer just leaves this source idle (T-02-13's original
    accepted risk) -- the reconnect supervisor keeps retrying it, logging
    every attempt, until it succeeds or `close()` ends the process (T-02-28).
    """

    def __init__(
        self,
        config: CameraConfig,
        speaker: _SpeakerSink,
        *,
        sink: SinkFormat = _DEFAULT_SINK,
        open_container: Callable[[CameraConfig], Any] = _open_rtsp_container,
        backoff_s: float = _DEFAULT_BACKOFF_S,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        on_reconnect: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        self._config = config
        self._speaker = speaker
        self._sink = sink
        self._open_container = open_container
        self._backoff_s = backoff_s
        self._sleep = sleep
        self._on_reconnect = on_reconnect
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

        # The reconnect supervisor's own explicit state (module docstring):
        # `_connected` and `_attempt_task` are what `reconnect()` reads to
        # decide "no-op" versus "actually try" -- never an incidentally
        # acquired lock, so both guarantees are visible to a reader.
        self._connected = False
        self._ever_connected = False
        self._attempt_task: asyncio.Task[Exception | None] | None = None
        self._last_error: Exception | None = None
        self._shutting_down = False

    def start(self) -> None:
        """Start the reconnect supervisor. Never awaited by the caller --
        see the class docstring."""
        self._consumer_task = asyncio.create_task(self._supervise())

    async def frames(self) -> AsyncIterator[bytes]:
        """Yield the camera's own undecoded packet bytes until this source
        is closed. A dropped-and-reconnected link is invisible here -- see
        the module docstring's sentinel-timing note."""
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

    def sink_format(self) -> SinkFormat:
        """The codec and sample rate the camera speaker actually plays.

        260922-cts: this is the transport declaring its own playback
        format, the same move `source_format()` above already makes for
        the microphone side -- `turn/controller.py`'s `_speak` reads this
        with `getattr(source, "sink_format", None)`, duck-typed exactly
        like `source.barge_in`/`source.send_event` already are, so a
        browser or WebRTC source with no `sink_format` at all still gets
        today's browser-PCM default. `CameraConfig` grows no field of its
        own for this -- `app.py` passes the real `config.tts.codec`/
        `config.tts.sample_rate` pair in at construction (`sink=`), which
        is what the module docstring's Fix section calls "a transport
        declares the audio format it plays," not something this class
        derives from its own microphone-format config.
        """
        return self._sink

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

    async def reconnect(self) -> None:
        """Establish the connection if this source is not already
        connected, holding to exactly one attempt at a time (T-02-29).

        A no-op while already connected: a supervisor retry racing a link
        that recovered on its own must not leave two readers on one
        camera. A no-op while another attempt is already in flight: a
        second request during that window waits for nothing and starts
        nothing, rather than opening a second container. Both checks run
        with no `await` between them, so two concurrent calls can never
        interleave past this method's own guard -- asyncio only switches
        tasks at an `await` point.
        """
        if self._connected:
            return
        if self._attempt_task is not None and not self._attempt_task.done():
            return
        self._attempt_task = asyncio.ensure_future(self._attempt_once())
        try:
            await self._attempt_task
        finally:
            self._attempt_task = None

    async def close(self) -> None:
        """End the reconnect supervisor and release the container, best
        effort.

        Shutdown is checked at the top of `_supervise()`'s loop and again
        after its backoff wait (T-02-30): a source torn down mid-attempt
        ends that attempt rather than reviving itself. The consumer task is
        cancelled and awaited *before* this method ever touches the
        container -- `container.close()` from this (the event loop) thread
        while `_read_loop`'s worker thread is still stepping its own
        `container.demux()` generator is a real concurrent-access hazard
        against PyAV's C extension, not merely a style choice (a genuine
        segfault, reproduced this session). `_read_loop`'s own `finally`
        block closes the container itself once the worker thread is done
        with it, from that same thread -- this method's own close call
        below is only for the case a container was left behind with no
        worker thread still running against it at all (a mid-open failure,
        or `_read_loop` already having returned).
        """
        self._shutting_down = True
        if self._consumer_task is not None:
            self._consumer_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._consumer_task
        if self._container is not None:
            with contextlib.suppress(Exception):
                self._container.close()

    async def _supervise(self) -> None:
        """The reconnect loop: one `reconnect()` attempt, a log line naming
        what ended it and how long until the next one, then a cancellable
        backoff wait -- until `close()` sets `_shutting_down`, or until an
        attempt ends with no exception at all.

        A clean, exception-free end of `container.demux()` is not a drop --
        for a live RTSP stream it practically never happens (the network
        simply stays open, or an error interrupts it); it is what a
        genuinely finite source (a test fixture file, PROV-07's own
        byte-identity tests) looks like once it legitimately has nothing
        left to give. Retrying that would be wrong twice over: it is not a
        failure to recover from, and it would turn every one of those
        tests into an infinite retry loop against the very fixture that
        already proved its point. Only an exception -- the open call
        failing, or `demux()` raising mid-read -- counts as a drop worth
        retrying (T-02-28).
        """
        try:
            while not self._shutting_down:
                await self.reconnect()
                if self._shutting_down:
                    return
                exc = self._last_error
                if exc is None:
                    return
                logger.warning(
                    "camera RTSP source disconnected (%s); reconnecting in %.1fs",
                    type(exc).__name__,
                    self._backoff_s,
                )
                await self._sleep(self._backoff_s)
                if self._shutting_down:
                    return
        finally:
            # Queued exactly once, here, when the supervisor itself ends --
            # never per dropped attempt (module docstring). `_supervise`
            # already runs on the event loop thread (only `_read_loop` runs
            # in a worker thread), so no `call_soon_threadsafe` is needed.
            self._queue.put_nowait(None)

    async def _attempt_once(self) -> Exception | None:
        """One connect-and-drain attempt, off the event loop thread
        (`container.demux()` blocks). Records whatever ended it -- an
        exception, or `None` for a clean stream end -- for `_supervise()`'s
        own retry log line."""
        loop = asyncio.get_running_loop()
        self._last_error = await asyncio.to_thread(self._read_loop, loop)
        return self._last_error

    def _fire_on_reconnect(self) -> None:
        """Runs on the event loop thread via `call_soon_threadsafe`,
        scheduled only for a genuine *re*-connect (never the first-ever
        connect): fires `on_reconnect` -- the speaker's backchannel needs
        re-establishing here too, not only at startup, since whatever
        dropped the camera link may have taken the receiving service's
        producer for the speaker down with it. Scheduling this alone
        through `call_soon_threadsafe`, and never `self._connected` itself
        (see `_read_loop`), is deliberate: `self._connected` must flip
        True and back to False in strict program order on the *same*
        thread that owns the attempt, or a fast-failing attempt's own
        `finally` could reset it to False before a delayed callback ever
        set it True, leaving it stuck True forever (reproduced this
        session as a tight, CPU-pinned `reconnect()` no-op loop).
        """
        if self._on_reconnect is not None:
            asyncio.ensure_future(self._on_reconnect())

    def _read_loop(self, loop: asyncio.AbstractEventLoop) -> Exception | None:
        """Runs in a worker thread: `container.demux()` is a blocking
        call, and bridging it onto the event loop's queue happens through
        `call_soon_threadsafe` rather than blocking that loop directly.

        Returns the exception that ended this attempt (the open itself
        failed, or the stream dropped mid-read), or `None` for a clean end
        -- `_supervise()` logs whichever it was and retries either way.
        """
        try:
            container = self._open_container(self._config)
        except Exception as exc:  # noqa: BLE001 -- any open failure is retried by the supervisor (T-02-28), never crashes lifespan
            # Returned, not logged here: `_supervise()` logs the exception's
            # TYPE only, never its message or `self._config.rtsp_url` --
            # both may carry the camera's embedded RTSP credentials (module
            # docstring) -- and it is the one place that already logs every
            # retry, open failure or mid-stream drop alike.
            return exc

        self._container = container
        # Both set here, directly, on this same worker thread -- not via
        # `call_soon_threadsafe` -- so this attempt's own `finally` below
        # can never race a delayed event-loop callback (see
        # `_fire_on_reconnect`'s docstring for the hazard this avoids).
        self._connected = True
        if self._ever_connected:
            loop.call_soon_threadsafe(self._fire_on_reconnect)
        self._ever_connected = True
        try:
            audio_stream = container.streams.audio[0]
            self._detector_codec_context = self._build_detector_codec_context(audio_stream)
            for packet in container.demux(audio_stream):
                if packet.dts is None:
                    continue  # the end-of-stream flush packet, not audio
                raw = bytes(packet)
                loop.call_soon_threadsafe(self._queue.put_nowait, raw)
            return None
        except Exception as exc:  # noqa: BLE001 -- a mid-stream drop is retried by the supervisor (T-02-28)
            return exc
        finally:
            self._connected = False
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
