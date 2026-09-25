"""Edge protocol v1, server side (D-01, D-02, D-06, D-09, Phase 10).

`EdgeAudioSource` is the fifth `AudioSource` implementation
(`transports/base.py`) -- a Raspberry Pi's reSpeaker XVF3800 array, dialing
in over one always-on WebSocket rather than the server dialing out. It
follows the camera's own lifecycle shape (`transports/camera.py`'s module
docstring): built once at startup, held for the process's life, with
`frames()` backed by one `asyncio.Queue` that only `close()` ever puts a
`None` sentinel onto.

Two paths leave one `serve()` loop, mirroring the camera's own raw-path
vs. detector-path split: `frames()` yields every interleaved byte the Pi
sends, both capture channels, for the session recorder (D-09, D-17);
`decode_for_detector()` is a second, independent de-interleave of the same
bytes, handing the wake detector only the configured ASR channel. Neither
path mutates or substitutes for the other.

This task builds the happy path: a hello on connect, `vad.start`/`vad.end`
text events, and binary audio frames. Task 2 adds the rest of protocol
v1 -- reconnect/rival-device handling, size and rate bounds, and the full
event schema (`doa`, `latency`, `pong`).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from typing import Any, AsyncIterator, Callable

from atlas.audio.channels import select_channel
from atlas.config import EdgeSourceConfig
from atlas.providers.tts_xai import SinkFormat
from atlas.transports.base import SourceFormat

logger = logging.getLogger("atlas.transports.edge")

PROTOCOL_VERSION = 1

# Message type names -- the wire vocabulary both `edge.py` (here) and
# `edge/src/atlas_edge/protocol.py` (the Pi side) restate independently;
# `tests/test_edge_protocol_contract.py` (Task 2) pins the two against
# each other.
MSG_HELLO = "hello"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_VAD_START = "vad.start"
MSG_VAD_END = "vad.end"
MSG_DOA = "doa"
MSG_LATENCY = "latency"

# 16 ms of capture per frame at 16 kHz -- keeps Pi-side buffering well
# under the project's latency budget and divides the 512-sample Silero
# window exactly (Claude's discretion, per 10-CONTEXT.md).
FRAME_SAMPLES = 256

# Close codes protocol v1 defines. `CLOSE_POLICY_VIOLATION` is the
# standard WebSocket 1008 code -- used when `require_edge_device`
# (`auth/edge_tokens.py`) refuses a token, and (Task 2) when a connection
# sends too many malformed messages. The other three are private-range
# codes (RFC 6455 4000-4999) specific to this protocol.
CLOSE_POLICY_VIOLATION = 1008
CLOSE_SUPERSEDED = 4000  # a newer connection from the same device replaced this one.
CLOSE_NOT_CONFIGURED = 4003  # /ws/edge is not the server's configured audio source.
CLOSE_BUSY = 4009  # a different device is already connected.


class EdgeProtocolError(ValueError):
    """Raised by `parse_edge_event` for any text frame that is not valid
    JSON, is not an object, names an unsupported `type`, or fails that
    type's own field validation."""


def parse_edge_event(text: str) -> dict[str, Any]:
    """Parse one JSON text frame from the Pi.

    This task accepts exactly `vad.start` and `vad.end`, each carrying an
    integer `seq >= 0`, and raises `EdgeProtocolError` for anything else.
    Task 2 extends this to the full v1 schema (`doa`, `latency`, `pong`).
    Returns a new, normalized dict holding only the known keys for the
    message's own type -- never the raw parsed object, so an unknown extra
    key in the wire message is silently dropped rather than carried
    forward into `speech_signals.publish`.
    """
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise EdgeProtocolError(f"edge event is not valid JSON: {text!r}") from exc
    if not isinstance(raw, dict):
        raise EdgeProtocolError(f"edge event must be a JSON object, got {raw!r}")

    msg_type = raw.get("type")
    if msg_type in (MSG_VAD_START, MSG_VAD_END):
        seq = raw.get("seq")
        if isinstance(seq, bool) or not isinstance(seq, int) or seq < 0:
            raise EdgeProtocolError(f"{msg_type} event needs an integer seq >= 0, got {seq!r}")
        return {"type": msg_type, "seq": seq}

    raise EdgeProtocolError(f"unsupported edge event type: {msg_type!r}")


def build_hello(config: EdgeSourceConfig, device_id: int) -> str:
    """The one `hello` message `EdgeAudioSource.serve` sends once, right
    after accept -- carries the server's own measured `asr_channel`,
    `pre_roll_ms`, and `tail_ms` (D-08), never a default the Pi would have
    to guess. Called only after `config.require_measured()` has already
    passed (at `EdgeAudioSource.__init__`), so every field here is a real
    int, never `None`.
    """
    return json.dumps(
        {
            "type": MSG_HELLO,
            "protocol": PROTOCOL_VERSION,
            "device_id": device_id,
            "sample_rate": config.sample_rate,
            "channels": config.channels,
            "asr_channel": config.asr_channel,
            "pre_roll_ms": config.pre_roll_ms,
            "tail_ms": config.tail_ms,
            "frame_samples": FRAME_SAMPLES,
        }
    )


class SpeechSignals:
    """The one place `vad.start`/`vad.end` (and, Task 2, `doa`/`latency`)
    fan out to every interested listener -- `SourceRunner`'s wake gate
    reads `in_speech` and a later plan's early-finalize hook subscribes.

    A `vad.start` begins a new bounded segment history (`max_segment_
    events`, oldest dropped first): a subscriber that joins mid-segment
    (`subscribe(..., replay_segment=True)`, the default) first receives
    that segment's history, so it never misses the events that already
    happened before it existed. A subscriber callback that raises is
    logged and kept -- one bad listener must never stop delivery to the
    others.
    """

    def __init__(self, hangover_s: float, max_segment_events: int = 64) -> None:
        self.hangover_s = hangover_s
        self._segment_events: "deque[dict[str, Any]]" = deque(maxlen=max_segment_events)
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._in_speech = False

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def publish(self, event: dict[str, Any]) -> None:
        if event.get("type") == MSG_VAD_START:
            self._segment_events.clear()
            self._in_speech = True
        self._segment_events.append(event)
        if event.get("type") == MSG_VAD_END:
            self._in_speech = False
        for callback in list(self._subscribers):
            try:
                callback(event)
            except Exception:
                logger.exception("edge speech_signals subscriber raised; keeping it subscribed")

    def subscribe(
        self, callback: Callable[[dict[str, Any]], None], *, replay_segment: bool = True
    ) -> Callable[[], None]:
        self._subscribers.append(callback)
        if replay_segment:
            for event in list(self._segment_events):
                try:
                    callback(event)
                except Exception:
                    logger.exception("edge speech_signals subscriber raised during replay; keeping it subscribed")

        def _unsubscribe() -> None:
            try:
                self._subscribers.remove(callback)
            except ValueError:
                pass  # already unsubscribed -- idempotent by design.

        return _unsubscribe


class EdgeAudioSource:
    """Satisfies `AudioSource` over one always-on `/ws/edge` connection.

    Built once at startup, like `CameraAudioSource` (`transports/camera.py`
    module docstring) -- never per-connection. `frames()`'s queue lives for
    the process's whole life; a Pi that disconnects and reconnects (Task 2)
    resumes writing into the same queue, the same "a dropped-and-reconnected
    link is invisible to `frames()`" contract the camera already holds.
    """

    def __init__(self, config: EdgeSourceConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        # D-18/Pitfall 3: refuses construction until the spike's numbers
        # (asr_channel, pre_roll_ms, tail_ms) are configured -- never a
        # guessed channel index.
        config.require_measured()
        self._config = config
        self._clock = clock
        self._frames_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue()
        self._websocket: Any | None = None
        self._connected_device_id: int | None = None
        self.last_rtt_ms: float | None = None
        self._speech_signals = SpeechSignals(config.end_of_speech_hangover_ms / 1000.0)

    @property
    def speech_signals(self) -> SpeechSignals:
        return self._speech_signals

    @property
    def connected_device_id(self) -> int | None:
        return self._connected_device_id

    async def frames(self) -> AsyncIterator[bytes]:
        while True:
            chunk = await self._frames_queue.get()
            if chunk is None:
                return
            yield chunk

    async def send_audio(self, chunk: bytes) -> None:
        """Reply audio for an edge turn goes back on the same socket
        (D-14) -- a no-op when no device is connected."""
        if self._websocket is not None:
            await self._websocket.send_bytes(chunk)

    async def send_event(self, event: dict[str, Any]) -> None:
        """No-op, like the camera's own `send_event` (`transports/
        camera.py`): a Pi has nowhere to render a partial transcript."""
        logger.debug("edge source send_event no-op: %s", event.get("type"))

    def source_format(self) -> SourceFormat:
        return SourceFormat(
            "pcm",
            self._config.sample_rate,
            channels=self._config.channels,
            asr_channel=self._config.asr_channel,
        )

    def sink_format(self) -> SinkFormat:
        """Mono PCM16 at the configured sample rate -- the Pi turns this
        back into two channels for the array's playback device (D-14)."""
        return SinkFormat("pcm", self._config.sample_rate)

    def decode_for_detector(self, chunk: bytes) -> bytes:
        """A second, independent de-interleave of `chunk` for the wake
        detector alone (D-09) -- never a mutation of, or substitute for,
        the raw bytes `frames()` already yielded to the session recorder."""
        return select_channel(chunk, self._config.channels, self._config.asr_channel)

    async def serve(self, websocket: Any, device: Any) -> None:
        """Send the hello, then loop on `websocket.receive()` until the
        Pi disconnects. Binary frames go onto the queue; `vad.start`/
        `vad.end` text frames go through `parse_edge_event` into
        `speech_signals.publish`. The caller (`app.py`'s `/ws/edge` route)
        has already called `websocket.accept()`.
        """
        await websocket.send_text(build_hello(self._config, device.id))
        self._websocket = websocket
        self._connected_device_id = device.id
        try:
            while True:
                message = await websocket.receive()
                if message["type"] == "websocket.disconnect":
                    return
                data = message.get("bytes")
                if data is not None:
                    self._frames_queue.put_nowait(data)
                    continue
                text = message.get("text")
                if text is None:
                    continue
                try:
                    event = parse_edge_event(text)
                except EdgeProtocolError:
                    logger.debug("edge source dropped a malformed text frame", exc_info=True)
                    continue
                if event["type"] in (MSG_VAD_START, MSG_VAD_END):
                    self._speech_signals.publish(event)
        finally:
            self._websocket = None
            self._connected_device_id = None

    async def close(self) -> None:
        """End `frames()` for good -- the `None` sentinel this class ever
        queues, matching `CameraAudioSource.close`'s own "exactly once, at
        shutdown" discipline."""
        self._frames_queue.put_nowait(None)
