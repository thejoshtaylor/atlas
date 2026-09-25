"""Edge protocol v1, server side (D-01, D-02, D-06, D-09, D-13, Phase 10).

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

Task 1 built the happy path: a hello on connect, `vad.start`/`vad.end`
text events, and binary audio frames. Task 2 (this revision) adds the
rest of protocol v1: reconnect/rival-device handling, size and rate
bounds against a hostile or merely buggy peer, and the full event schema
(`doa`, `latency`, `pong`).

**Connection identity (Task 2).** `EdgeAudioSource` tracks the
`asyncio.Task` running the current `serve()` call, not just its socket and
device id -- a second connection from the *same* device cancels that task
after closing its old socket with `CLOSE_SUPERSEDED` (4000), and the
cancelled call's own `finally` block checks whether it is still the
active connection before touching any shared state, so a superseded call
racing its own cleanup can never clobber the connection that replaced it.
A connection from a *different* device while one is already active is
refused with `CLOSE_BUSY` (4009) after accept, leaving the first
connection completely untouched -- refusal is the new connection's own
`serve()` returning immediately, never a cancellation of anything.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
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
# `tests/test_edge_protocol_contract.py` pins the two against each other.
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
# (`auth/edge_tokens.py`) refuses a token, and when a connection sends too
# many malformed messages (`MAX_INVALID_MESSAGES` below). The other three
# are private-range codes (RFC 6455 4000-4999) specific to this protocol.
CLOSE_POLICY_VIOLATION = 1008
CLOSE_SUPERSEDED = 4000  # a newer connection from the same device replaced this one.
CLOSE_NOT_CONFIGURED = 4003  # /ws/edge is not the server's configured audio source.
CLOSE_BUSY = 4009  # a different device is already connected.

# Every bound a network peer's own input controls, each sized against a
# concrete reason rather than a round number:
#
# A binary frame this large already exceeds several multiples of one
# 256-sample/2-channel/16-bit capture block (1024 bytes) -- large enough
# that a legitimate sender never approaches it, small enough that a flood
# of oversize frames cannot build unbounded server-side buffers.
MAX_AUDIO_FRAME_BYTES = 8192
# Every real v1 text message (`<interfaces>`) is well under 300 bytes even
# with a full 8-element `doa` payload; 2048 leaves headroom for a verbose
# future field without opening the door to an arbitrarily large text frame.
MAX_TEXT_FRAME_BYTES = 2048
# A connection sending this many malformed messages is not a peer with a
# transient bug -- it is hostile or broken beyond usefulness, and 1008
# ends the connection rather than looping forever on garbage.
MAX_INVALID_MESSAGES = 20
# ~16 seconds of buffered 16ms frames (1024 * 16ms) -- far more than any
# turn's own drain latency should ever need; a queue this deep that is
# still growing means nothing is reading it, and dropping the oldest
# frame is the bounded response (D-13: a lost frame never hangs a turn).
MAX_QUEUED_FRAMES = 1024
# `SpeechSignals`' own bounded segment history -- long enough that a
# subscriber joining mid-segment sees a real segment's worth of events,
# short enough that a segment that never ends cannot grow this without
# bound.
MAX_SEGMENT_EVENTS = 64
# The XVF3800 in this phase's 2-channel firmware reports at most a
# handful of beams; 8 is a firmware-agnostic ceiling matching the `doa`
# schema's own documented range, never a number this file derives from
# one specific firmware build.
MAX_DOA_VALUES = 8
# Keep only the latest few outstanding ping ids (10-07-PLAN.md) -- a Pi
# that never answers a ping cannot grow this without bound, the same
# `deque(maxlen=...)` discipline `SpeechSignals`' own segment history
# already uses for the identical reason.
MAX_OUTSTANDING_PINGS = 8
# The ROADMAP's Phase 10 bullet: "less than 50 ms of added delay" --
# `added_delay_ms_p95` is measured against this budget with no shared
# clock needed (10-CONTEXT.md Open Question 1): the Pi's own measured
# capture-to-send p95 plus half the server-measured ping round trip.
ADDED_DELAY_BUDGET_MS = 50


class EdgeProtocolError(ValueError):
    """Raised by `parse_edge_event` for any text frame that is not valid
    JSON, is not an object, names an unsupported `type`, or fails that
    type's own field validation."""


def _require_finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise EdgeProtocolError(f"{name} must be a number, got {value!r}")
    number = float(value)
    if not math.isfinite(number):
        raise EdgeProtocolError(f"{name} must be finite, got {value!r}")
    return number


def _require_range(value: float, name: str, *, lo: float, hi: "float | None", hi_exclusive: bool = False) -> None:
    if value < lo:
        raise EdgeProtocolError(f"{name} must be >= {lo}, got {value!r}")
    if hi is None:
        return
    if hi_exclusive:
        if value >= hi:
            raise EdgeProtocolError(f"{name} must be < {hi}, got {value!r}")
    elif value > hi:
        raise EdgeProtocolError(f"{name} must be <= {hi}, got {value!r}")


def _require_finite_number_list(
    values: Any,
    name: str,
    *,
    min_len: int,
    max_len: int,
    lo: float,
    hi: "float | None" = None,
    hi_exclusive: bool = False,
) -> "list[float]":
    if not isinstance(values, list):
        raise EdgeProtocolError(f"{name} must be a list, got {values!r}")
    if not (min_len <= len(values) <= max_len):
        raise EdgeProtocolError(f"{name} must have between {min_len} and {max_len} elements, got {len(values)}")
    result: "list[float]" = []
    for index, raw_value in enumerate(values):
        number = _require_finite_number(raw_value, f"{name}[{index}]")
        _require_range(number, f"{name}[{index}]", lo=lo, hi=hi, hi_exclusive=hi_exclusive)
        result.append(number)
    return result


def _require_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise EdgeProtocolError(f"{name} must be an integer, got {value!r}")
    return value


def parse_edge_event(text: str) -> dict[str, Any]:
    """Parse one JSON text frame from the Pi into a normalized dict
    holding only the known keys for its own `type` -- never the raw
    parsed object, so an unknown extra key in the wire message is
    silently dropped rather than carried forward into
    `speech_signals.publish`.

    Accepts all five Pi-to-server message types (`<interfaces>`):
    `vad.start`/`vad.end` (integer `seq >= 0`), `doa` (1-8 finite
    azimuths in `[0, 360)`, 0-8 finite non-negative energies), `latency`
    (three finite values in `[0, 60000]` plus `frames >= 0`), and `pong`
    (integer `id` and `server_t_ms`). Raises `EdgeProtocolError` for
    anything else: invalid JSON, a non-object, an unknown `type`, or any
    field that is the wrong type, out of range, non-finite (`NaN`/
    `Infinity` -- valid JSON extensions Python's own parser accepts), or
    an oversize list.
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
            raise EdgeProtocolError(f"{msg_type}.seq must be an integer >= 0, got {seq!r}")
        return {"type": msg_type, "seq": seq}

    if msg_type == MSG_DOA:
        azimuth_deg = _require_finite_number_list(
            raw.get("azimuth_deg"), "doa.azimuth_deg", min_len=1, max_len=MAX_DOA_VALUES, lo=0.0, hi=360.0, hi_exclusive=True
        )
        speech_energy = _require_finite_number_list(
            raw.get("speech_energy", []), "doa.speech_energy", min_len=0, max_len=MAX_DOA_VALUES, lo=0.0
        )
        return {"type": msg_type, "azimuth_deg": azimuth_deg, "speech_energy": speech_energy}

    if msg_type == MSG_LATENCY:
        p50 = _require_finite_number(raw.get("capture_to_send_ms_p50"), "latency.capture_to_send_ms_p50")
        _require_range(p50, "latency.capture_to_send_ms_p50", lo=0.0, hi=60000.0)
        p95 = _require_finite_number(raw.get("capture_to_send_ms_p95"), "latency.capture_to_send_ms_p95")
        _require_range(p95, "latency.capture_to_send_ms_p95", lo=0.0, hi=60000.0)
        max_ms = _require_finite_number(raw.get("capture_to_send_ms_max"), "latency.capture_to_send_ms_max")
        _require_range(max_ms, "latency.capture_to_send_ms_max", lo=0.0, hi=60000.0)
        frames = raw.get("frames")
        if isinstance(frames, bool) or not isinstance(frames, int) or frames < 0:
            raise EdgeProtocolError(f"latency.frames must be an integer >= 0, got {frames!r}")
        return {
            "type": msg_type,
            "capture_to_send_ms_p50": p50,
            "capture_to_send_ms_p95": p95,
            "capture_to_send_ms_max": max_ms,
            "frames": frames,
        }

    if msg_type == MSG_PONG:
        ping_id = _require_int(raw.get("id"), "pong.id")
        server_t_ms = _require_int(raw.get("server_t_ms"), "pong.server_t_ms")
        return {"type": msg_type, "id": ping_id, "server_t_ms": server_t_ms}

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
    """The one place `vad.start`/`vad.end`/`doa`/`latency` fan out to
    every interested listener -- `SourceRunner`'s wake gate reads
    `in_speech` and a later plan's early-finalize hook subscribes. `pong`
    never reaches here (`EdgeAudioSource._handle_event` routes it to
    `last_rtt_ms` instead) -- it answers this server's own keepalive, not
    a speech-shaped event anything downstream should see.

    A `vad.start` begins a new bounded segment history (`max_segment_
    events`, oldest dropped first): a subscriber that joins mid-segment
    (`subscribe(..., replay_segment=True)`, the default) first receives
    that segment's history, so it never misses the events that already
    happened before it existed. A subscriber callback that raises is
    logged and kept -- one bad listener must never stop delivery to the
    others.
    """

    def __init__(self, hangover_s: float, max_segment_events: int = MAX_SEGMENT_EVENTS) -> None:
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
    the process's whole life; a Pi that disconnects and reconnects resumes
    writing into the same queue, the same "a dropped-and-reconnected link
    is invisible to `frames()`" contract the camera already holds. Only
    `close()` ever queues the `None` sentinel -- a disconnect, a
    supersede, or a rival refusal never does (module docstring).
    """

    def __init__(self, config: EdgeSourceConfig, *, clock: Callable[[], float] = time.monotonic) -> None:
        # D-18/Pitfall 3: refuses construction until the spike's numbers
        # (asr_channel, pre_roll_ms, tail_ms) are configured -- never a
        # guessed channel index.
        config.require_measured()
        self._config = config
        self._clock = clock
        self._frames_queue: "asyncio.Queue[bytes | None]" = asyncio.Queue(maxsize=MAX_QUEUED_FRAMES)
        self._websocket: Any | None = None
        self._connected_device_id: int | None = None
        self._active_task: "asyncio.Task[None] | None" = None
        self.last_rtt_ms: float | None = None
        self._last_seq: int = 0
        self._speech_signals = SpeechSignals(config.end_of_speech_hangover_ms / 1000.0)
        # Warn-once-per-episode flags, the same discipline `sources/
        # runner.py`'s own `_pending_writes_warned` already uses for its
        # own saturation case.
        self._queue_saturation_warned = False
        self._send_audio_warned = False
        # Per-connection ping/added-delay state (10-07-PLAN.md) -- reset at
        # the top of `serve()` every time a connection becomes the active
        # one, so a reconnect never carries a stale outstanding-ping id or
        # a previous connection's own worst/last delay figure forward.
        self._outstanding_pings: "deque[int]" = deque(maxlen=MAX_OUTSTANDING_PINGS)
        self._next_ping_id: int = 0
        self._connection_worst_added_delay_ms: float | None = None
        self._connection_last_added_delay_ms: float | None = None
        self._added_delay_over_budget_warned = False

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
        (D-14) -- dropped, with one warning per disconnected episode, when
        no device is connected."""
        if self._websocket is None:
            if not self._send_audio_warned:
                self._send_audio_warned = True
                logger.warning("edge source: send_audio with no device connected -- dropping the reply chunk")
            return
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
        Pi disconnects, a newer connection from the same device supersedes
        this one, or too many malformed messages close it. The caller
        (`app.py`'s `/ws/edge` route) has already called
        `websocket.accept()`.

        A second connection for the device already connected closes the
        old socket with `CLOSE_SUPERSEDED` and cancels its still-running
        `serve()` call before this one takes over. A connection for a
        *different* device while one is active is refused with
        `CLOSE_BUSY` at once -- no hello, no state change, the existing
        connection untouched.
        """
        previous_websocket = self._websocket
        previous_device_id = self._connected_device_id
        previous_task = self._active_task

        if previous_device_id is not None and previous_device_id != device.id:
            logger.warning(
                "edge source: device %s refused -- device %s is already connected",
                device.id,
                previous_device_id,
            )
            await websocket.close(code=CLOSE_BUSY)
            return

        if previous_device_id == device.id and previous_websocket is not None:
            await previous_websocket.close(code=CLOSE_SUPERSEDED)
            if previous_task is not None:
                previous_task.cancel()

        self._active_task = asyncio.current_task()
        self._websocket = websocket
        self._connected_device_id = device.id
        self._send_audio_warned = False
        invalid_message_count = 0
        # 10-07-PLAN.md: fresh per connection -- a reconnect never inherits
        # a previous connection's own outstanding ping ids or delay figures.
        self._outstanding_pings = deque(maxlen=MAX_OUTSTANDING_PINGS)
        self._next_ping_id = 0
        self._connection_worst_added_delay_ms = None
        self._connection_last_added_delay_ms = None
        self._added_delay_over_budget_warned = False

        await websocket.send_text(build_hello(self._config, device.id))
        ping_task = asyncio.create_task(self._ping_loop(websocket))
        try:
            try:
                while True:
                    message = await websocket.receive()
                    if message["type"] == "websocket.disconnect":
                        return
                    data = message.get("bytes")
                    if data is not None:
                        if len(data) > MAX_AUDIO_FRAME_BYTES or not self._is_whole_number_of_frames(data):
                            invalid_message_count += 1
                            if invalid_message_count >= MAX_INVALID_MESSAGES:
                                await websocket.close(code=CLOSE_POLICY_VIOLATION)
                                return
                            continue
                        self._enqueue_frame(data)
                        continue
                    text = message.get("text")
                    if text is None:
                        continue
                    if len(text.encode("utf-8")) > MAX_TEXT_FRAME_BYTES:
                        invalid_message_count += 1
                        if invalid_message_count >= MAX_INVALID_MESSAGES:
                            await websocket.close(code=CLOSE_POLICY_VIOLATION)
                            return
                        continue
                    try:
                        event = parse_edge_event(text)
                    except EdgeProtocolError:
                        logger.debug("edge source dropped a malformed text frame", exc_info=True)
                        invalid_message_count += 1
                        if invalid_message_count >= MAX_INVALID_MESSAGES:
                            await websocket.close(code=CLOSE_POLICY_VIOLATION)
                            return
                        continue
                    if not self._handle_event(event):
                        # T-10-25: an unmatched pong is the one event
                        # `_handle_event` can call invalid post-parse --
                        # counted the same way a malformed text frame is.
                        invalid_message_count += 1
                        if invalid_message_count >= MAX_INVALID_MESSAGES:
                            await websocket.close(code=CLOSE_POLICY_VIOLATION)
                            return
            finally:
                # A superseded connection's own cancellation lands here too --
                # `self._active_task` already names the connection that
                # replaced it by the time this runs (module docstring), so
                # this guard is what keeps a superseded call's cleanup from
                # clobbering the connection that is now active.
                if self._active_task is asyncio.current_task():
                    if self._speech_signals.in_speech:
                        # D-13: a lost `vad.end` (a dropped connection mid-
                        # segment) never hangs a turn -- ending the segment
                        # synthetically is what "the fallbacks hold even
                        # without one" means in practice.
                        self._speech_signals.publish(
                            {"type": MSG_VAD_END, "seq": self._last_seq, "reason": "disconnected"}
                        )
                    self._websocket = None
                    self._connected_device_id = None
                    self._active_task = None
        finally:
            # 10-07-PLAN.md: cancelled regardless of supersede/rival
            # outcome -- this ping task belongs to this specific `serve()`
            # call, never the shared connection state the guard above
            # protects, so it is always torn down when this call ends.
            ping_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await ping_task
            logger.info(
                "edge source: connection ended (device_id=%s, worst_added_delay_ms_p95=%s, "
                "last_added_delay_ms_p95=%s)",
                device.id,
                self._connection_worst_added_delay_ms,
                self._connection_last_added_delay_ms,
            )

    async def _ping_loop(self, websocket: Any) -> None:
        """Send `ping` every `config.ping_interval_s`, cancelled with the
        connection (10-07-PLAN.md). `server_t_ms` is an integer -- the Pi
        echoes it back verbatim in its own `pong`, and `parse_edge_event`'s
        `pong` schema requires an integer, so a float sent here would make
        a real device's own honest echo fail parsing on the way back."""
        while True:
            await asyncio.sleep(self._config.ping_interval_s)
            ping_id = self._next_ping_id
            self._next_ping_id += 1
            self._outstanding_pings.append(ping_id)
            server_t_ms = int(round(self._clock() * 1000.0))
            await websocket.send_text(json.dumps({"type": MSG_PING, "id": ping_id, "server_t_ms": server_t_ms}))

    async def disconnect_device(self, device_id: int, *, code: int, reason: str) -> None:
        """Close the live connection for `device_id` at once, if one is
        active -- a no-op for any other device or when nothing is
        connected (plan 10-04, D-03: revoking a device must not merely
        refuse its *next* reconnect, T-10-12).

        Cancels the same `_active_task` the supersede path in `serve()`
        cancels, so the closed connection's own `finally` block runs
        exactly once, the same "still the active connection at cleanup
        time" guard that path already relies on."""
        if self._connected_device_id != device_id or self._websocket is None:
            return
        websocket = self._websocket
        active_task = self._active_task
        await websocket.close(code=code, reason=reason)
        if active_task is not None:
            active_task.cancel()

    async def close(self) -> None:
        """End `frames()` for good -- the `None` sentinel this class ever
        queues, matching `CameraAudioSource.close`'s own "exactly once, at
        shutdown" discipline. Drops the oldest queued frame first if the
        queue is already at its bound, so the sentinel always gets in."""
        try:
            self._frames_queue.put_nowait(None)
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._frames_queue.get_nowait()
            self._frames_queue.put_nowait(None)

    def _is_whole_number_of_frames(self, data: bytes) -> bool:
        frame_bytes = 2 * self._config.channels
        return frame_bytes > 0 and len(data) % frame_bytes == 0

    def _enqueue_frame(self, data: bytes) -> None:
        """Push `data` onto the frame queue, dropping the oldest frame
        first if it is already at `MAX_QUEUED_FRAMES` -- a new frame is
        never refused, an old one is discarded instead (D-13's "a lost
        frame never hangs a turn," applied to backpressure rather than a
        dropped connection). One warning per saturation episode, reset the
        moment the queue next has room -- `sources/runner.py`'s own
        `_pending_writes_warned` precedent."""
        try:
            self._frames_queue.put_nowait(data)
            self._queue_saturation_warned = False
        except asyncio.QueueFull:
            with contextlib.suppress(asyncio.QueueEmpty):
                self._frames_queue.get_nowait()
            self._frames_queue.put_nowait(data)
            if not self._queue_saturation_warned:
                self._queue_saturation_warned = True
                logger.warning(
                    "edge source: frame queue saturated at %d frames; dropping the oldest",
                    MAX_QUEUED_FRAMES,
                )

    def _handle_event(self, event: dict[str, Any]) -> bool:
        """Route one parsed event: `vad.start`/`vad.end`/`doa` publish to
        `speech_signals` unchanged; `latency` is enriched with `rtt_ms`/
        `added_delay_ms_p95` first (`_publish_latency_event`, 10-07-PLAN.md).
        `pong` never reaches `speech_signals` (`SpeechSignals`' own
        docstring) -- it updates `last_rtt_ms` instead, through
        `_handle_pong`.

        Returns `False` for the one case that counts as malformed after
        successfully parsing (T-10-25): a `pong` whose `id` matches no
        outstanding ping. `serve()`'s own loop counts that the same way it
        counts a malformed text frame. Every other event type always
        returns `True`.
        """
        msg_type = event["type"]
        if msg_type in (MSG_VAD_START, MSG_VAD_END):
            self._last_seq = event["seq"]
            self._speech_signals.publish(event)
        elif msg_type == MSG_DOA:
            self._speech_signals.publish(event)
        elif msg_type == MSG_LATENCY:
            self._publish_latency_event(event)
        elif msg_type == MSG_PONG:
            return self._handle_pong(event)
        return True

    def _handle_pong(self, event: dict[str, Any]) -> bool:
        """Match `event["id"]` against `self._outstanding_pings`; on a
        match, remove it and set `last_rtt_ms` from this server's own
        `server_t_ms` echoed back. An unmatched id never changes
        `last_rtt_ms` and is reported invalid (T-10-25)."""
        ping_id = event["id"]
        try:
            self._outstanding_pings.remove(ping_id)
        except ValueError:
            return False
        now_ms = self._clock() * 1000.0
        self.last_rtt_ms = max(0.0, now_ms - event["server_t_ms"])
        return True

    def _publish_latency_event(self, event: dict[str, Any]) -> None:
        """Enrich `event` with `rtt_ms` (the last matched RTT, or `None`
        without one -- never guessed) and `added_delay_ms_p95` (the Pi's
        own `capture_to_send_ms_p95` plus half that RTT, or `None` when
        `rtt_ms` is `None`) before publishing, and track this
        connection's own worst/last `added_delay_ms_p95` (10-07-PLAN.md,
        ROADMAP Phase 10's 50 ms budget). One warning per connection when
        the budget is exceeded -- the same warn-once-per-episode
        discipline `_enqueue_frame`'s own saturation warning already
        uses."""
        rtt_ms = self.last_rtt_ms
        added_delay_ms_p95 = event["capture_to_send_ms_p95"] + rtt_ms / 2 if rtt_ms is not None else None
        enriched = {**event, "rtt_ms": rtt_ms, "added_delay_ms_p95": added_delay_ms_p95}
        if added_delay_ms_p95 is not None:
            if (
                self._connection_worst_added_delay_ms is None
                or added_delay_ms_p95 > self._connection_worst_added_delay_ms
            ):
                self._connection_worst_added_delay_ms = added_delay_ms_p95
            self._connection_last_added_delay_ms = added_delay_ms_p95
            if added_delay_ms_p95 > ADDED_DELAY_BUDGET_MS and not self._added_delay_over_budget_warned:
                self._added_delay_over_budget_warned = True
                logger.warning(
                    "edge source: device %s added_delay_ms_p95=%.1fms exceeds the %dms budget",
                    self._connected_device_id,
                    added_delay_ms_p95,
                    ADDED_DELAY_BUDGET_MS,
                )
        self._speech_signals.publish(enriched)
