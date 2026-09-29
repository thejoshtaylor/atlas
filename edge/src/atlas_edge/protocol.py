"""Edge protocol v1, Pi side.

Stdlib only -- this module (and this whole package) never imports `atlas`.
It restates the same constants and message shapes
`src/atlas/transports/edge.py` defines on the server, so the two sides can
be pinned against each other (`tests/test_edge_protocol_contract.py`)
without either importing the other's module.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

PROTOCOL_VERSION = 1

MSG_HELLO = "hello"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_VAD_START = "vad.start"
MSG_VAD_END = "vad.end"
MSG_DOA = "doa"
MSG_LATENCY = "latency"
# Server to Pi: the turn state to show on the LED ring.
MSG_LED = "led"
# Server to Pi: set the speaker volume. Pi to server: the answer.
MSG_VOLUME = "volume"
MSG_VOLUME_RESULT = "volume.result"

# The two steps of a relative `volume` message, and the longest error text a
# `volume.result` may carry. The server restates the same values in
# `src/atlas/transports/edge.py`.
VOLUME_DIRECTIONS = ("up", "down")
MAX_VOLUME_ERROR_CHARS = 200

# The four LED states, in turn order. The server restates the same values
# in `src/atlas/transports/edge.py`.
LED_IDLE = "idle"
LED_LISTENING = "listening"
LED_THINKING = "thinking"
LED_REPLYING = "replying"
LED_STATES = (LED_IDLE, LED_LISTENING, LED_THINKING, LED_REPLYING)

FRAME_SAMPLES = 256


class ProtocolError(ValueError):
    """Raised by `parse_server_message` for a message this Pi does not
    understand: not valid JSON, not an object, an unknown `type`, or a
    `hello` naming a protocol version other than `PROTOCOL_VERSION`."""


@dataclass(frozen=True)
class Hello:
    """Every field the server's one `hello` message carries (D-08) --
    sent once, right after the server accepts this Pi's connection."""

    protocol: int
    device_id: int
    sample_rate: int
    channels: int
    asr_channel: int
    pre_roll_ms: int
    tail_ms: int
    frame_samples: int


@dataclass(frozen=True)
class Ping:
    """The server's keepalive -- this Pi answers with `pong(id, server_t_ms)`
    at once."""

    id: int
    server_t_ms: int


@dataclass(frozen=True)
class Led:
    """The turn state the server wants on the LED ring -- one of
    `LED_STATES`."""

    state: str


@dataclass(frozen=True)
class Volume:
    """A request to set the speaker volume. Either `level` (an absolute
    percent) or `direction` with `step_percent` is set, never both. The
    server always sends the `min_percent` and `max_percent` limits, and this
    Pi keeps every level inside them."""

    id: int
    min_percent: int
    max_percent: int
    level: "int | None" = None
    direction: "str | None" = None
    step_percent: "int | None" = None


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_volume(raw: "dict[str, Any]") -> Volume:
    request_id = raw.get("id")
    if not _is_int(request_id) or request_id < 0:
        raise ProtocolError(f"server volume message has a bad id: {request_id!r}")
    min_percent = raw.get("min_percent")
    max_percent = raw.get("max_percent")
    if (
        not _is_int(min_percent)
        or not _is_int(max_percent)
        or not (0 <= min_percent <= max_percent <= 100)
    ):
        raise ProtocolError(
            f"server volume message has bad limits: {min_percent!r}, {max_percent!r}"
        )
    level = raw.get("level")
    direction = raw.get("direction")
    if (level is None) == (direction is None):
        raise ProtocolError("server volume message needs exactly one of level and direction")
    if level is not None:
        if not _is_int(level) or not (0 <= level <= 100):
            raise ProtocolError(f"server volume message has a bad level: {level!r}")
        return Volume(id=request_id, min_percent=min_percent, max_percent=max_percent, level=level)
    step_percent = raw.get("step_percent")
    if direction not in VOLUME_DIRECTIONS:
        raise ProtocolError(f"server volume message has a bad direction: {direction!r}")
    if not _is_int(step_percent) or not (1 <= step_percent <= 100):
        raise ProtocolError(f"server volume message has a bad step_percent: {step_percent!r}")
    return Volume(
        id=request_id,
        min_percent=min_percent,
        max_percent=max_percent,
        direction=direction,
        step_percent=step_percent,
    )


def parse_server_message(text: str) -> "Hello | Ping | Led | Volume":
    """Parse one text frame from the server -- `hello`, `ping`, `led`, or
    `volume`, the only four message types the server ever sends. Raises `ProtocolError` for
    anything else, including a `hello` naming a protocol version other
    than `PROTOCOL_VERSION` -- this Pi must never guess at a wire shape a
    version mismatch might have changed.
    """
    try:
        raw = json.loads(text)
    except (json.JSONDecodeError, TypeError) as exc:
        raise ProtocolError(f"server message is not valid JSON: {text!r}") from exc
    if not isinstance(raw, dict):
        raise ProtocolError(f"server message must be a JSON object, got {raw!r}")

    msg_type = raw.get("type")
    if msg_type == MSG_HELLO:
        protocol = raw.get("protocol")
        if protocol != PROTOCOL_VERSION:
            raise ProtocolError(
                f"server hello names protocol {protocol!r}, this Pi speaks "
                f"{PROTOCOL_VERSION!r} only"
            )
        try:
            return Hello(
                protocol=protocol,
                device_id=int(raw["device_id"]),
                sample_rate=int(raw["sample_rate"]),
                channels=int(raw["channels"]),
                asr_channel=int(raw["asr_channel"]),
                pre_roll_ms=int(raw["pre_roll_ms"]),
                tail_ms=int(raw["tail_ms"]),
                frame_samples=int(raw["frame_samples"]),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"server hello is missing or malformed a required field: {raw!r}") from exc

    if msg_type == MSG_PING:
        try:
            return Ping(id=int(raw["id"]), server_t_ms=int(raw["server_t_ms"]))
        except (KeyError, TypeError, ValueError) as exc:
            raise ProtocolError(f"server ping is missing or malformed a required field: {raw!r}") from exc

    if msg_type == MSG_LED:
        state = raw.get("state")
        if not isinstance(state, str) or state not in LED_STATES:
            raise ProtocolError(f"server led message names an unknown state: {state!r}")
        return Led(state=state)

    if msg_type == MSG_VOLUME:
        return _parse_volume(raw)

    raise ProtocolError(f"unsupported server message type: {msg_type!r}")


def vad_start(seq: int) -> str:
    return json.dumps({"type": MSG_VAD_START, "seq": seq})


def vad_end(seq: int) -> str:
    return json.dumps({"type": MSG_VAD_END, "seq": seq})


def doa(azimuth_deg: "list[float]", speech_energy: "list[float] | None" = None) -> str:
    payload: dict[str, Any] = {"type": MSG_DOA, "azimuth_deg": list(azimuth_deg)}
    payload["speech_energy"] = list(speech_energy) if speech_energy is not None else []
    return json.dumps(payload)


def latency(p50: float, p95: float, max_ms: float, frames: int) -> str:
    return json.dumps(
        {
            "type": MSG_LATENCY,
            "capture_to_send_ms_p50": p50,
            "capture_to_send_ms_p95": p95,
            "capture_to_send_ms_max": max_ms,
            "frames": frames,
        }
    )


def pong(ping_id: int, server_t_ms: int) -> str:
    return json.dumps({"type": MSG_PONG, "id": ping_id, "server_t_ms": server_t_ms})


def volume_result(request_id: int, *, level: "int | None" = None, error: "str | None" = None) -> str:
    """The answer to a `volume` message: the `level` read back from the
    mixer, or an `error` text. Give exactly one, or `ValueError` is raised.
    The error is cut to `MAX_VOLUME_ERROR_CHARS`."""
    if (level is None) == (error is None):
        raise ValueError("give exactly one of level and error")
    payload: dict[str, Any] = {"type": MSG_VOLUME_RESULT, "id": request_id}
    if level is not None:
        payload["level"] = level
    else:
        payload["error"] = error[:MAX_VOLUME_ERROR_CHARS]
    return json.dumps(payload)
