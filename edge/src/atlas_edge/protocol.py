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


def parse_server_message(text: str) -> "Hello | Ping":
    """Parse one text frame from the server -- `hello` or `ping`, the only
    two message types the server ever sends. Raises `ProtocolError` for
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
