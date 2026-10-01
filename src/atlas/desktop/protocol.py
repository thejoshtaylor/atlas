"""The `/ws/desktop` wire protocol, version 1 (Phase 14, D-09, D-21).

The shared fixtures in `desktop/protocol/v1/` are the contract. This module
is the Python half of it, and `swift test` reads the same files for the Mac
half. Every constant below has a twin in `constants.json` or
`close_codes.json`, and `tests/test_desktop_protocol_contract.py` fails if
a twin drifts.

Every frame is one JSON text object with a string `type`. A known type is
validated strictly. An unknown `type` is not an error: an old server must
keep working with a newer app, so it parses to `UnknownMessage` and the hub
ignores it.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    StrictInt,
    StrictStr,
    TypeAdapter,
    ValidationError,
)

from atlas.desktop.cards import CardMessage
from atlas.desktop.display_text import (
    ID_MAX,
    LABEL_MAX,
    MS_MAX,
    TIMER_ID_MAX,
    TRANSCRIPT_TEXT_MAX,
    WORD_MAX,
    sanitize_display_text,
)

PROTOCOL_VERSION = 1

PING_INTERVAL_S = 15
PONG_TIMEOUT_S = 10
HELLO_TIMEOUT_S = 10.0
IDLE_TIMEOUT_S = 45.0
MAX_TEXT_FRAME_BYTES = 2048
MAX_INVALID_MESSAGES = 20
TEST_TIMEOUT_S = 5.0

# The server's own pre-accept refusal of a Mac token is an HTTP 403 with this
# header (D-30). A proxy, WAF or ingress also answers 403, but never with this
# header, so the app counts only a marked 403 toward the two-strike unpair.
REFUSAL_STATUS = 403
REFUSAL_HEADER = "X-Atlas-Refusal"
REFUSAL_TOKEN = "token"

CLOSE_GOING_AWAY = 1001
# 1008 stays "policy violation": a bad or missing hello, or too many bad
# frames. Revoke has its own code below on purpose, because a live-socket
# 1008 already means "too many bad frames" (RESEARCH Pattern 3).
CLOSE_POLICY_VIOLATION = 1008
CLOSE_SUPERSEDED = 4000
CLOSE_REVOKED = 4001
CLOSE_PROTOCOL_MISMATCH = 4002

MSG_HELLO = "hello"
MSG_HELLO_ACK = "hello.ack"
MSG_PING = "ping"
MSG_PONG = "pong"
MSG_ERROR = "error"
MSG_WAKE_CONFIRMED = "wake.confirmed"
MSG_STATE = "state"
MSG_TRANSCRIPT_PARTIAL = "transcript.partial"
MSG_TRANSCRIPT_FINAL = "transcript.final"
MSG_CARD = "card"
MSG_TURN_ENDED = "turn.ended"
MSG_TIMER_RINGING = "timer.ringing"
MSG_TIMER_STOPPED = "timer.stopped"
MSG_TIMER_STOP = "timer.stop"

TURN_STATES = ("listening", "thinking", "speaking")
WIRE_OUTCOMES = ("completed", "no_speech", "stopped", "failed")

_PING_ID_MAX = 2147483647

_ShortText = Annotated[StrictStr, Field(min_length=1, max_length=WORD_MAX)]
_Capability = Annotated[StrictStr, Field(min_length=1, max_length=64)]
_PingId = Annotated[StrictInt, Field(ge=0, le=_PING_ID_MAX)]
_TurnId = Annotated[StrictStr, Field(min_length=1, max_length=ID_MAX)]
_TranscriptText = Annotated[StrictStr, Field(max_length=TRANSCRIPT_TEXT_MAX)]
_Label = Annotated[StrictStr, Field(max_length=LABEL_MAX)]
_Millis = Annotated[StrictInt, Field(ge=0, le=MS_MAX)]
_TimerId = Annotated[StrictInt, Field(ge=0, le=TIMER_ID_MAX)]


class DesktopProtocolError(ValueError):
    """The client's text is not a valid frame of a type this server knows."""


class DesktopHello(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["hello"]
    protocol: StrictInt
    app_version: _ShortText
    os_version: _ShortText
    capabilities: Annotated[list[_Capability], Field(max_length=32)]


class DesktopPing(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["ping"]
    id: _PingId


class DesktopPong(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["pong"]
    id: _PingId


class DesktopHelloAck(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["hello.ack"]
    protocol: StrictInt
    device_id: StrictInt
    ping_interval_s: StrictInt


class DesktopError(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["error"]
    code: StrictStr
    detail: StrictStr


class DesktopWakeConfirmed(BaseModel):
    """The server confirmed a wake for this turn (Phase 15, D-05, D-10)."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["wake.confirmed"]
    turn_id: _TurnId


class DesktopTurnState(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["state"]
    turn_id: _TurnId
    state: Literal["listening", "thinking", "speaking"]


class DesktopTranscriptPartial(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["transcript.partial"]
    turn_id: _TurnId
    text: _TranscriptText


class DesktopTranscriptFinal(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["transcript.final"]
    turn_id: _TurnId
    text: _TranscriptText


class DesktopTurnEnded(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["turn.ended"]
    turn_id: _TurnId
    outcome: Literal["completed", "no_speech", "stopped", "failed"]
    follow_up_window_ms: _Millis
    playback_ms_left: _Millis


class DesktopTimerRinging(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["timer.ringing"]
    timer_id: _TimerId
    kind: Literal["timer", "alarm"]
    label: _Label


class DesktopTimerStopped(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["timer.stopped"]
    timer_id: _TimerId


class DesktopTimerStop(BaseModel):
    """A paired Mac asks the server to stop a ringing timer."""

    model_config = ConfigDict(extra="ignore")

    type: Literal["timer.stop"]
    timer_id: _TimerId


class UnknownMessage(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: StrictStr


_CLIENT_MODELS: dict[str, type[BaseModel]] = {
    MSG_HELLO: DesktopHello,
    MSG_PING: DesktopPing,
    MSG_PONG: DesktopPong,
    MSG_TIMER_STOP: DesktopTimerStop,
}

ServerMessage = Annotated[
    DesktopHelloAck
    | DesktopPing
    | DesktopPong
    | DesktopError
    | DesktopWakeConfirmed
    | DesktopTurnState
    | DesktopTranscriptPartial
    | DesktopTranscriptFinal
    | CardMessage
    | DesktopTurnEnded
    | DesktopTimerRinging
    | DesktopTimerStopped,
    Field(discriminator="type"),
]
SERVER_MESSAGE_ADAPTER: TypeAdapter[ServerMessage] = TypeAdapter(ServerMessage)


def parse_client_message(
    text: str,
) -> DesktopHello | DesktopPing | DesktopPong | DesktopTimerStop | UnknownMessage:
    """Parse one frame from a Mac. Raises `DesktopProtocolError` for text
    that is not a JSON object, has no string `type`, or fails validation
    for a known type. Any other `type` is `UnknownMessage`."""
    try:
        payload = json.loads(text)
    except ValueError as exc:
        raise DesktopProtocolError("frame is not JSON") from exc
    if not isinstance(payload, dict):
        raise DesktopProtocolError("frame is not a JSON object")
    message_type = payload.get("type")
    if not isinstance(message_type, str):
        raise DesktopProtocolError("frame has no string type")
    model = _CLIENT_MODELS.get(message_type)
    if model is None:
        return UnknownMessage(type=message_type)
    try:
        return model.model_validate(payload)  # type: ignore[return-value]
    except ValidationError as exc:
        raise DesktopProtocolError(f"invalid {message_type} frame") from exc


def raw_hello_protocol(text: str) -> int | None:
    """The `protocol` integer of a hello frame, read from the raw JSON before
    any model validation. `None` when the frame is not a hello object or the
    field is not an integer. A future protocol may change the hello shape, so
    the server must be able to name a version mismatch for a hello that this
    version cannot validate (D-09)."""
    try:
        payload = json.loads(text)
    except ValueError:
        return None
    if not isinstance(payload, dict) or payload.get("type") != MSG_HELLO:
        return None
    version = payload.get("protocol")
    if isinstance(version, bool) or not isinstance(version, int):
        return None
    return version


def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(), separators=(",", ":"))


def build_hello_ack(device_id: int) -> str:
    return _dump(
        DesktopHelloAck(
            type=MSG_HELLO_ACK,
            protocol=PROTOCOL_VERSION,
            device_id=device_id,
            ping_interval_s=PING_INTERVAL_S,
        )
    )


def build_ping(ping_id: int) -> str:
    return _dump(DesktopPing(type=MSG_PING, id=ping_id))


def build_pong(ping_id: int) -> str:
    return _dump(DesktopPong(type=MSG_PONG, id=ping_id))


def build_error(code: str, detail: str) -> str:
    return _dump(DesktopError(type=MSG_ERROR, code=code, detail=detail))


def build_wake_confirmed(turn_id: str) -> str:
    return _dump(DesktopWakeConfirmed(type=MSG_WAKE_CONFIRMED, turn_id=turn_id))


def build_turn_state(turn_id: str, state: str) -> str:
    return _dump(DesktopTurnState(type=MSG_STATE, turn_id=turn_id, state=state))  # type: ignore[arg-type]


def build_transcript_partial(turn_id: str, text: str) -> str:
    clean = sanitize_display_text(text, TRANSCRIPT_TEXT_MAX, keep="end")
    return _dump(DesktopTranscriptPartial(type=MSG_TRANSCRIPT_PARTIAL, turn_id=turn_id, text=clean))


def build_transcript_final(turn_id: str, text: str) -> str:
    clean = sanitize_display_text(text, TRANSCRIPT_TEXT_MAX, keep="end")
    return _dump(DesktopTranscriptFinal(type=MSG_TRANSCRIPT_FINAL, turn_id=turn_id, text=clean))


def _clamp_ms(value: int) -> int:
    return max(0, min(MS_MAX, value))


def build_turn_ended(
    turn_id: str, outcome: str, follow_up_window_ms: int, playback_ms_left: int
) -> str:
    return _dump(
        DesktopTurnEnded(
            type=MSG_TURN_ENDED,
            turn_id=turn_id,
            outcome=outcome,  # type: ignore[arg-type]
            follow_up_window_ms=_clamp_ms(follow_up_window_ms),
            playback_ms_left=_clamp_ms(playback_ms_left),
        )
    )


def build_timer_ringing(timer_id: int, kind: str, label: str) -> str:
    return _dump(
        DesktopTimerRinging(
            type=MSG_TIMER_RINGING,
            timer_id=timer_id,
            kind=kind,  # type: ignore[arg-type]
            label=sanitize_display_text(label, LABEL_MAX, keep="start"),
        )
    )


def build_timer_stopped(timer_id: int) -> str:
    return _dump(DesktopTimerStopped(type=MSG_TIMER_STOPPED, timer_id=timer_id))
