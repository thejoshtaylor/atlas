"""The card envelope on the wire (Phase 15, D-11, CARD-08).

Code builds every card from typed values. The only free text is the reply
text, and it is sanitized and capped before it gets here. No card field comes
from any other model output, and every model forbids extra keys, so a card
with a markup field or an unknown kind fails validation.

This phase emits only the `text` kind. The `timer` kind exists so the Mac's
decoder is proven against a fixture (RESEARCH Pitfall 6); the ring view is
driven by `timer.ringing`, not by a card.

This module must not import `protocol.py`: the protocol's `ServerMessage`
union nests `CardMessage`.
"""

from __future__ import annotations

import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictInt, StrictStr

from atlas.desktop.display_text import (
    CARD_TEXT_MAX,
    ID_MAX,
    LABEL_MAX,
    TIMER_ID_MAX,
    sanitize_display_text,
)

_Id = Annotated[StrictStr, Field(min_length=1, max_length=ID_MAX)]
_Fallback = Annotated[StrictStr, Field(max_length=CARD_TEXT_MAX)]


class TextCardData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    text: Annotated[StrictStr, Field(min_length=1, max_length=CARD_TEXT_MAX)]


class TimerCardData(BaseModel):
    model_config = ConfigDict(extra="forbid")

    timer_id: Annotated[StrictInt, Field(ge=0, le=TIMER_ID_MAX)]
    timer_kind: Literal["timer", "alarm"]
    label: Annotated[StrictStr, Field(max_length=LABEL_MAX)]


class TextCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["card"]
    turn_id: _Id
    card_id: _Id
    kind: Literal["text"]
    fallback_text: _Fallback
    data: TextCardData


class TimerCard(BaseModel):
    model_config = ConfigDict(extra="forbid")

    type: Literal["card"]
    turn_id: _Id
    card_id: _Id
    kind: Literal["timer"]
    fallback_text: _Fallback
    data: TimerCardData


CardMessage = Annotated[TextCard | TimerCard, Field(discriminator="kind")]


def _dump(model: BaseModel) -> str:
    return json.dumps(model.model_dump(), separators=(",", ":"))


def build_text_card(turn_id: str, card_id: str, text: str) -> str | None:
    """A text card for the reply text, or `None` when nothing is left to show
    after sanitizing."""
    clean = sanitize_display_text(text, CARD_TEXT_MAX, keep="start")
    if not clean:
        return None
    return _dump(
        TextCard(
            type="card",
            turn_id=turn_id,
            card_id=card_id,
            kind="text",
            fallback_text=clean,
            data=TextCardData(text=clean),
        )
    )


def build_timer_card(
    turn_id: str, card_id: str, timer_id: int, timer_kind: Literal["timer", "alarm"], label: str
) -> str:
    """A timer card from typed values. Only the label is text, and it is
    sanitized."""
    clean = sanitize_display_text(label, LABEL_MAX, keep="start")
    word = "Timer" if timer_kind == "timer" else "Alarm"
    return _dump(
        TimerCard(
            type="card",
            turn_id=turn_id,
            card_id=card_id,
            kind="timer",
            fallback_text=f"{word}: {clean}" if clean else word,
            data=TimerCardData(timer_id=timer_id, timer_kind=timer_kind, label=clean),
        )
    )
