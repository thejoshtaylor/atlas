"""Panel events built from plain values (Phase 15).

`run_turn` emits two events for the desktop panel besides `wake.confirmed`:
`reply.started` (the panel shows "speaking" and the reply text) and
`turn.ended` (the panel shows "done", starts its hide timer and decides about
the follow-up window). The shapes and the rules behind them live here as pure
functions, so `controller.py` only calls them.

The outcome stays the internal value. The desktop bridge maps it to the wire
set. The question flag and the outcome come from code, never from the model.
"""

from __future__ import annotations

from typing import Any


def asks_question(request: Any) -> bool:
    """True when the follow-up `request` is a question the operator should answer.

    D-06: the panel stays open for the follow-up window only when the turn
    asks a question. A confirmation and a clarification are questions. A
    Phase 13 answer window after a plain answer is not: it opens a listening
    window, but the reply asked nothing. It counts only when the reply's own
    `expects_reply` flag is set. `getattr` keeps a test double usable.
    """
    if request is None:
        return False
    kind = getattr(request, "kind", None)
    if kind in ("confirmation", "clarification"):
        return True
    return kind == "answer" and bool(getattr(request, "expects_reply", False))


def request_of_this_turn(current: Any, before: Any) -> Any:
    """The follow-up request this turn made, or None.

    A channel keeps one `requested` slot across turns. A request left from an
    earlier turn is the same object as `before` and never counts as this
    turn's own.
    """
    if current is None or current is before:
        return None
    return current


def turn_ended_event(
    *,
    outcome: str,
    failed: bool,
    cancelled: bool,
    request: Any,
    playback_end_at: float | None,
    now: float,
) -> dict[str, Any]:
    """The `turn.ended` event: how the turn ended and what the panel does next.

    `failed` wins over `cancelled`, and both win over `outcome`, so a crashed
    turn never reads "completed". `playback_end_at` and `now` are in
    `time.monotonic` seconds. `playback_ms_left` is how long the reply audio
    still plays, never negative.
    """
    if failed:
        ended = "failed"
    elif cancelled:
        ended = "cancelled"
    else:
        ended = outcome
    ms_left = 0 if playback_end_at is None else max(0, round((playback_end_at - now) * 1000))
    return {
        "type": "turn.ended",
        "outcome": ended,
        "follow_up": request is not None,
        "asks_question": asks_question(request),
        "playback_ms_left": ms_left,
    }


def reply_started_event(text: str) -> dict[str, Any]:
    """The `reply.started` event: the turn starts to speak an answer."""
    return {"type": "reply.started", "text": text}
