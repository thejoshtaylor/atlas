"""Panel events built from plain values (Phase 15)."""

from __future__ import annotations

from typing import Any


def asks_question(request: Any) -> bool:
    return None


def request_of_this_turn(current: Any, before: Any) -> Any:
    return None


def turn_ended_event(
    *, outcome: str, failed: bool, cancelled: bool, request: Any, playback_end_at: float | None, now: float
) -> dict[str, Any]:
    return {}


def reply_started_event(text: str) -> dict[str, Any]:
    return {}
