"""Timer and alarm operations over a `TimerRepository`.

The voice tools and the REST routes both call these functions, so the limits
and the time math live in one place. Every refusal is a `TimerError`.
"""

from __future__ import annotations

from datetime import datetime, timedelta

from atlas.db.timer_repository import Timer, TimerRepository
from atlas.timers.core import MAX_TIMERS, TimerLimitError, TimerSpec


async def _check_room(repo: TimerRepository) -> None:
    if len(await repo.list_timers()) >= MAX_TIMERS:
        raise TimerLimitError(
            f"there are already {MAX_TIMERS} timers and alarms, which is the limit; delete one first"
        )


async def create_timer(repo: TimerRepository, spec: TimerSpec, *, now: datetime) -> Timer:
    await _check_room(repo)
    return await repo.create_timer(
        kind="timer",
        label=spec.label,
        due_at=now + timedelta(seconds=spec.duration_seconds),
        remaining_s=None,
        duration_s=spec.duration_seconds,
        time_of_day=None,
        repeat_days=0,
        enabled=True,
        created_at=now,
    )
