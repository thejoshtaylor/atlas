"""`FakeTimerRepository`: an in-memory `TimerRepository` for tests.

It follows the same `advance_fired` rules as the Postgres implementation.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime

from atlas.db.timer_repository import Timer


class FakeTimerRepository:
    def __init__(self) -> None:
        self._by_id: dict[int, Timer] = {}
        self._next_id = 1

    async def list_timers(self) -> "list[Timer]":
        return [self._by_id[key] for key in sorted(self._by_id)]

    async def get_timer(self, timer_id: int) -> "Timer | None":
        return self._by_id.get(timer_id)

    async def create_timer(
        self,
        *,
        kind: str,
        label: str,
        due_at: "datetime | None",
        remaining_s: "int | None",
        duration_s: "int | None",
        time_of_day: "str | None",
        repeat_days: int,
        enabled: bool,
        created_at: datetime,
    ) -> Timer:
        timer = Timer(
            id=self._next_id,
            kind=kind,
            label=label,
            due_at=due_at,
            remaining_s=remaining_s,
            duration_s=duration_s,
            time_of_day=time_of_day,
            repeat_days=repeat_days,
            enabled=enabled,
            created_at=created_at,
        )
        self._next_id += 1
        self._by_id[timer.id] = timer
        return timer

    async def save_timer(self, timer: Timer) -> "Timer | None":
        if timer.id not in self._by_id:
            return None
        self._by_id[timer.id] = timer
        return timer

    async def delete_timer(self, timer_id: int) -> bool:
        return self._by_id.pop(timer_id, None) is not None

    async def advance_fired(
        self, timer_id: int, *, now: datetime, next_due_at: "datetime | None"
    ) -> bool:
        timer = self._by_id.get(timer_id)
        if timer is None or not timer.enabled or timer.due_at is None or timer.due_at > now:
            return False
        if timer.kind == "timer":
            del self._by_id[timer_id]
        elif next_due_at is None:
            self._by_id[timer_id] = replace(timer, enabled=False, due_at=None)
        else:
            self._by_id[timer_id] = replace(timer, due_at=next_due_at)
        return True
