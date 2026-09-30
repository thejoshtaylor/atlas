"""`Timer` and the `TimerRepository` Protocol (quick task 260930-06x)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Timer:
    """One timer or one alarm.

    `due_at` is aware UTC. It is None for a paused timer and for an alarm
    that is off. `remaining_s` is set for a paused timer only. `duration_s`
    is for timers and `time_of_day` ("HH:MM") is for alarms. `repeat_days`
    is a bitmask where bit i is WEEKDAYS[i] (Monday is bit 0). 0 means once.
    """

    id: int
    kind: str
    label: str
    due_at: datetime | None
    remaining_s: int | None
    duration_s: int | None
    time_of_day: str | None
    repeat_days: int
    enabled: bool
    created_at: datetime


class TimerRepository(Protocol):
    """Storage for timers and alarms.

    `advance_fired` is the at-most-once gate. It runs in one transaction
    with the row locked. It returns False when the row is gone, when
    `enabled` is false, when `due_at` is NULL, or when `due_at > now`. For
    a timer it deletes the row. For an alarm with `next_due_at` None it
    sets enabled=false and due_at=NULL. For any other alarm it sets
    due_at=next_due_at. It then commits and returns True. The caller rings
    only after True.
    """

    async def list_timers(self) -> "list[Timer]": ...

    async def get_timer(self, timer_id: int) -> "Timer | None": ...

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
    ) -> Timer: ...

    async def save_timer(self, timer: Timer) -> "Timer | None": ...

    async def delete_timer(self, timer_id: int) -> bool: ...

    async def advance_fired(
        self, timer_id: int, *, now: datetime, next_due_at: "datetime | None"
    ) -> bool: ...
