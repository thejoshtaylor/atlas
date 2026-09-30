"""`PostgresTimerRepository`: `TimerRepository` over a real Postgres.

It reuses the naive-UTC boundary helpers of `db/postgres.py`.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from atlas.db.postgres import _to_aware_utc, _to_naive_utc
from atlas.db.timer_models import TimerRow
from atlas.db.timer_repository import Timer


def _naive(dt: "datetime | None") -> "datetime | None":
    return None if dt is None else _to_naive_utc(dt)


def _timer_from_row(row: TimerRow) -> Timer:
    return Timer(
        id=row.id,
        kind=row.kind,
        label=row.label,
        due_at=_to_aware_utc(row.due_at),
        remaining_s=row.remaining_s,
        duration_s=row.duration_s,
        time_of_day=row.time_of_day,
        repeat_days=row.repeat_days,
        enabled=row.enabled,
        created_at=_to_aware_utc(row.created_at),
    )


class PostgresTimerRepository:
    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def list_timers(self) -> "list[Timer]":
        async with self._sessionmaker() as session:
            rows = (await session.execute(select(TimerRow).order_by(TimerRow.id))).scalars().all()
            return [_timer_from_row(row) for row in rows]

    async def get_timer(self, timer_id: int) -> "Timer | None":
        async with self._sessionmaker() as session:
            row = await session.get(TimerRow, timer_id)
            return None if row is None else _timer_from_row(row)

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
        async with self._sessionmaker() as session:
            row = TimerRow(
                kind=kind,
                label=label,
                due_at=_naive(due_at),
                remaining_s=remaining_s,
                duration_s=duration_s,
                time_of_day=time_of_day,
                repeat_days=repeat_days,
                enabled=enabled,
                created_at=_to_naive_utc(created_at),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _timer_from_row(row)

    async def save_timer(self, timer: Timer) -> "Timer | None":
        async with self._sessionmaker() as session:
            row = await session.get(TimerRow, timer.id, with_for_update=True)
            if row is None:
                return None
            row.label = timer.label
            row.due_at = _naive(timer.due_at)
            row.remaining_s = timer.remaining_s
            row.duration_s = timer.duration_s
            row.time_of_day = timer.time_of_day
            row.repeat_days = timer.repeat_days
            row.enabled = timer.enabled
            await session.commit()
            await session.refresh(row)
            return _timer_from_row(row)

    async def delete_timer(self, timer_id: int) -> bool:
        async with self._sessionmaker() as session:
            row = await session.get(TimerRow, timer_id)
            if row is None:
                return False
            await session.delete(row)
            await session.commit()
            return True

    async def advance_fired(
        self, timer_id: int, *, now: datetime, next_due_at: "datetime | None"
    ) -> bool:
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(TimerRow).where(TimerRow.id == timer_id).with_for_update()
                )
            ).scalar_one_or_none()
            if row is None or not row.enabled or row.due_at is None:
                return False
            if row.due_at > _to_naive_utc(now):
                return False
            if row.kind == "timer":
                await session.delete(row)
            elif next_due_at is None:
                row.enabled = False
                row.due_at = None
            else:
                row.due_at = _to_naive_utc(next_due_at)
            await session.commit()
            return True
