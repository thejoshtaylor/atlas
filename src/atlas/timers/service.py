"""Timer and alarm operations over a `TimerRepository`.

The voice tools and the REST routes both call these functions, so the limits
and the time math live in one place. Every refusal is a `TimerError`.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timedelta, tzinfo

from atlas.db.timer_repository import Timer, TimerRepository
from atlas.timers.core import (
    ALARM_ONLY_FIELDS,
    MAX_TIMER_SECONDS,
    MAX_TIMERS,
    TIMER_ONLY_FIELDS,
    AlarmSpec,
    TimerChanges,
    TimerError,
    TimerLimitError,
    TimerNotFoundError,
    TimerSpec,
    days_to_mask,
    next_alarm_at,
    remaining_seconds,
)

_NO_ZONE_MESSAGE = (
    "an alarm needs a time zone, and server.timezone is not set -- set server.timezone in the "
    "configuration file to a real IANA zone name (for example America/New_York)"
)


def _require_zone(zone: "tzinfo | None") -> tzinfo:
    if zone is None:
        raise TimerError(_NO_ZONE_MESSAGE)
    return zone


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


async def create_alarm(
    repo: TimerRepository, spec: AlarmSpec, *, now: datetime, zone: "tzinfo | None"
) -> Timer:
    zone = _require_zone(zone)
    await _check_room(repo)
    mask = days_to_mask(spec.days)
    return await repo.create_timer(
        kind="alarm",
        label=spec.label,
        due_at=next_alarm_at(spec.time, mask, now, zone),
        remaining_s=None,
        duration_s=None,
        time_of_day=spec.time,
        repeat_days=mask,
        enabled=True,
        created_at=now,
    )


def _refuse_other_kind(timer: Timer, changes: TimerChanges) -> None:
    foreign = ALARM_ONLY_FIELDS if timer.kind == "timer" else TIMER_ONLY_FIELDS
    given = [name for name in foreign if getattr(changes, name) is not None]
    if given:
        raise TimerError(f"a {timer.kind} cannot take {', '.join(given)}")


def _change_timer(timer: Timer, changes: TimerChanges, now: datetime) -> Timer:
    left = remaining_seconds(timer, now) or 0
    if changes.label is not None:
        timer = replace(timer, label=changes.label)
    new_left: "int | None" = None
    if changes.remaining_seconds is not None:
        new_left = changes.remaining_seconds
        timer = replace(timer, duration_s=new_left)
    elif changes.add_seconds is not None:
        new_left = left + changes.add_seconds
        if not 1 <= new_left <= MAX_TIMER_SECONDS:
            raise TimerError(f"the time left must stay between 1 second and {MAX_TIMER_SECONDS} seconds")
        timer = replace(timer, duration_s=max(1, (timer.duration_s or left) + changes.add_seconds))
    if new_left is not None:
        if timer.due_at is None:
            timer = replace(timer, remaining_s=new_left)
        elif changes.add_seconds is not None:
            timer = replace(timer, due_at=timer.due_at + timedelta(seconds=changes.add_seconds))
        else:
            timer = replace(timer, due_at=now + timedelta(seconds=new_left))
    if changes.paused is True and timer.due_at is not None:
        timer = replace(timer, remaining_s=max(1, remaining_seconds(timer, now) or 0), due_at=None)
    elif changes.paused is False and timer.due_at is None:
        timer = replace(timer, due_at=now + timedelta(seconds=timer.remaining_s or 1), remaining_s=None)
    return timer


def _change_alarm(timer: Timer, changes: TimerChanges, now: datetime, zone: "tzinfo | None") -> Timer:
    if changes.label is not None:
        timer = replace(timer, label=changes.label)
    reschedule = timer.enabled and timer.due_at is None
    if changes.time is not None and changes.time != timer.time_of_day:
        timer = replace(timer, time_of_day=changes.time)
        reschedule = True
    if changes.days is not None and days_to_mask(changes.days) != timer.repeat_days:
        timer = replace(timer, repeat_days=days_to_mask(changes.days))
        reschedule = True
    if changes.enabled is False:
        return replace(timer, enabled=False, due_at=None)
    if changes.enabled is True and not timer.enabled:
        timer = replace(timer, enabled=True)
        reschedule = True
    if reschedule and timer.enabled:
        timer = replace(
            timer,
            due_at=next_alarm_at(timer.time_of_day or "00:00", timer.repeat_days, now, _require_zone(zone)),
        )
    return timer


async def update_timer(
    repo: TimerRepository,
    timer_id: int,
    changes: TimerChanges,
    *,
    now: datetime,
    zone: "tzinfo | None",
) -> Timer:
    timer = await repo.get_timer(timer_id)
    if timer is None:
        raise TimerNotFoundError(f"there is no timer or alarm with id {timer_id}")
    _refuse_other_kind(timer, changes)
    updated = (
        _change_timer(timer, changes, now)
        if timer.kind == "timer"
        else _change_alarm(timer, changes, now, zone)
    )
    saved = await repo.save_timer(updated)
    if saved is None:
        raise TimerNotFoundError(f"there is no timer or alarm with id {timer_id}")
    return saved


async def delete_timer(repo: TimerRepository, timer_id: int) -> None:
    if not await repo.delete_timer(timer_id):
        raise TimerNotFoundError(f"there is no timer or alarm with id {timer_id}")
