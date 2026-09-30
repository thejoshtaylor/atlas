"""Pure timer and alarm logic: limits, input models, time math and text.

No I/O and no clock read. Every function gets `now` from its caller, so a
test can fix the time. Error messages are safe to speak and to show.
"""

from __future__ import annotations

import math
import re
from datetime import datetime, time, timedelta, timezone, tzinfo
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from atlas.db.timer_repository import Timer

MAX_TIMERS = 50
MAX_TIMER_SECONDS = 86400
MAX_LABEL_LENGTH = 40
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

_TIME_PATTERN = r"^([01]\d|2[0-3]):[0-5]\d$"
_LABEL_RE = re.compile(r"[\w' -]{0,%d}" % MAX_LABEL_LENGTH)
_NON_SPACE_WHITESPACE_RE = re.compile(r"[^\S ]")


class TimerError(ValueError):
    """A refusal whose message is safe to speak and to show."""


class TimerNotFoundError(TimerError):
    pass


class TimerLimitError(TimerError):
    pass


def clean_label(text: str) -> str:
    """Collapse runs of spaces and strip. Accept letters, digits, space,
    hyphen and apostrophe only, up to 40 characters. A newline or tab is
    refused, never folded into a space: a label goes back into the model's
    context on every turn."""
    if _NON_SPACE_WHITESPACE_RE.search(text):
        raise ValueError("a label may not contain a line break or a tab")
    collapsed = re.sub(r" +", " ", text).strip()
    if _LABEL_RE.fullmatch(collapsed) is None:
        raise ValueError(
            f"a label may hold only letters, digits, spaces, hyphens and apostrophes, "
            f"at most {MAX_LABEL_LENGTH} characters"
        )
    return collapsed


class TimerSpec(BaseModel):
    """A countdown the user calls a timer."""

    model_config = ConfigDict(extra="forbid")

    duration_seconds: int = Field(
        ge=1, le=MAX_TIMER_SECONDS, description="How long the countdown runs, in seconds, 1 to 86400."
    )
    label: str = Field(
        default="", description="The short name the user gave, for example pasta. Empty when none was given."
    )

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        return clean_label(value)


Weekday = Literal["mon", "tue", "wed", "thu", "fri", "sat", "sun"]


def _unique_days(days: "list[str] | None") -> "list[str] | None":
    if days is not None and len(set(days)) != len(days):
        raise ValueError("days may not repeat a weekday")
    return days


class AlarmSpec(BaseModel):
    """An alarm at a clock time in the house's time zone."""

    model_config = ConfigDict(extra="forbid")

    time: str = Field(pattern=_TIME_PATTERN, description="The clock time, 24-hour HH:MM, for example 07:00.")
    days: list[Weekday] = Field(
        default_factory=list,
        description="The weekdays the alarm repeats on, from mon tue wed thu fri sat sun. Empty means once.",
    )
    label: str = Field(default="", description="The short name the user gave, for example work. Empty when none.")

    @field_validator("days")
    @classmethod
    def _days_unique(cls, value: "list[str]") -> "list[str]":
        return _unique_days(value) or []

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: str) -> str:
        return clean_label(value)


class TimerChanges(BaseModel):
    """What to change on a timer or an alarm. Give at least one field."""

    model_config = ConfigDict(extra="forbid")

    label: str | None = Field(default=None, description="A new name for the timer or alarm.")
    remaining_seconds: int | None = Field(
        default=None, ge=1, le=MAX_TIMER_SECONDS, description="Timer only: set the time left, in seconds."
    )
    add_seconds: int | None = Field(
        default=None,
        ge=-MAX_TIMER_SECONDS,
        le=MAX_TIMER_SECONDS,
        description="Timer only: add this many seconds to the time left. Negative takes time away.",
    )
    paused: bool | None = Field(default=None, description="Timer only: true pauses it, false resumes it.")
    time: str | None = Field(
        default=None, pattern=_TIME_PATTERN, description="Alarm only: a new clock time, 24-hour HH:MM."
    )
    days: list[Weekday] | None = Field(
        default=None, description="Alarm only: the weekdays it repeats on. An empty list means once."
    )
    enabled: bool | None = Field(default=None, description="Alarm only: true turns it on, false turns it off.")

    @field_validator("days")
    @classmethod
    def _days_unique(cls, value: "list[str] | None") -> "list[str] | None":
        return _unique_days(value)

    @field_validator("label")
    @classmethod
    def _clean_label(cls, value: "str | None") -> "str | None":
        return None if value is None else clean_label(value)

    @model_validator(mode="after")
    def _one_change(self) -> "TimerChanges":
        if all(getattr(self, name) is None for name in CHANGE_FIELDS):
            raise ValueError("give at least one thing to change")
        if self.remaining_seconds is not None and self.add_seconds is not None:
            raise ValueError("give remaining_seconds or add_seconds, not both")
        return self


CHANGE_FIELDS = ("label", "remaining_seconds", "add_seconds", "paused", "time", "days", "enabled")
TIMER_ONLY_FIELDS = ("remaining_seconds", "add_seconds", "paused")
ALARM_ONLY_FIELDS = ("time", "days", "enabled")


def days_to_mask(days: "list[str] | tuple[str, ...]") -> int:
    mask = 0
    for day in days:
        mask |= 1 << WEEKDAYS.index(day)
    return mask


def mask_to_days(mask: int) -> "list[str]":
    return [day for index, day in enumerate(WEEKDAYS) if mask & (1 << index)]


def remaining_seconds(timer: Timer, now: datetime) -> "int | None":
    """Seconds left on a timer, rounded up. None for an alarm."""
    if timer.kind != "timer":
        return None
    if timer.due_at is None:
        return timer.remaining_s
    return max(0, math.ceil((timer.due_at - now).total_seconds()))


def _plural(count: int, unit: str) -> str:
    return f"{count} {unit}" if count == 1 else f"{count} {unit}s"


def spoken_duration(seconds: int) -> str:
    """The two largest non-zero units, for example "7 minutes 12 seconds"."""
    seconds = max(0, int(seconds))
    hours, rest = divmod(seconds, 3600)
    minutes, secs = divmod(rest, 60)
    parts = [
        _plural(value, unit)
        for value, unit in ((hours, "hour"), (minutes, "minute"), (secs, "second"))
        if value
    ]
    return " ".join(parts[:2]) if parts else "0 seconds"


def clock_text(time_of_day: str) -> str:
    """"07:00" becomes "7:00 AM". Computed here, not with strftime."""
    hour, minute = (int(part) for part in time_of_day.split(":"))
    suffix = "AM" if hour < 12 else "PM"
    hour12 = hour % 12 or 12
    return f"{hour12}:{minute:02d} {suffix}"


def announcement(timer: Timer) -> str:
    if timer.kind == "timer":
        return f"Your {timer.label} timer is done." if timer.label else "Your timer is done."
    clock = clock_text(timer.time_of_day or "00:00")
    if timer.label:
        return f"It's {clock}. Your {timer.label} alarm."
    return f"It's {clock}. This is your alarm."



def next_alarm_at(time_of_day: str, repeat_mask: int, now: datetime, zone: tzinfo) -> datetime:
    """The first instant strictly after `now` that is `time_of_day` in `zone`
    on an allowed weekday (any day when the mask is 0). Aware UTC.

    The candidate is built with `datetime.combine(..., tzinfo=zone)`, which
    uses fold=0. A local time that a DST gap skips rings at the first real
    instant after the gap. A local time that happens twice rings at its first
    occurrence. The process time zone is never read."""
    hour, minute = (int(part) for part in time_of_day.split(":"))
    local_today = now.astimezone(zone).date()
    for offset in range(8):
        day = local_today + timedelta(days=offset)
        if repeat_mask and not repeat_mask & (1 << day.weekday()):
            continue
        candidate = datetime.combine(day, time(hour, minute), tzinfo=zone).astimezone(timezone.utc)
        if candidate > now:
            return candidate
    raise TimerError("could not find the next time for that alarm")  # pragma: no cover


def _singular(text: str) -> str:
    return re.sub(r"\b(hour|minute|second)s\b", r"\1", text)


def _timer_line(timer: Timer, now: datetime) -> str:
    adjective = _singular(spoken_duration(timer.duration_s)) + " " if timer.duration_s else ""
    head = f"- id {timer.id}: {adjective}timer"
    if timer.label:
        head += f' "{timer.label}"'
    left = spoken_duration(remaining_seconds(timer, now) or 0)
    if timer.due_at is None:
        return f"{head}, paused with {left} left"
    return f"{head}, {left} left"


def _alarm_line(timer: Timer) -> str:
    head = f"- id {timer.id}: alarm"
    if timer.label:
        head += f' "{timer.label}"'
    days = " ".join(mask_to_days(timer.repeat_days)) or "once"
    state = "on" if timer.enabled else "off"
    return f"{head} at {clock_text(timer.time_of_day or '00:00')}, {days}, {state}"


def describe_timers(timers: "Sequence[Timer]", now: datetime) -> str:
    """The per-turn context block. Deterministic for a fixed input and `now`.
    Labels are quoted: they are data the user chose, not instructions."""
    if not timers:
        return "Timers and alarms: none are set."
    lines = [
        "Timers and alarms (use these ids with update_timer_or_alarm and delete_timer_or_alarm, "
        "never ask the user for an id):"
    ]
    for timer in timers:
        lines.append(_timer_line(timer, now) if timer.kind == "timer" else _alarm_line(timer))
    return "\n".join(lines)
