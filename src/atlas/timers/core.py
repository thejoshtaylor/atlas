"""Pure timer and alarm logic: limits, input models, time math and text.

No I/O and no clock read. Every function gets `now` from its caller, so a
test can fix the time. Error messages are safe to speak and to show.
"""

from __future__ import annotations

import math
import re
from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field, field_validator

from atlas.db.timer_repository import Timer

MAX_TIMERS = 50
MAX_TIMER_SECONDS = 86400
MAX_LABEL_LENGTH = 40
WEEKDAYS = ("mon", "tue", "wed", "thu", "fri", "sat", "sun")

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

