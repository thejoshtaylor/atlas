"""Timer and alarm commands that run on the server with no brain call.

A plain command, such as "set a timer for 5 minutes" or "cancel my alarm",
does not need a language model. `handle` matches the whole transcript
against a small grammar. On a match it calls the timer service and returns
the sentence to speak. On no match it returns None, and the turn goes to
the brain as before. The grammar is strict: one extra word and the
transcript goes to the brain.

The transcript is untrusted input. The worst a television can do here is
the same as through the brain: set or cancel a timer. A label passes the
same `clean_label` check as a tool argument.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, tzinfo

from atlas.db.timer_repository import Timer, TimerRepository
from atlas.timers import service
from atlas.timers.core import (
    AlarmSpec,
    TimerError,
    TimerSpec,
    clean_label,
    next_alarm_at,
    remaining_seconds,
    spoken_duration,
)

_ONES = {
    "zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}  # fmt: skip
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50}
_UNITS = {"second": 1, "minute": 60, "hour": 3600}

# Words that start a command, never a timer label.
_RESERVED = frozenset(
    {"a", "an", "the", "my", "for", "set", "start", "timer", "alarm", "and", "me"}
    | set(_ONES)
    | set(_TENS)
)


def _normalize(text: str) -> str:
    text = text.lower().replace("-", " ")
    text = re.sub(r"\ba\.?\s?m\b\.?", "am", text)
    text = re.sub(r"\bp\.?\s?m\b\.?", "pm", text)
    text = re.sub(r"[^\w\s:']", " ", text)
    words = text.split()
    if words[:2] == ["hey", "atlas"]:
        words = words[2:]
    elif words[:1] == ["atlas"]:
        words = words[1:]
    words = [word for word in words if word != "please"]
    return " ".join(words)


def _number(words: list[str]) -> "int | None":
    """Digits, or one to two number words such as "twenty five"."""
    if len(words) == 1 and words[0].isdigit():
        return int(words[0])
    if len(words) == 1 and words[0] in _ONES:
        return _ONES[words[0]]
    if len(words) == 1 and words[0] in _TENS:
        return _TENS[words[0]]
    if len(words) == 2 and words[0] in _TENS and words[1] in _ONES and 0 < _ONES[words[1]] < 10:
        return _TENS[words[0]] + _ONES[words[1]]
    return None


def _unit(word: str) -> "int | None":
    return _UNITS.get(word[:-1] if word.endswith("s") else word)


def _count(words: list[str]) -> "tuple[int, list[str]] | None":
    """The count that opens `words` ("a", "an", one or two number words, or
    digits) and the words after it."""
    if words[:1] in (["a"], ["an"]):
        return 1, words[1:]
    for size in (2, 1):
        count = _number(words[:size])
        if count is not None:
            return count, words[size:]
    return None


def _duration(text: str) -> "int | None":
    """Seconds for "5 minutes", "an hour and a half", "1 hour 30 minutes",
    "half an hour" or "2 and a half minutes". None when it is not one."""
    if text == "half an hour":
        return 1800
    total = 0
    words = text.split()
    while words:
        if words[0] == "and" and total:
            words = words[1:]
        found = _count(words)
        if found is None:
            return None
        count, words = found
        half = words[:3] == ["and", "a", "half"]
        if half:
            words = words[3:]
        unit = _unit(words[0]) if words else None
        if unit is None:
            return None
        words = words[1:]
        if not half and words[:3] == ["and", "a", "half"]:
            half, words = True, words[3:]
        total += count * unit + (unit // 2 if half else 0)
    return total or None


def _clock(text: str) -> "tuple[int, int, str | None] | None":
    """(hour, minute, "am"/"pm"/None) for "6", "6 am", "6:30 pm", "six thirty",
    "six oh five am", "noon" or "midnight"."""
    if text == "noon":
        return 12, 0, "pm"
    if text == "midnight":
        return 0, 0, "am"
    words = text.split()
    meridiem = None
    if words and words[-1] in ("am", "pm"):
        meridiem = words.pop()
    elif words[-3:] == ["in", "the", "morning"]:
        meridiem, words = "am", words[:-3]
    elif words[-3:] in (["in", "the", "evening"], ["in", "the", "afternoon"]) or words[-2:] == ["at", "night"]:
        meridiem, words = "pm", words[:-3] if words[-1] != "night" else words[:-2]
    if words and words[-1] == "o'clock":
        words = words[:-1]
    if len(words) == 1 and re.fullmatch(r"\d{1,2}:\d{2}", words[0]):
        hour, minute = (int(part) for part in words[0].split(":"))
    elif words and _number(words[:1]) is not None:
        hour = _number(words[:1])
        rest = words[1:]
        if not rest:
            minute = 0
        elif rest[0] == "oh" and len(rest) == 2:
            minute = _number(rest[1:])
        else:
            minute = _number(rest)
        if hour is None or minute is None:
            return None
    else:
        return None
    if minute > 59 or hour > 23 or (meridiem and not 1 <= hour <= 12):
        return None
    return hour, minute, meridiem


def _alarm_time(clock: "tuple[int, int, str | None]", now: datetime, zone: tzinfo) -> str:
    """HH:MM, 24-hour. With no am or pm, the next of h:mm and (h+12):mm."""
    hour, minute, meridiem = clock
    if meridiem == "am":
        return f"{hour % 12:02d}:{minute:02d}"
    if meridiem == "pm":
        return f"{hour % 12 + 12:02d}:{minute:02d}"
    if hour == 0 or hour > 12:
        return f"{hour:02d}:{minute:02d}"
    candidates = [f"{hour % 12:02d}:{minute:02d}", f"{hour % 12 + 12:02d}:{minute:02d}"]
    return min(candidates, key=lambda hhmm: next_alarm_at(hhmm, 0, now, zone))


def spoken_clock(hhmm: str) -> str:
    """"06:00" becomes "6 a.m.", "18:30" becomes "6:30 p.m.", "12:00" is noon."""
    hour, minute = (int(part) for part in hhmm.split(":"))
    if (hour, minute) == (12, 0):
        return "noon"
    if (hour, minute) == (0, 0):
        return "midnight"
    suffix = "a.m." if hour < 12 else "p.m."
    hour12 = hour % 12 or 12
    return f"{hour12} {suffix}" if minute == 0 else f"{hour12}:{minute:02d} {suffix}"


@dataclass(frozen=True)
class _SetTimer:
    seconds: int
    label: str


@dataclass(frozen=True)
class _SetAlarm:
    clock: "tuple[int, int, str | None]"


@dataclass(frozen=True)
class _Cancel:
    kind: str
    every: bool


@dataclass(frozen=True)
class _TimeLeft:
    pass


_SET = r"(?:(?:set|start|make)(?: me)? (?:a|an|my) )?"
_TIMER_FOR_RE = re.compile(rf"^{_SET}(?:(?P<label>[\w']+(?: [\w']+)?) )?timer (?:for )?(?P<dur>.+)$")
_DUR_TIMER_RE = re.compile(rf"^{_SET}(?P<dur>.+?) (?:(?P<label>[\w']+) )?timer$")
_DUR_ONLY_TIMER_RE = re.compile(rf"^{_SET}(?P<dur>.+) timer$")
_ALARM_RE = re.compile(
    r"^(?:(?:set|make) (?:an|a|my) )?alarm (?:for|at) (?P<clock>.+?)(?: tomorrow)?$"
    r"|^wake me(?: up)? at (?P<clock2>.+?)(?: tomorrow)?$"
)
_CANCEL_RE = re.compile(
    r"^(?:cancel|delete|remove|clear|turn off|stop)"
    r"(?P<every> all(?: of)?(?: the| my)?| the| my| both(?: the| my)?)? (?P<kind>timer|alarm)(?P<plural>s)?$"
)
_TIME_LEFT_RE = re.compile(
    r"^how (?:much time|long)(?: is| has)?(?: left| remaining| to go)"
    r"(?: on (?:the|my) timer)?$"
)


def _label(raw: "str | None") -> "str | None":
    """"" for no label, the cleaned label, or None when it is not a label."""
    if not raw:
        return ""
    if any(word in _RESERVED for word in raw.split()):
        return None
    try:
        return clean_label(raw)
    except ValueError:
        return None


def parse(text: str) -> "_SetTimer | _SetAlarm | _Cancel | _TimeLeft | None":
    """The command `text` holds, or None."""
    text = _normalize(text)
    if not text:
        return None
    match = _CANCEL_RE.match(text)
    if match:
        every = bool(match["plural"]) or (match["every"] or "").strip().startswith(("all", "both"))
        return _Cancel(kind=match["kind"], every=every)
    if _TIME_LEFT_RE.match(text):
        return _TimeLeft()
    match = _ALARM_RE.match(text)
    if match:
        clock = _clock(match["clock"] or match["clock2"])
        return _SetAlarm(clock) if clock else None
    for pattern in (_TIMER_FOR_RE, _DUR_ONLY_TIMER_RE, _DUR_TIMER_RE):
        match = pattern.match(text)
        if match:
            seconds = _duration(match["dur"])
            label = _label(match.groupdict().get("label"))
            if seconds is not None and label is not None:
                return _SetTimer(seconds, label)
    return None


def _of_kind(timers: "list[Timer]", kind: str) -> "list[Timer]":
    if kind == "timer":
        return [timer for timer in timers if timer.kind == "timer"]
    return [timer for timer in timers if timer.kind == "alarm" and timer.enabled]


async def handle(text: str, repo: TimerRepository, *, now: datetime, zone: "tzinfo | None") -> "str | None":
    """Run the command in `text` and return the reply, or None for no match.

    A command that needs a choice the grammar cannot make, such as "cancel
    my timer" with two timers set, returns None, so the brain asks.
    """
    command = parse(text)
    if command is None:
        return None
    try:
        if isinstance(command, _SetTimer):
            spec = TimerSpec(duration_seconds=command.seconds, label=command.label)
            await service.create_timer(repo, spec, now=now)
            name = f"{command.label} timer" if command.label else "Timer"
            return f"{name[0].upper()}{name[1:]} set for {spoken_duration(command.seconds)}."
        if isinstance(command, _SetAlarm):
            if zone is None:
                return None
            hhmm = _alarm_time(command.clock, now, zone)
            await service.create_alarm(repo, AlarmSpec(time=hhmm), now=now, zone=zone)
            return f"Alarm set for {spoken_clock(hhmm)}."
        timers = await repo.list_timers()
        if isinstance(command, _TimeLeft):
            running = _of_kind(timers, "timer")
            if len(running) != 1:
                return None
            left = remaining_seconds(running[0], now) or 0
            return f"{spoken_duration(left)} left."
        matches = _of_kind(timers, command.kind)
        if not matches:
            return f"There is no {command.kind} set."
        if len(matches) > 1 and not command.every:
            return None
        for timer in matches:
            await service.delete_timer(repo, timer.id)
        if len(matches) == 1:
            return f"{command.kind.capitalize()} cancelled."
        return f"All {len(matches)} {command.kind}s cancelled."
    except TimerError as exc:
        return f"{str(exc)[0].upper()}{str(exc)[1:]}."

