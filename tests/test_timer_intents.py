"""Timer and alarm commands that run with no brain call (261001-b7l)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

import pytest

from atlas.timers.intents import handle, parse, spoken_clock
from tests.timer_fakes import FakeTimerRepository

ZONE = ZoneInfo("America/Los_Angeles")
# 2026-10-01 20:00 in Los Angeles.
NOW = datetime(2026, 10, 2, 3, 0, tzinfo=timezone.utc)


@pytest.mark.parametrize(
    ("text", "seconds", "label"),
    [
        ("Hey Atlas, set a timer for 5 minutes.", 300, ""),
        ("Set a timer for five minutes", 300, ""),
        ("10 minute timer", 600, ""),
        ("start a 3 minute timer", 180, ""),
        ("set a pasta timer for ten minutes", 600, "pasta"),
        ("set a 5 minute pasta timer", 300, "pasta"),
        ("timer for an hour and a half", 5400, ""),
        ("set a timer for one hour thirty minutes", 5400, ""),
        ("set a timer for twenty five seconds", 25, ""),
        ("set a timer for half an hour", 1800, ""),
        ("set a timer for 2 and a half minutes", 150, ""),
    ],
)
def test_timer_commands_parse(text, seconds, label):
    command = parse(text)
    assert (command.seconds, command.label) == (seconds, label)


@pytest.mark.parametrize(
    "text",
    [
        "turn on the kitchen light",
        "set a timer",
        "what time is it",
        "Off my alarm.",
        "set a timer for 5 minutes and turn off the lights",
        "set an alarm for seven every weekday",
        "set a timer for the oven",
    ],
)
def test_anything_else_goes_to_the_brain(text):
    assert parse(text) is None


async def test_set_timer_creates_it_and_says_so():
    repo = FakeTimerRepository()
    reply = await handle("set a pasta timer for 10 minutes", repo, now=NOW, zone=ZONE)

    assert reply == "Pasta timer set for 10 minutes."
    [timer] = await repo.list_timers()
    assert (timer.kind, timer.label, timer.due_at) == ("timer", "pasta", NOW + timedelta(minutes=10))


@pytest.mark.parametrize(
    ("text", "hhmm", "spoken"),
    [
        ("Set an alarm for six A M.", "06:00", "6 a.m."),
        ("set an alarm for 6:30 a.m.", "06:30", "6:30 a.m."),
        ("wake me up at six thirty in the morning", "06:30", "6:30 a.m."),
        ("set an alarm for noon", "12:00", "noon"),
        # 8 p.m. now, so a bare 7 is 7 a.m. and a bare 9 is 9 p.m.
        ("alarm for seven", "07:00", "7 a.m."),
        ("alarm for nine", "21:00", "9 p.m."),
    ],
)
async def test_set_alarm_in_the_house_zone(text, hhmm, spoken):
    repo = FakeTimerRepository()
    reply = await handle(text, repo, now=NOW, zone=ZONE)

    assert reply == f"Alarm set for {spoken}."
    [alarm] = await repo.list_timers()
    assert (alarm.kind, alarm.time_of_day, alarm.repeat_days) == ("alarm", hhmm, 0)


async def test_an_alarm_with_no_zone_goes_to_the_brain():
    assert await handle("set an alarm for 6 am", FakeTimerRepository(), now=NOW, zone=None) is None


async def test_cancel_one_or_all():
    repo = FakeTimerRepository()
    assert await handle("cancel my alarm", repo, now=NOW, zone=ZONE) == "There is no alarm set."
    await handle("set an alarm for 6 am", repo, now=NOW, zone=ZONE)
    assert await handle("Cancel my alarm.", repo, now=NOW, zone=ZONE) == "Alarm cancelled."
    assert await repo.list_timers() == []

    await handle("set an alarm for 6 am", repo, now=NOW, zone=ZONE)
    await handle("set an alarm for 7 am", repo, now=NOW, zone=ZONE)
    # Two alarms and no "all": the brain asks which one.
    assert await handle("cancel the alarm", repo, now=NOW, zone=ZONE) is None
    assert await handle("Cancel all alarms.", repo, now=NOW, zone=ZONE) == "All 2 alarms cancelled."
    assert await repo.list_timers() == []


async def test_time_left_on_the_one_timer():
    repo = FakeTimerRepository()
    await handle("set a timer for 5 minutes", repo, now=NOW, zone=ZONE)
    reply = await handle("how much time is left", repo, now=NOW + timedelta(seconds=90), zone=ZONE)
    assert reply == "3 minutes 30 seconds left."


def test_spoken_clock():
    assert [spoken_clock(t) for t in ("00:00", "06:00", "12:00", "18:30")] == [
        "midnight",
        "6 a.m.",
        "noon",
        "6:30 p.m.",
    ]
