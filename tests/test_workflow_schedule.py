"""`resolve_schedule` (plan 05-01, D-04, PA-D4): the one boundary in this
project that turns a caller's way of saying "when" into an absolute, aware
UTC `due_at`. Every rule this module's own docstring states is covered
here directly -- no database, no event loop, pure function in and pure
function out.
"""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from zoneinfo import ZoneInfo

import pytest

from atlas.workflow.schedule import ScheduleError, resolve_schedule

_UTC_NOON = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)


def test_both_delay_seconds_and_at_given_is_refused():
    with pytest.raises(ScheduleError):
        resolve_schedule(delay_seconds=60, at="2027-01-01T12:00:00Z", now=_UTC_NOON, zone=None)


def test_neither_delay_seconds_nor_at_given_is_refused():
    with pytest.raises(ScheduleError):
        resolve_schedule(delay_seconds=None, at=None, now=_UTC_NOON, zone=None)


def test_a_negative_delay_is_refused():
    with pytest.raises(ScheduleError):
        resolve_schedule(delay_seconds=-1, at=None, now=_UTC_NOON, zone=None)


def test_delay_seconds_is_now_plus_that_many_seconds_with_no_zone_read():
    due = resolve_schedule(delay_seconds=1200, at=None, now=_UTC_NOON, zone=None)
    assert due == datetime(2027, 1, 1, 12, 20, tzinfo=timezone.utc)


def test_delay_seconds_is_indifferent_to_a_daylight_saving_boundary_in_zone():
    """The delay form does pure elapsed-seconds arithmetic on `now` -- it
    reads no zone at all, so a DST transition the delay happens to cross
    (in wall-clock terms, in whatever zone is passed) changes nothing
    about the result. `zone` is passed here specifically to prove it is
    ignored: an implementation that accidentally localized `now` before
    adding the delay would land on a different UTC instant than this one.
    """
    # 2027-03-14 06:30 UTC is 01:30 America/New_York, thirty minutes
    # before that zone's spring-forward transition (2:00am -> 3:00am).
    now = datetime(2027, 3, 14, 6, 30, tzinfo=timezone.utc)
    due = resolve_schedule(
        delay_seconds=3600, at=None, now=now, zone=ZoneInfo("America/New_York")
    )
    assert due == datetime(2027, 3, 14, 7, 30, tzinfo=timezone.utc)


def test_an_at_carrying_an_explicit_offset_is_used_as_given_with_no_zone_read():
    due = resolve_schedule(
        delay_seconds=None, at="2027-06-01T19:30:00-04:00", now=_UTC_NOON, zone=None
    )
    assert due == datetime(2027, 6, 1, 23, 30, tzinfo=timezone.utc)


def test_an_at_carrying_a_z_suffix_is_used_as_given():
    due = resolve_schedule(
        delay_seconds=None, at="2027-06-01T19:30:00Z", now=_UTC_NOON, zone=None
    )
    assert due == datetime(2027, 6, 1, 19, 30, tzinfo=timezone.utc)


def test_a_zoneless_at_is_resolved_against_the_zone_argument_not_the_process_default():
    """Sets a deliberately contradictory `TZ` in the process environment
    (`time.tzset()` makes it live) and proves the result still matches
    `zone`, never the process's own local zone -- a phone in another
    timezone, or a container whose `TZ` nobody set, must not be able to
    schedule the house for the wrong moment (D-04).
    """
    original_tz = os.environ.get("TZ")
    os.environ["TZ"] = "Pacific/Kiritimati"  # UTC+14 -- nowhere near America/New_York
    time.tzset()
    try:
        due = resolve_schedule(
            delay_seconds=None,
            at="2027-06-01T19:30:00",
            now=_UTC_NOON,
            zone=ZoneInfo("America/New_York"),
        )
    finally:
        if original_tz is None:
            os.environ.pop("TZ", None)
        else:
            os.environ["TZ"] = original_tz
        time.tzset()
    # 19:30 America/New_York in June is EDT (-04:00) -> 23:30 UTC. A result
    # derived from the contradictory Pacific/Kiritimati (+14:00) TZ instead
    # would land on 05:30 UTC the next day -- a different instant entirely.
    assert due == datetime(2027, 6, 1, 23, 30, tzinfo=timezone.utc)


def test_a_zoneless_at_with_no_zone_configured_is_refused_by_name():
    with pytest.raises(ScheduleError) as exc:
        resolve_schedule(delay_seconds=None, at="2027-06-01T19:30:00", now=_UTC_NOON, zone=None)
    assert "server.timezone" in str(exc.value)


def test_an_ambiguous_local_time_is_refused_by_name_not_resolved_by_fold():
    """2027-11-07 01:30 America/New_York occurs twice -- once in EDT, once
    in EST, when clocks fall back. Python's own `fold` would silently pick
    one of the two; this function refuses instead (D-04)."""
    with pytest.raises(ScheduleError) as exc:
        resolve_schedule(
            delay_seconds=None,
            at="2027-11-07T01:30:00",
            now=_UTC_NOON,
            zone=ZoneInfo("America/New_York"),
        )
    assert "ambiguous" in str(exc.value)


def test_a_nonexistent_local_time_is_refused_by_name():
    """2027-03-14 02:30 America/New_York is skipped entirely -- clocks
    jump from 01:59:59 straight to 03:00:00 -- so there is no real instant
    this wall-clock string could mean."""
    with pytest.raises(ScheduleError) as exc:
        resolve_schedule(
            delay_seconds=None,
            at="2027-03-14T02:30:00",
            now=_UTC_NOON,
            zone=ZoneInfo("America/New_York"),
        )
    assert "does not exist" in str(exc.value) or "skip" in str(exc.value)
