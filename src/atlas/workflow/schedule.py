"""`resolve_schedule`: the one boundary in this project where a caller's
way of saying "when" becomes the database's absolute, aware UTC `due_at`
(05-CONTEXT.md D-04, 05-01-PLAN.md PA-D4).

One pure function. No I/O, no global read, deterministic for fixed
arguments -- everything it needs (the current instant, the server's own
resolved zone) is handed in, never read from `os.environ` or a module
global itself. `workflow/tool.py` (plan 05-01) and `routes/workflows.py`
(plan 05-04) are this function's only two callers, and both hand it their
own request's `delay_seconds`/`at`, `app.state`'s resolved zone, and an
injected clock -- neither does time arithmetic of its own (this module's
own acceptance criterion: `db/postgres.py` and `workflow/tool.py` contain
no `ZoneInfo`, `fromisoformat`, `timedelta`, or `astimezone` of their own).

Five rules, each named for the specific way this can silently produce the
wrong moment:

1. Exactly one of `delay_seconds`/`at`. Both, or neither, is a caller
   error -- raised as `ScheduleError`, never resolved from a guess.
2. `delay_seconds` is `now + that many seconds`, in absolute instants. No
   zone is read and none is needed: an elapsed interval crosses a
   daylight-saving boundary without noticing it, which is exactly why this
   is the preferred form for "in twenty minutes" (FLOW-01's own example).
3. An `at` string carrying an explicit UTC offset (or `Z`) is converted to
   UTC as given. No zone is read -- the offset already says which instant
   was meant.
4. An `at` string carrying no offset -- a local wall-clock string, exactly
   what a browser's `datetime-local` input produces (05-01-PLAN.md PA-03)
   -- is interpreted in `zone`, and `zone` is the server's own resolved
   `server.timezone`, **never the caller's and never the process
   default**. A phone in another timezone, or a container whose `TZ`
   nobody set, must not be able to schedule the house for the wrong
   moment -- `tests/test_workflow_schedule.py` proves this by setting a
   contradictory `TZ` in the test process itself.
5. A zone-less `at` naming a local time that is **ambiguous** (the
   repeated hour when clocks go back) or **nonexistent** (the skipped hour
   when clocks go forward) is refused by name rather than resolved by
   `fold` -- silently picking one of two 1:30ams is not an answer to when
   a house does something. `zone` of `None` (the operator left
   `server.timezone` unset, a real, supported choice --
   `ServerConfig.timezone`'s own docstring) with a zone-less `at` is
   refused the same way, naming the configuration key that would let that
   form work, never falling back to the process's own zone.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo


class ScheduleError(Exception):
    """Raised by `resolve_schedule` when a caller's way of saying "when"
    cannot become an absolute `due_at` -- both/neither of
    `delay_seconds`/`at` given, an `at` with no offset and no `zone` to
    resolve it against, or a zone-less `at` naming an ambiguous or
    nonexistent local time. Raised, not returned, matching this project's
    own `ConfigError`/`Denied` doctrine: a caller cannot silently continue
    scheduling from a guess."""


def _local_datetime_status(dt_naive: datetime, zone: ZoneInfo) -> str:
    """Classifies `dt_naive` (a naive wall-clock reading) as it would fall
    in `zone`: `"unambiguous"`, `"ambiguous"` (the repeated hour when
    clocks go back -- two real instants share this wall time), or
    `"nonexistent"` (the skipped hour when clocks go forward -- no real
    instant has this wall time).

    Follows PEP 495's own `fold` mechanism directly: attaching `zone` with
    `fold=0` resolves to the offset in force *before* whatever transition
    is nearest this wall time, `fold=1` to the offset *after* it. When the
    two offsets agree, the wall time is unambiguous by construction. When
    they disagree, one further check tells ambiguous from nonexistent: an
    ambiguous wall time round-trips through UTC and back to the same wall
    time (both real instants land on it); a nonexistent one does not (the
    nearest real instant this wall time's own offset produces is a
    *different* wall time entirely, on the other side of the gap).
    """
    fold0 = dt_naive.replace(tzinfo=zone, fold=0)
    fold1 = dt_naive.replace(tzinfo=zone, fold=1)
    if fold0.utcoffset() == fold1.utcoffset():
        return "unambiguous"
    round_tripped = fold0.astimezone(timezone.utc).astimezone(zone)
    if round_tripped.replace(tzinfo=None) != dt_naive:
        return "nonexistent"
    return "ambiguous"


def resolve_schedule(
    *,
    delay_seconds: int | None,
    at: str | None,
    now: datetime,
    zone: ZoneInfo | None,
) -> datetime:
    """Turn exactly one of `delay_seconds`/`at` into an absolute, aware
    UTC `datetime`. See this module's own docstring for the five rules
    this enforces. `now` must already be an aware UTC `datetime` (the
    caller's injected clock, matching `RetentionScheduler`'s own clock
    shape) -- this function does not read the wall clock itself."""
    if (delay_seconds is None) == (at is None):
        raise ScheduleError(
            "resolve_schedule needs exactly one of delay_seconds or at -- got "
            f"delay_seconds={delay_seconds!r}, at={at!r}"
        )

    if delay_seconds is not None:
        if delay_seconds < 0:
            raise ScheduleError(f"delay_seconds must be >= 0, got {delay_seconds!r}")
        return now + timedelta(seconds=delay_seconds)

    assert at is not None
    parsed = datetime.fromisoformat(at)
    if parsed.tzinfo is not None:
        # An explicit offset (including "Z") was given -- used as given,
        # no zone read at all (rule 3).
        return parsed.astimezone(timezone.utc)

    # No offset: a local wall-clock string, resolved against the server's
    # own zone and never the caller's or the process default (rule 4).
    if zone is None:
        raise ScheduleError(
            f"at={at!r} carries no UTC offset, and server.timezone is not set -- set "
            "server.timezone in the configuration file to a real IANA zone name (for "
            "example 'America/Los_Angeles'), or give 'at' with an explicit UTC offset "
            "instead"
        )

    status = _local_datetime_status(parsed, zone)
    if status == "ambiguous":
        raise ScheduleError(
            f"at={at!r} is ambiguous in {zone} -- this local time occurs twice, once on "
            "each side of a daylight-saving transition. Name a UTC offset explicitly to "
            "say which one you mean."
        )
    if status == "nonexistent":
        raise ScheduleError(
            f"at={at!r} does not exist in {zone} -- this local time is skipped by a "
            "daylight-saving transition (clocks jump past it). Choose a time on either "
            "side of the transition."
        )
    return parsed.replace(tzinfo=zone).astimezone(timezone.utc)
