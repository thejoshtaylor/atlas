"""`PendingProposal`: the validated shape of one `pending_action` handoff
payload, and the code-composed readback and executing-call arguments built
from it (D-07, D-08).

Every structural fact in a readback comes from the stored proposal, never
from model prose -- `compose_readback` is the one place this plan turns a
`PendingProposal` into words, the same "composed in code, never a second
model round" discipline `turn/controller.py::_compose_mixed_outcome_reply`
and `_compose_clarifying_question` already establish for their own kinds of
turn-ending sentences.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, model_validator

# D-05, D-08: the executing tool for each pending action `action` value --
# the model never sees or calls these names (`mcp/atlas_mcp/google_tools.py`
# lists them under `CODE_ONLY_TOOL_NAMES`). Only `calendar_create` is
# reachable today (plan 09-04 builds `calendar_propose_event` only);
# `calendar_delete` is named here because `PendingProposal` and this map
# are the one shared shape plan 09-05's delete proposal reuses, not a
# second mechanism.
EXECUTING_TOOL_BY_ACTION: dict[str, str] = {
    "calendar_create": "calendar_insert_event",
    "calendar_delete": "calendar_delete_event",
}

# The one fixed phrase every timed `compose_readback` sentence is built
# from -- defined once, used at the one call site that formats a timed
# (not all-day) event's readback, so the sentence shape cannot silently
# drift between two call sites that both mean "read this proposal back".
_CREATE_READBACK_CARRIER = "add {title} to {calendar_phrase}, {when}, {duration}?"

# Plan 09-05, Task 3 (D-07, D-10): the two carriers `compose_readback`
# below chooses between for a `calendar_delete` proposal -- a plain
# single occurrence, and one instance of a recurring event, which speaks
# a different sentence naming that the rest of the series is untouched
# (the operator must never read "delete Standup" and reasonably wonder
# whether the whole series just vanished).
_DELETE_READBACK_CARRIER = "delete {title} from {calendar_phrase}, {when}?"
_DELETE_OCCURRENCE_READBACK_CARRIER = (
    "delete only this one: {title} on {when}, from {calendar_phrase}? the rest of the series stays."
)

# Fixed, spoken-word-for-word replies -- never composed by a model (D-07,
# D-14), the same "a second inference pass could reword this" doctrine
# `turn/controller.py`'s own fixed replies already follow.
CONFIRMATION_UNAVAILABLE_REPLY = (
    "i can only change your calendar where i can hear your answer -- ask me with the wake word"
)
BULK_REFUSAL_REPLY = "i can only add or delete one event at a time -- make bulk changes in google calendar"
HANDOFF_NOT_ALONE_REPLY = "ask me for that on its own"
FOLLOW_UP_LIMIT_REPLY = "let's start over -- say the wake word and ask again"
PROPOSAL_INVALID_REPLY = "i couldn't work out how to add that -- try asking again"


class PendingProposal(BaseModel):
    """The validated shape of one `pending_action` handoff payload
    (`mcp/atlas_mcp/google.py::handle_calendar_propose_event`'s own return
    shape, minus the `kind` discriminator `atlas.turn.handoff.parse_handoff`
    already consumed).

    Impossible-by-construction, the same discipline `providers/tier_reply.py`
    already applies to `TierReply`: an end not after its own start, an empty
    title, a create proposal carrying an event id, and a naive (no time
    zone) start or end on a timed event are all unrepresentable -- Pydantic
    rejects each one before this object can exist, so nothing downstream of
    construction needs to re-check them.
    """

    action: Literal["calendar_create", "calendar_delete"]
    account: str
    calendar_id: str
    calendar_name: str
    calendar_primary: bool
    title: str
    start: datetime
    end: datetime
    all_day: bool
    time_zone: str
    event_id: "str | None" = None
    # Plan 09-05, Task 3: True when the event being deleted is one instance
    # of a recurring series (Google's own `recurringEventId` was present on
    # the fetched event) -- decides which of the two carriers
    # `compose_readback` below speaks, and never re-derived from anything
    # but the stored proposal itself (D-07). Meaningless, and always
    # `False`, on a `calendar_create` proposal.
    recurring_instance: bool = False

    @model_validator(mode="after")
    def _validate(self) -> "PendingProposal":
        if not self.all_day and (self.start.tzinfo is None or self.end.tzinfo is None):
            raise ValueError(
                "a pending proposal's start and end must be timezone-aware -- a naive "
                "datetime cannot be read back or executed unambiguously"
            )
        if self.end <= self.start:
            raise ValueError("a pending proposal's end must be after its start")
        if not self.title.strip():
            raise ValueError("a pending proposal must carry a non-empty title")
        if self.action == "calendar_create" and self.event_id is not None:
            raise ValueError("a calendar_create proposal must not carry an event id")
        if self.action == "calendar_delete" and self.event_id is None:
            raise ValueError("a calendar_delete proposal must carry the event's own id")
        return self


def execution_arguments(proposal: PendingProposal) -> dict[str, Any]:
    """The exact keyword arguments the executing tool
    (`EXECUTING_TOOL_BY_ACTION[proposal.action]`) is later called with --
    derived from the stored proposal alone, never re-parsed from a
    transcript or re-asked of a model (D-08)."""
    if proposal.action == "calendar_create":
        return {
            "account": proposal.account,
            "calendar_id": proposal.calendar_id,
            "title": proposal.title,
            "start": proposal.start.isoformat(),
            "end": proposal.end.isoformat(),
            "all_day": proposal.all_day,
            "time_zone": proposal.time_zone,
        }
    if proposal.action == "calendar_delete":
        return {
            "account": proposal.account,
            "calendar_id": proposal.calendar_id,
            "event_id": proposal.event_id,
        }
    raise ValueError(f"unknown pending action {proposal.action!r}")  # pragma: no cover - Literal-closed


_MONTH_NAMES: tuple[str, ...] = (
    "january", "february", "march", "april", "may", "june",
    "july", "august", "september", "october", "november", "december",
)
_WEEKDAY_NAMES: tuple[str, ...] = (
    "monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday",
)


def _ordinal(day: int) -> str:
    if 11 <= day % 100 <= 13:
        suffix = "th"
    else:
        suffix = {1: "st", 2: "nd", 3: "rd"}.get(day % 10, "th")
    return f"{day}{suffix}"


def _spoken_time(dt: datetime) -> str:
    """"at 3 pm" for an on-the-hour time, "at 3:30 pm" otherwise, "at 12
    pm"/"at 12 am" for noon/midnight -- never "at 0 pm" or "at 12:00 pm"."""
    hour24 = dt.hour
    minute = dt.minute
    period = "am" if hour24 < 12 else "pm"
    hour12 = hour24 % 12
    if hour12 == 0:
        hour12 = 12
    if minute:
        return f"{hour12}:{minute:02d} {period}"
    return f"{hour12} {period}"


def spoken_when(dt: datetime, *, all_day: bool, now: datetime) -> str:
    """"friday october 2nd at 3 pm" (or "friday october 2nd, all day" for
    an all-day event) -- the year is spoken only when `dt.year` differs
    from `now.year` (D-07: nothing here is a guess about what the operator
    already knows)."""
    date_part = f"{_WEEKDAY_NAMES[dt.weekday()]} {_MONTH_NAMES[dt.month - 1]} {_ordinal(dt.day)}"
    if dt.year != now.year:
        date_part = f"{date_part} {dt.year}"
    if all_day:
        return f"{date_part}, all day"
    return f"{date_part} at {_spoken_time(dt)}"


def spoken_duration(minutes: int) -> str:
    """"for an hour", "for an hour and a half", "for 2 hours", or "for 30
    minutes" -- an exact hour or half-hour speaks as hours, everything else
    speaks as minutes."""
    if minutes % 60 == 0:
        hours = minutes // 60
        if hours == 1:
            return "for an hour"
        return f"for {hours} hours"
    if minutes % 60 == 30:
        hours = minutes // 60
        if hours == 0:
            return f"for {minutes} minutes"
        if hours == 1:
            return "for an hour and a half"
        return f"for {hours} hours and a half"
    return f"for {minutes} minutes"


def compose_readback(proposal: PendingProposal, *, now: datetime) -> str:
    """The one spoken sentence read back for a proposal, composed entirely
    from `proposal`'s own fields (D-07) -- a primary calendar is named by
    its account's label alone ("the home calendar"); any other calendar
    names itself and its account ("the Team Offsite calendar in work").

    A `calendar_delete` proposal speaks one of two carriers: the plain one
    for a standalone event, and `_DELETE_OCCURRENCE_READBACK_CARRIER` --
    naming that the rest of the series stays untouched -- when
    `proposal.recurring_instance` is True (plan 09-05, Task 3, D-07).
    """
    if proposal.calendar_primary:
        calendar_phrase = f"the {proposal.account} calendar"
    else:
        calendar_phrase = f"the {proposal.calendar_name} calendar in {proposal.account}"

    when = spoken_when(proposal.start, all_day=proposal.all_day, now=now)

    if proposal.action == "calendar_delete":
        if proposal.recurring_instance:
            return _DELETE_OCCURRENCE_READBACK_CARRIER.format(
                title=proposal.title, when=when, calendar_phrase=calendar_phrase
            )
        return _DELETE_READBACK_CARRIER.format(title=proposal.title, calendar_phrase=calendar_phrase, when=when)

    if proposal.all_day:
        return f"add {proposal.title} to {calendar_phrase}, {when}?"

    duration_minutes = int((proposal.end - proposal.start).total_seconds() // 60)
    duration = spoken_duration(duration_minutes)
    return _CREATE_READBACK_CARRIER.format(
        title=proposal.title, calendar_phrase=calendar_phrase, when=when, duration=duration
    )
