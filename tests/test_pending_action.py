"""Plan 09-04 Task 1: a spoken "add dentist on Friday at 3" is stored as an
exact pending action and read back in full -- nothing in Google Calendar
changes yet. Drives the real `run_turn` through a fake tool host that calls
the child's own `handle_calendar_propose_event` in-process, following
`tests/test_turn_controller.py::_FakeToolHost`'s pattern.

Every account, calendar, and entity name below is invented -- no real
house appears in this file (the convention `tests/conftest.py` and
`mcp/atlas_mcp/safety.py` both state).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError

from atlas_mcp.google import handle_calendar_list_events, handle_calendar_propose_delete, handle_calendar_propose_event
from atlas_mcp.google_boundary import AccountGrant, CalendarGrant, Clarification, resolve_write_target
from atlas_mcp.ha import handle_call_service
from atlas_mcp.safety import Denied, Policy

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import FollowUpChannel
from atlas.turn.handoff import HandoffContext
from atlas.turn.pending_action import (
    BULK_REFUSAL_REPLY,
    CONFIRMATION_UNAVAILABLE_REPLY,
    PendingProposal,
    compose_readback,
    spoken_duration,
    spoken_when,
)

from google_fakes import FakeGoogle
from pending_action_fakes import FakePendingActionRepository

_ZONE = ZoneInfo("America/New_York")
_NOW = datetime(2026, 9, 24, 12, 0, tzinfo=timezone.utc)


def _home_account(*, calendars=None) -> AccountGrant:
    return AccountGrant(
        label="home",
        email="home@example.com",
        is_default=True,
        access_token="at-home",
        unreachable_reason=None,
        calendars=calendars
        if calendars is not None
        else (CalendarGrant(calendar_id="cal-home-primary", name="Home", primary=True, access="read_write"),),
    )


class _GoogleToolHost:
    """Calls straight into `atlas_mcp.google`'s and `atlas_mcp.ha`'s
    handler functions -- the same functional-double shape
    `tests/test_turn_controller.py::_FakeToolHost` already establishes for
    `atlas_mcp.ha`, extended to also reach `calendar_propose_event`.
    """

    def __init__(
        self,
        accounts,
        *,
        ha=None,
        policy: "Policy | None" = None,
        zone: ZoneInfo = _ZONE,
        google_client: "httpx.AsyncClient | None" = None,
    ) -> None:
        self._accounts = accounts
        self._zone = zone
        self._ha = ha
        self._policy = policy
        self._google_client = google_client
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        try:
            if name == "calendar_propose_event":
                result = await handle_calendar_propose_event(self._accounts, self._zone, **arguments)
            elif name == "calendar_propose_delete":
                result = await handle_calendar_propose_delete(
                    self._accounts, self._google_client, self._zone, **arguments
                )
            elif name == "calendar_list_events":
                result = await handle_calendar_list_events(
                    self._accounts, self._google_client, self._zone, **arguments
                )
            elif name == "ha_call_service" and self._ha is not None:
                result = await handle_call_service(
                    self._policy, self._ha.client, "http://ha.invalid", "test-token", **arguments
                )
            else:
                raise AssertionError(f"unknown tool: {name}")
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


# --- Task 1: the turn end to end -------------------------------------------


async def test_calendar_create_is_read_back_and_stored_not_executed(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00", "account": "home"},
                    )
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="add dentist on friday at 3")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == ["add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?"]
    assert timings.turn_outcome == "needs_confirmation"
    assert brain.call_count == 1
    assert [name for name, _ in tool_host.calls] == ["calendar_propose_event"]

    rows = list(pending_actions._rows.values())
    assert len(rows) == 1
    row = rows[0]
    assert row.status == "awaiting"
    assert row.tool_name == "calendar_insert_event"
    assert row.arguments["title"] == "Dentist"

    requested = source.follow_up.requested
    assert requested is not None
    assert requested.kind == "confirmation"
    assert requested.pending_action_id == row.id


async def test_no_follow_up_channel_speaks_confirmation_unavailable_and_stores_nothing(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00", "account": "home"},
                    )
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts)
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="add dentist on friday at 3")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == [CONFIRMATION_UNAVAILABLE_REPLY]
    assert timings.turn_outcome == "confirmation_unavailable"
    assert pending_actions._rows == {}


async def test_two_proposals_in_one_round_are_refused_as_bulk_change(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00", "account": "home"},
                    ),
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Vet", "start": "2026-10-03T15:00", "account": "home"},
                    ),
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="add dentist and vet appointments")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == [BULK_REFUSAL_REPLY]
    assert timings.turn_outcome == "bulk_refused"
    assert pending_actions._rows == {}


async def test_a_proposal_sharing_its_round_with_another_call_is_not_honoured(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_ha
):
    accounts = (_home_account(),)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00", "account": "home"},
                    ),
                    ToolCall(
                        name="ha_call_service",
                        arguments={
                            "domain": "switch",
                            "service": "turn_on",
                            "entity_id": "switch.example_fan",
                        },
                    ),
                ]
            ),
            BrainReply(text="i added the dentist appointment and turned on the fan"),
        ]
    )
    policy = Policy.from_config(None)
    tool_host = _GoogleToolHost(accounts, ha=fake_ha, policy=policy)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="add dentist and turn on the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar and a home",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert pending_actions._rows == {}
    assert len(fake_ha.requests) == 1
    reply = tts.received_text[0]
    assert "ask me for that on its own" in reply
    assert "succeeded" in reply


# --- compose_readback / spoken_when / spoken_duration -----------------------


def _proposal(**overrides) -> PendingProposal:
    fields = dict(
        action="calendar_create",
        account="home",
        calendar_id="cal-home-primary",
        calendar_name="Home",
        calendar_primary=True,
        title="Dentist",
        start=datetime(2026, 10, 2, 15, 0, tzinfo=_ZONE),
        end=datetime(2026, 10, 2, 16, 0, tzinfo=_ZONE),
        all_day=False,
        time_zone="America/New_York",
    )
    fields.update(overrides)
    return PendingProposal(**fields)


def test_compose_readback_renders_the_on_the_hour_example():
    proposal = _proposal()
    assert compose_readback(proposal, now=_NOW) == (
        "add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?"
    )


def test_compose_readback_renders_half_past_the_hour():
    proposal = _proposal(
        start=datetime(2026, 10, 2, 15, 30, tzinfo=_ZONE), end=datetime(2026, 10, 2, 16, 30, tzinfo=_ZONE)
    )
    assert "at 3:30 pm" in compose_readback(proposal, now=_NOW)


def test_compose_readback_renders_noon_as_12_pm():
    proposal = _proposal(
        start=datetime(2026, 10, 2, 12, 0, tzinfo=_ZONE), end=datetime(2026, 10, 2, 13, 0, tzinfo=_ZONE)
    )
    assert "at 12 pm" in compose_readback(proposal, now=_NOW)


def test_spoken_duration_examples():
    assert spoken_duration(90) == "for an hour and a half"
    assert spoken_duration(30) == "for 30 minutes"
    assert spoken_duration(120) == "for 2 hours"
    assert spoken_duration(60) == "for an hour"


def test_compose_readback_renders_an_all_day_event():
    proposal = _proposal(
        all_day=True, start=datetime(2026, 10, 2, 0, 0), end=datetime(2026, 10, 3, 0, 0)
    )
    assert compose_readback(proposal, now=_NOW) == "add Dentist to the home calendar, friday october 2nd, all day?"


def test_compose_readback_renders_a_non_primary_calendar():
    proposal = _proposal(calendar_primary=False, calendar_name="Team Offsite", account="work")
    assert "to the Team Offsite calendar in work" in compose_readback(proposal, now=_NOW)


def test_compose_readback_adds_the_year_only_when_it_differs():
    proposal = _proposal(
        start=datetime(2027, 10, 2, 15, 0, tzinfo=_ZONE), end=datetime(2027, 10, 2, 16, 0, tzinfo=_ZONE)
    )
    when = spoken_when(proposal.start, all_day=False, now=_NOW)
    assert "2027" in when
    same_year = spoken_when(datetime(2026, 10, 2, 15, 0, tzinfo=_ZONE), all_day=False, now=_NOW)
    assert "2026" not in same_year


# --- PendingProposal validation ---------------------------------------------


def test_pending_proposal_rejects_an_end_not_after_its_start():
    with pytest.raises(ValidationError):
        _proposal(end=datetime(2026, 10, 2, 14, 0, tzinfo=_ZONE))


def test_pending_proposal_rejects_an_empty_title():
    with pytest.raises(ValidationError):
        _proposal(title="   ")


def test_pending_proposal_rejects_a_create_carrying_an_event_id():
    with pytest.raises(ValidationError):
        _proposal(event_id="evt-123")


def test_pending_proposal_rejects_a_naive_datetime():
    with pytest.raises(ValidationError):
        _proposal(
            start=datetime(2026, 10, 2, 15, 0), end=datetime(2026, 10, 2, 16, 0)
        )


# --- google_boundary.resolve_write_target / handle_calendar_propose_event --


def test_resolve_write_target_denies_a_calendar_that_is_off():
    account = _home_account()
    with pytest.raises(Denied):
        resolve_write_target((account,), "home", "Nonexistent")


def test_resolve_write_target_denies_a_read_only_calendar():
    account = _home_account(
        calendars=(CalendarGrant(calendar_id="cal-ro", name="Read Only", primary=True, access="read_only"),)
    )
    with pytest.raises(Denied):
        resolve_write_target((account,), "home", "Read Only")


def test_resolve_write_target_asks_when_a_named_account_has_two_writable_calendars_and_none_named():
    account = _home_account(
        calendars=(
            CalendarGrant(calendar_id="cal-a", name="Home", primary=False, access="read_write"),
            CalendarGrant(calendar_id="cal-b", name="Chores", primary=False, access="read_write"),
        )
    )
    result = resolve_write_target((account,), "home", None)
    assert isinstance(result, Clarification)
    assert result.about == "calendar"
    assert result.candidates == ("Chores", "Home")


async def test_handle_calendar_propose_event_makes_no_http_call_and_defaults_to_an_hour():
    accounts = (_home_account(),)
    payload = await handle_calendar_propose_event(
        accounts, _ZONE, title="Dentist", start="2026-10-02T15:00", account="home"
    )
    body = payload["atlas_handoff"]
    assert body["kind"] == "pending_action"
    assert body["action"] == "calendar_create"
    assert body["start"] == "2026-10-02T15:00:00-04:00"
    assert body["end"] == "2026-10-02T16:00:00-04:00"


# --- Task 2: the right account when none is named ---------------------------


def _account(label: str, *, is_default: bool = False, access_token: str | None = "at", calendars=None) -> AccountGrant:
    return AccountGrant(
        label=label,
        email=f"{label}@example.com",
        is_default=is_default,
        access_token=access_token,
        unreachable_reason=None if access_token is not None else "needs_relink",
        calendars=calendars
        if calendars is not None
        else (CalendarGrant(calendar_id=f"cal-{label}-primary", name=label.title(), primary=True, access="read_write"),),
    )


def test_resolve_write_target_unnamed_picks_the_only_account_with_a_writable_calendar():
    home = _account("home")
    work = _account(
        "work",
        calendars=(CalendarGrant(calendar_id="cal-work-primary", name="Work", primary=True, access="read_only"),),
    )
    result = resolve_write_target((home, work), None, None)
    assert result == (home, home.calendars[0])


def test_resolve_write_target_unnamed_picks_the_default_when_several_are_writable():
    home = _account("home", is_default=False)
    work = _account("work", is_default=True)
    result = resolve_write_target((home, work), None, None)
    assert result == (work, work.calendars[0])


def test_resolve_write_target_unnamed_asks_when_several_writable_and_no_default():
    home = _account("home")
    work = _account("work")
    result = resolve_write_target((home, work), None, None)
    assert isinstance(result, Clarification)
    assert result.about == "account"
    assert result.candidates == ("home", "work")


def test_resolve_write_target_unnamed_denies_when_no_account_has_a_writable_calendar():
    home = _account(
        "home",
        calendars=(CalendarGrant(calendar_id="cal-home", name="Home", primary=True, access="read_only"),),
    )
    with pytest.raises(Denied):
        resolve_write_target((home,), None, None)


def test_resolve_write_target_denies_a_named_account_with_no_access_token():
    home = _account("home", access_token=None)
    with pytest.raises(Denied):
        resolve_write_target((home,), "home", None)


async def test_no_account_named_and_several_candidates_asks_which_one_and_stores_nothing(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    home = _account("home")
    work = _account("work")
    accounts = (home, work)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00"},
                    )
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="add a dentist appointment")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == ["i'm not sure which one you mean -- home, work?"]
    assert timings.turn_outcome == "needs_clarification"
    assert pending_actions._rows == {}


# --- Task 3: build_handoff_context degrades cleanly with no repository -----


def test_build_handoff_context_carries_none_when_the_app_state_has_no_repository():
    from types import SimpleNamespace as NS

    from atlas.google.turn_context import build_handoff_context

    app = NS(state=NS())  # no pending_action_repo, tool_host_lookup, or brain attribute at all
    ctx = build_handoff_context(app, "camera")
    assert ctx.pending_actions is None
    assert ctx.tool_host is None
    assert ctx.brain is None


async def test_a_proposal_with_a_repository_free_context_speaks_confirmation_unavailable(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    from atlas.google.turn_context import build_handoff_context

    accounts = (_home_account(),)
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T15:00", "account": "home"},
                    )
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="add dentist on friday at 3")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    app = SimpleNamespace(state=SimpleNamespace())  # no pending_action_repo attribute at all
    handoff_context = build_handoff_context(app, "camera")

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == [CONFIRMATION_UNAVAILABLE_REPLY]
    assert timings.turn_outcome == "confirmation_unavailable"


# --- Task 3: delete proposals -------------------------------------------------


def _work_account(*, calendars=None) -> AccountGrant:
    return AccountGrant(
        label="work",
        email="work@example.com",
        is_default=True,
        access_token="at-work",
        unreachable_reason=None,
        calendars=calendars
        if calendars is not None
        else (CalendarGrant(calendar_id="cal-work", name="Team", primary=True, access="read_write"),),
    )


async def test_a_single_occurrence_delete_is_read_back_and_stored_with_its_own_instance_id(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_work_account(),)
    fake_google = FakeGoogle()
    fake_google.add_event(
        "at-work",
        "cal-work",
        {
            "id": "evt-1",
            "summary": "Standup",
            "start": {"dateTime": "2026-10-06T09:00:00-04:00"},
            "end": {"dateTime": "2026-10-06T09:15:00-04:00"},
        },
    )
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_delete",
                        arguments={"account": "work", "calendar_id": "cal-work", "event_id": "evt-1"},
                    )
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="delete tuesday's standup")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == ["delete Standup from the work calendar, tuesday october 6th at 9 am?"]
    assert timings.turn_outcome == "needs_confirmation"
    rows = list(pending_actions._rows.values())
    assert len(rows) == 1
    row = rows[0]
    assert row.tool_name == "calendar_delete_event"
    assert row.arguments["event_id"] == "evt-1"


async def test_a_recurring_occurrence_delete_speaks_the_series_stays_carrier_and_stores_the_instance_id(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_work_account(),)
    fake_google = FakeGoogle()
    fake_google.add_event(
        "at-work",
        "cal-work",
        {
            "id": "evt-1_20261006T090000Z",
            "recurringEventId": "evt-1",
            "summary": "Standup",
            "start": {"dateTime": "2026-10-06T09:00:00-04:00"},
            "end": {"dateTime": "2026-10-06T09:15:00-04:00"},
        },
    )
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_delete",
                        arguments={
                            "account": "work",
                            "calendar_id": "cal-work",
                            "event_id": "evt-1_20261006T090000Z",
                        },
                    )
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="delete tuesday's standup")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == [
        "delete only this one: Standup on tuesday october 6th at 9 am, from the work calendar? "
        "the rest of the series stays."
    ]
    row = list(pending_actions._rows.values())[0]
    assert row.arguments["event_id"] == "evt-1_20261006T090000Z"
    assert row.arguments["event_id"] != "evt-1"


async def test_bulk_delete_refused_by_voice(fake_audio_source, fake_stt, fake_brain, fake_tts):
    accounts = (_work_account(),)
    fake_google = FakeGoogle()
    fake_google.add_event(
        "at-work",
        "cal-work",
        {
            "id": "evt-1",
            "summary": "Standup",
            "start": {"dateTime": "2026-10-06T09:00:00-04:00"},
            "end": {"dateTime": "2026-10-06T09:15:00-04:00"},
        },
    )
    fake_google.add_event(
        "at-work",
        "cal-work",
        {
            "id": "evt-2",
            "summary": "Vet",
            "start": {"dateTime": "2026-10-07T09:00:00-04:00"},
            "end": {"dateTime": "2026-10-07T09:15:00-04:00"},
        },
    )
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_delete",
                        arguments={"account": "work", "calendar_id": "cal-work", "event_id": "evt-1"},
                    ),
                    ToolCall(
                        name="calendar_propose_delete",
                        arguments={"account": "work", "calendar_id": "cal-work", "event_id": "evt-2"},
                    ),
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel()
    stt = fake_stt(events=[FinalTranscript(text="delete standup and vet")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    pending_actions = FakePendingActionRepository()
    handoff_context = HandoffContext(
        source_name="camera", tool_host=tool_host, pending_actions=pending_actions, brain=brain, now=_NOW
    )

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert tts.received_text == [BULK_REFUSAL_REPLY]
    assert timings.turn_outcome == "bulk_refused"
    assert pending_actions._rows == {}
    assert fake_google.deleted_event_ids == []
