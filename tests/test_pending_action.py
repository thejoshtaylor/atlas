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

from atlas_mcp.google import (
    handle_calendar_delete_event,
    handle_calendar_insert_event,
    handle_calendar_list_events,
    handle_calendar_propose_delete,
    handle_calendar_propose_event,
)
from atlas_mcp.google_boundary import AccountGrant, CalendarGrant, Clarification, resolve_write_target
from atlas_mcp.google_tools import PROPOSAL_TURN_TOOL_NAMES
from atlas_mcp.ha import handle_call_service
from atlas_mcp.safety import Denied, Policy

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import FollowUpChannel, FollowUpRequest
from atlas.turn.handoff import (
    AMENDED_CONTINUATION_REFUSAL,
    Handoff,
    HandoffContext,
    dispatch_handoff,
)
from atlas.turn.pending_action import (
    BULK_REFUSAL_REPLY,
    CONFIRM_CANCEL_TOOLS,
    CONFIRMATION_UNAVAILABLE_REPLY,
    CONFIRMED_CREATE_REPLY,
    EXECUTION_DID_NOT_COMPLETE_REPLY,
    PROPOSAL_STORE_FAILED_REPLY,
    PendingProposal,
    compose_readback,
    execute_pending_action,
    run_confirmation_round,
    spoken_duration,
    spoken_when,
)

from brain_fakes import RecordingFakeBrain
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
            elif name == "calendar_insert_event":
                result = await handle_calendar_insert_event(self._accounts, self._google_client, **arguments)
            elif name == "calendar_delete_event":
                result = await handle_calendar_delete_event(self._accounts, self._google_client, **arguments)
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


def test_pending_proposal_flattens_and_caps_an_oversized_title():
    """A-CR-01 defense in depth: `calendar_propose_delete`'s own title is
    Google's own raw event summary, never capped by
    `mcp/atlas_mcp/google.py::_sanitize_title` the way a create proposal's
    title already is -- `PendingProposal` itself must never carry an
    unbounded or multi-line title into a readback or a confirmation-round
    message, regardless of which handler built the proposal."""
    proposal = _proposal(title="line one\r\nline two\n" + "x" * 200)
    assert "\n" not in proposal.title
    assert "\r" not in proposal.title
    assert len(proposal.title) == 120
    assert proposal.title.startswith("line one line two")


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


async def test_handle_calendar_propose_event_all_day_accepts_a_bare_date():
    accounts = (_home_account(),)
    payload = await handle_calendar_propose_event(
        accounts, _ZONE, title="Dentist", start="2026-10-02", account="home", all_day=True
    )
    body = payload["atlas_handoff"]
    assert body["all_day"] is True
    assert body["start"] == "2026-10-02"
    assert body["end"] == "2026-10-03"


async def test_handle_calendar_propose_event_all_day_denies_a_malformed_date_instead_of_crashing():
    """B1-WR-01 regression: the all-day branch must raise `Denied` (routed
    through `except Denied` at the tool boundary, a spoken refusal) on a
    malformed date, the same way every other date/time value this module
    parses already does -- never a bare, unhandled `ValueError`."""
    accounts = (_home_account(),)
    with pytest.raises(Denied):
        await handle_calendar_propose_event(
            accounts, _ZONE, title="Dentist", start="not-a-date", account="home", all_day=True
        )


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


def test_resolve_write_target_unnamed_refuses_an_unreachable_sole_candidate_by_name():
    """B1-WR-02, R2-WR-08: an unreachable account (GOOG-12, `access_token
    is None`) with a read-write calendar is refused up front, before any
    readback. The refusal names the real problem -- the account cannot be
    reached -- not a calendar that is already turned on."""
    home = _account("home", access_token=None)
    with pytest.raises(Denied) as excinfo:
        resolve_write_target((home,), None, None)
    assert str(excinfo.value) == "i can't reach your home account right now"


def test_resolve_write_target_unnamed_never_redirects_away_from_an_unreachable_default():
    """R2-WR-08: D-04 sends a write with no signal to the default account,
    and GOOG-12 never skips an unreachable account silently. An unreachable
    default is refused by name -- the write never lands on another
    account without the operator knowing."""
    home = _account("home", is_default=True, access_token=None)
    work = _account("work", is_default=False)
    with pytest.raises(Denied) as excinfo:
        resolve_write_target((home, work), None, None)
    assert str(excinfo.value) == "i can't reach your home account right now"


def test_resolve_write_target_unnamed_asks_over_every_writable_account_when_one_is_unreachable():
    """No default and two writable accounts, one unreachable: the question
    names both. Naming the unreachable one then gets the named path's own
    "i can't reach" refusal, never a silent skip."""
    home = _account("home", access_token=None)
    work = _account("work")
    result = resolve_write_target((home, work), None, None)
    assert isinstance(result, Clarification)
    assert result.candidates == ("home", "work")


def test_resolve_write_target_unnamed_denial_with_no_writable_calendar_names_the_calendar_setting():
    home = _account(
        "home",
        calendars=(CalendarGrant(calendar_id="cal-home", name="Home", primary=True, access="read_only"),),
    )
    with pytest.raises(Denied) as excinfo:
        resolve_write_target((home,), None, None)
    assert "turn on a calendar for read and write" in str(excinfo.value)


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


# --- Plan 09-06 Task 1: the restricted confirmation round --------------------


class _RecordingConfirmationBrain:
    """Records every `chat()` call's own `messages`/`tools`, and answers
    with exactly one scripted tool call -- `confirm`, or `cancel` (with
    `amended` when asked) -- the same shape a restricted confirmation
    round is ever offered."""

    def __init__(self, decision: str, *, amended: bool = False) -> None:
        self._decision = decision
        self._amended = amended
        self.calls: list[dict] = []

    async def chat(self, messages, tools=None):
        self.calls.append({"messages": messages, "tools": tools})
        arguments = {"amended": True} if self._decision == "cancel" and self._amended else {}
        return BrainReply(tool_calls=[ToolCall(name=self._decision, arguments=arguments)])


async def test_calendar_create_requires_confirmation_round(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    confirmation_brain = _RecordingConfirmationBrain("confirm")
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="yes")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    # Never called: an incoming confirmation bypasses the tier race
    # entirely (Task 1's own `<behavior>`: "exactly one brain call").
    brain = fake_brain(replies=[])

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

    assert len(confirmation_brain.calls) == 1
    call = confirmation_brain.calls[0]
    assert call["tools"] == CONFIRM_CANCEL_TOOLS
    # A-CR-01, R2-WR-03: the readback is JSON data in the first user
    # message, and the operator's own reply is the second, unlabeled --
    # neither ever occupies the system role, which carries only the fixed
    # instruction, free of any proposal-derived text.
    assert [message["role"] for message in call["messages"]] == ["system", "user", "user"]
    assert created.readback not in call["messages"][0]["content"]
    assert json.loads(call["messages"][1]["content"])["question_you_asked"] == created.readback
    assert call["messages"][2]["content"] == "yes"

    assert tts.received_text == [CONFIRMED_CREATE_REPLY]
    assert timings.turn_outcome == "confirmed"
    assert [name for name, _ in tool_host.calls] == ["calendar_insert_event"]
    _, insert_arguments = tool_host.calls[0]
    assert insert_arguments["title"] == "Dentist"
    assert fake_google.inserted[0]["body"]["summary"] == "Dentist"

    row = await pending_actions.get(created.id)
    assert row.status == "executed"


@pytest.mark.parametrize("transcript", ["okay", "OK.", "sounds good", "absolutely", "that works", "uh-huh"])
async def test_run_confirmation_round_follows_the_model_confirm_for_common_agreement(transcript):
    """R2-WR-01, D-08: the operator chose model interpretation over a fixed
    yes-word list. A `confirm` call is the decision -- no word list in code
    can veto a real "okay" or "sounds good"."""
    decision = await run_confirmation_round(
        _RecordingConfirmationBrain("confirm"),
        readback="add dentist to the home calendar, friday at 3 pm, for an hour?",
        transcript=transcript,
        timeout_s=5.0,
    )

    assert decision.decision == "confirm"


@pytest.mark.parametrize("transcript", ["no, don't do that", "not right now", "please cancel it", "no, go back"])
async def test_run_confirmation_round_follows_the_model_cancel_for_a_negated_reply(transcript):
    decision = await run_confirmation_round(
        _RecordingConfirmationBrain("cancel"),
        readback="add dentist to the home calendar, friday at 3 pm, for an hour?",
        transcript=transcript,
        timeout_s=5.0,
    )

    assert decision.decision == "cancel"
    assert decision.amended is False


async def test_run_confirmation_round_settles_on_cancel_for_a_non_brain_error_exception(caplog):
    """R3-WR-01: no provider in this codebase raises `BrainError` -- the
    `openai` SDK's own exceptions (a connection error, a 5xx status error)
    propagate as themselves. The round must settle on `cancel` and log the
    failure for any of them, not just the two named exceptions, so the
    operator who just said "yes" never hears silence."""

    class _ConnectionFailingBrain:
        async def chat(self, messages, tools=None):
            raise RuntimeError("connection reset by peer")

    with caplog.at_level("ERROR"):
        decision = await run_confirmation_round(
            _ConnectionFailingBrain(),
            readback="add dentist to the home calendar, friday at 3 pm, for an hour?",
            transcript="yes",
            timeout_s=5.0,
        )

    assert decision.decision == "cancel"
    assert decision.amended is False
    assert any("confirmation round failed" in record.getMessage() for record in caplog.records)


async def test_run_confirmation_round_settles_on_cancel_for_malformed_streamed_tool_call_arguments(caplog):
    """A `json.JSONDecodeError` from `accumulate_stream` reading malformed
    streamed tool-call arguments is exactly the shape of exception this
    round previously let escape uncaught."""

    class _MalformedStreamBrain:
        async def chat(self, messages, tools=None):
            raise json.JSONDecodeError("Expecting value", "", 0)

    with caplog.at_level("ERROR"):
        decision = await run_confirmation_round(
            _MalformedStreamBrain(),
            readback="add dentist to the home calendar, friday at 3 pm, for an hour?",
            transcript="yes",
            timeout_s=5.0,
        )

    assert decision.decision == "cancel"
    assert any("confirmation round failed" in record.getMessage() for record in caplog.records)


async def test_run_confirmation_round_leaves_cancellation_to_propagate():
    """`asyncio.CancelledError` is task cancellation, not a round failure --
    it must never be rewritten into a `cancel` decision."""
    import asyncio

    class _CancellingBrain:
        async def chat(self, messages, tools=None):
            raise asyncio.CancelledError()

    with pytest.raises(asyncio.CancelledError):
        await run_confirmation_round(
            _CancellingBrain(),
            readback="add dentist to the home calendar, friday at 3 pm, for an hour?",
            transcript="yes",
            timeout_s=5.0,
        )


# --- A-WR-01: exception handling on the pending-action/email-draft paths ----


async def test_dispatch_handoff_speaks_a_fixed_reply_when_pending_actions_create_raises():
    """A-WR-01 regression: `dispatch_handoff`'s pending-action branch must
    never abort the turn silently before the readback is ever spoken."""

    class _RaisingPendingActions:
        async def create(self, **kwargs: Any) -> Any:
            raise RuntimeError("db unavailable")

    handoff = Handoff(
        kind="pending_action",
        payload={
            "kind": "pending_action",
            "action": "calendar_create",
            "account": "home",
            "calendar_id": "cal-home-primary",
            "calendar_name": "Home",
            "calendar_primary": True,
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
    )
    ctx = HandoffContext(
        source_name="camera",
        tool_host=None,
        pending_actions=_RaisingPendingActions(),
        brain=None,
        now=_NOW,
    )

    outcome = await dispatch_handoff(
        handoff, ctx, transcript="add dentist on friday at 3", follow_up_available=True, chain_depth=1
    )

    assert outcome.turn_outcome == "proposal_store_failed"
    assert outcome.reply_text == PROPOSAL_STORE_FAILED_REPLY
    assert outcome.follow_up is None


async def test_execute_pending_action_speaks_a_fixed_reply_when_the_tool_host_raises():
    """A-WR-01 regression: `execute_pending_action`'s own `tool_host.call_tool`
    raising must produce a failed, not-succeeded `ExecutionResult` -- never
    propagate and leave the caller (`handle_confirmation_reply`) with a row
    already claimed `confirmed` and no terminal status to resolve it to."""

    class _RaisingToolHost:
        async def call_tool(self, name: str, arguments: dict) -> Any:
            raise RuntimeError("stdio child died mid-call")

    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    claimed = await pending_actions.claim_for_confirmation(created.id, _NOW)
    assert claimed is not None

    result = await execute_pending_action(claimed, _RaisingToolHost())

    assert result.succeeded is False
    assert result.reply_text == EXECUTION_DID_NOT_COMPLETE_REPLY


async def test_a_raised_tool_host_exception_during_confirm_resolves_the_row_to_failed_not_stuck(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """A-WR-01 regression, end to end: a `tool_host.call_tool` raise on the
    confirm path must resolve the row to a terminal `failed` status --
    never leave it stuck at `confirmed` forever -- and the operator must
    hear something, not silence."""

    class _RaisingToolHost:
        async def call_tool(self, name: str, arguments: dict) -> Any:
            raise RuntimeError("stdio child died mid-call")

    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    confirmation_brain = _RecordingConfirmationBrain("confirm")
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=_RaisingToolHost(),
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="yes")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    brain = fake_brain(replies=[])

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    assert timings.turn_outcome == "confirm_failed"
    assert tts.received_text == [EXECUTION_DID_NOT_COMPLETE_REPLY]
    row = await pending_actions.get(created.id)
    assert row.status == "failed"


async def test_a_stored_delete_confirms_and_the_fake_google_records_one_delete(
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
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_delete",
        tool_name="calendar_delete_event",
        arguments={"account": "work", "calendar_id": "cal-work", "event_id": "evt-1"},
        readback="delete Standup from the work calendar, tuesday october 6th at 9 am?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    confirmation_brain = _RecordingConfirmationBrain("confirm")
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="delete tuesday's standup",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="yes")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    brain = fake_brain(replies=[])

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

    assert tts.received_text == ["done, it's deleted"]
    assert timings.turn_outcome == "confirmed"
    assert fake_google.deleted_event_ids == ["evt-1"]
    row = await pending_actions.get(created.id)
    assert row.status == "executed"


async def test_a_cancel_call_resolves_cancelled_and_speaks_the_cancelled_phrase(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    confirmation_brain = _RecordingConfirmationBrain("cancel")
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="no")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    brain = fake_brain(replies=[])

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

    assert tts.received_text == ["cancelled, nothing was changed"]
    assert timings.turn_outcome == "cancelled"
    assert fake_google.inserted == []
    row = await pending_actions.get(created.id)
    assert row.status == "cancelled"


async def test_an_unreadable_reply_cancels_the_same_as_an_explicit_no(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    # A brain that calls neither confirm nor cancel -- `decision_from_reply`
    # settles this on `cancel` (D-08, D-11), same as an explicit no.
    class _NoToolCallBrain:
        async def chat(self, messages, tools=None):
            return BrainReply(text="i'm not sure what you mean")

    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=_NoToolCallBrain(),
        now=_NOW + timedelta(seconds=5),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="what do you mean")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    brain = fake_brain(replies=[])

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

    assert tts.received_text == ["cancelled, nothing was changed"]
    assert timings.turn_outcome == "cancelled"
    assert fake_google.inserted == []
    row = await pending_actions.get(created.id)
    assert row.status == "cancelled"


async def test_silence_resolves_the_row_expired_not_cancelled(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    # A brain that would be called if the round ran at all -- it must
    # not be, since silence never reaches the confirmation round.
    confirmation_brain = _RecordingConfirmationBrain("confirm")
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    brain = fake_brain(replies=[])

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

    assert tts.received_text == ["cancelled, nothing was changed"]
    assert timings.turn_outcome == "follow_up_silence"
    assert confirmation_brain.calls == []
    row = await pending_actions.get(created.id)
    assert row.status == "expired"


async def test_a_confirm_for_an_expired_row_executes_nothing(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        # Already expired by the time the confirm round runs.
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=1),
    )
    confirmation_brain = _RecordingConfirmationBrain("confirm")
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=30),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="yes")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    brain = fake_brain(replies=[])

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

    assert tts.received_text == ["cancelled, nothing was changed"]
    assert timings.turn_outcome == "confirm_expired"
    assert fake_google.inserted == []
    row = await pending_actions.get(created.id)
    assert row.status == "awaiting"  # untouched -- claim_for_confirmation never succeeded


async def test_an_executing_tool_error_speaks_the_error_verbatim_and_resolves_failed(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(calendars=(CalendarGrant(
        calendar_id="cal-home-primary", name="Home", primary=True, access="read_only"
    ),)),)
    fake_google = FakeGoogle()
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    confirmation_brain = _RecordingConfirmationBrain("confirm")
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="yes")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    brain = fake_brain(replies=[])

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

    assert timings.turn_outcome == "confirm_failed"
    assert tts.received_text != ["done, it's on your calendar"]
    assert "read" in tts.received_text[0].lower() or "denied" in tts.received_text[0].lower() or tts.received_text[0]
    row = await pending_actions.get(created.id)
    assert row.status == "failed"


async def test_an_amendment_supersedes_and_produces_a_new_readback_with_deeper_chain_depth(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    confirmation_brain = _RecordingConfirmationBrain("cancel", amended=True)

    # The ordinary pipeline's own brain -- the amendment reaches this one,
    # not the confirmation round's brain, and proposes a new time.
    ordinary_brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T16:00", "account": "home"},
                    )
                ]
            ),
        ]
    )
    tool_host = _GoogleToolHost(accounts, google_client=fake_google.client)
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
        # R3-IN-03: the default is now `frozenset()` (fail closed) -- the
        # amended continuation below re-proposes through the restricted
        # round, which needs the bare proposal-tool names to be offered.
        proposal_tool_names=PROPOSAL_TURN_TOOL_NAMES,
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="yes, but make it 4")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        ordinary_brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    # The superseded row is resolved, and exactly one new row exists.
    original = await pending_actions.get(created.id)
    assert original.status == "superseded"
    rows = [row for row in pending_actions._rows.values() if row.id != created.id]
    assert len(rows) == 1
    new_row = rows[0]
    assert new_row.status == "awaiting"
    assert new_row.arguments["start"] == "2026-10-02T16:00:00-04:00"

    # The ordinary brain's own single call saw the prior exchange ahead
    # of the amendment -- not the confirmation brain, which was never
    # asked to phrase a new proposal.
    assert ordinary_brain.call_count == 1

    # A new follow-up was requested, one chain_depth deeper than the
    # incoming one.
    requested = source.follow_up.requested
    assert requested is not None
    assert requested.kind == "confirmation"
    assert requested.chain_depth == 2


async def test_an_amended_continuation_offering_ha_call_service_never_executes_it(
    fake_audio_source, fake_stt, fake_tts, fake_ha
):
    """A-CR-02 regression: the one turn that continues an `amended`
    confirmation reply must never let a tool call reach anything but a
    fresh calendar proposal -- even when the ordinary tier's own brain
    names an unrelated action tool (`ha_call_service`), simulating a
    model persuaded by, or simply confused about, the open-mic window's
    own untrusted transcript. Proves the property at the one place that
    actually matters: `fake_ha.requests` (the real HTTP transport a
    dispatched `ha_call_service` call would have to reach) stays empty.
    """
    accounts = (_home_account(),)
    fake_google = FakeGoogle()
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={
            "account": "home",
            "calendar_id": "cal-home-primary",
            "title": "Dentist",
            "start": "2026-10-02T15:00:00-04:00",
            "end": "2026-10-02T16:00:00-04:00",
            "all_day": False,
            "time_zone": "America/New_York",
        },
        readback="add Dentist to the home calendar, friday october 2nd at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )
    confirmation_brain = _RecordingConfirmationBrain("cancel", amended=True)

    # The ordinary pipeline's own brain -- reached only after the amended
    # decision above -- names a tool that has nothing to do with a
    # calendar proposal, exactly the escape hatch A-CR-02 describes.
    ordinary_brain = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={"domain": "lock", "service": "unlock", "entity_id": "lock.front_door"},
                    )
                ]
            ),
        ]
    )
    policy = Policy.from_config(None)
    tool_host = _GoogleToolHost(accounts, ha=fake_ha, policy=policy, google_client=fake_google.client)
    handoff_context = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=confirmation_brain,
        now=_NOW + timedelta(seconds=5),
        # R3-IN-03: the default is now `frozenset()` (fail closed) -- this
        # test asserts the ordinary tier is offered exactly the two bare
        # proposal-tool names, which requires the real set here.
        proposal_tool_names=PROPOSAL_TURN_TOOL_NAMES,
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    stt = fake_stt(events=[FinalTranscript(text="yes, but also unlock the front door")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    # A real, non-empty tools_schema the ordinary tier is nominally
    # offered -- both proposal tools and `ha_call_service` -- so the test
    # proves the restriction is enforced at dispatch, not merely by what
    # this list happens to contain.
    full_tools_schema = [
        {"type": "function", "function": {"name": "calendar_propose_event"}},
        {"type": "function", "function": {"name": "calendar_propose_delete"}},
        {"type": "function", "function": {"name": "ha_call_service"}},
    ]

    await run_turn(
        source,
        stt,
        ordinary_brain,
        tts,
        tool_host,
        tools_schema=full_tools_schema,
        system_prompt="you manage a calendar and a home",
        max_tool_rounds=3,
        timings=timings,
        handoff_context=handoff_context,
    )

    # The one assertion that actually proves "never executes": no request
    # ever reached the fake Home Assistant transport.
    assert fake_ha.requests == []

    # The ordinary tier was offered only the two proposal tools -- the
    # schema restriction that backs up the dispatch-time check above.
    assert ordinary_brain.call_count == 1
    offered_names = {entry["function"]["name"] for entry in ordinary_brain.calls[0].tools}
    assert offered_names == {"calendar_propose_event", "calendar_propose_delete"}

    # The turn ends with the fixed refusal, not silence and not a second
    # brain round pretending the call succeeded.
    assert tts.received_text == [AMENDED_CONTINUATION_REFUSAL]
