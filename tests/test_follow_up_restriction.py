"""A-CR-02 regression: once a turn is proposal-restricted, every follow-up
it spawns stays proposal-restricted, however many links later.

The turn that continues an `amended` confirmation reply may only build a
new calendar proposal (D-08). Before this fix, that restriction was a
local variable of one turn. A clarification that the restricted turn
opened -- from `calendar_propose_event` itself, or from a triage tier's
own `needs_clarification` reply -- gave the next no-wake-word turn the
full tool set. The first test below is the reviewer's own proof of
concept, turned around so that it asserts the safe outcome.

Every account, calendar, and entity name below is invented.
"""

from __future__ import annotations

from datetime import timedelta

from atlas_mcp.google_tools import PROPOSAL_TURN_TOOL_NAMES
from atlas_mcp.safety import Policy

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.providers.tier_reply import FillerPhrase, TierReply
from atlas.timing import TurnTimings
from atlas.turn import brain_race
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import FollowUpChannel, FollowUpRequest
from atlas.turn.handoff import AMENDED_CONTINUATION_REFUSAL, HandoffContext

import test_pending_action as tpa
from brain_fakes import RecordingFakeBrain
from google_fakes import FakeGoogle
from pending_action_fakes import FakePendingActionRepository

_SCHEMA = [
    {"type": "function", "function": {"name": "calendar_propose_event"}},
    {"type": "function", "function": {"name": "calendar_propose_delete"}},
    {"type": "function", "function": {"name": "ha_call_service"}},
]

_UNLOCK = ToolCall(
    name="ha_call_service",
    arguments={"domain": "lock", "service": "unlock", "entity_id": "lock.front_door"},
)


async def _stored_create(pending_actions: FakePendingActionRepository):
    return await pending_actions.create(
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
        created_at=tpa._NOW,
        expires_at=tpa._NOW + timedelta(seconds=60),
    )


def _two_accounts_no_default():
    # Two writable accounts and no default: an unnamed create asks which one.
    return (tpa._account("home"), tpa._account("work"))


def _context(tool_host, pending_actions, brain) -> HandoffContext:
    # R3-IN-03: `HandoffContext.proposal_tool_names` now defaults to
    # `frozenset()` (fail closed). These tests exercise a restricted round
    # that must actually be able to propose a calendar event or delete, so
    # they pass the bare names explicitly -- the same names a real running
    # Google plugin's own `offered_tool_names_for_module` would resolve to.
    return HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=pending_actions,
        brain=brain,
        now=tpa._NOW + timedelta(seconds=5),
        proposal_tool_names=PROPOSAL_TURN_TOOL_NAMES,
    )


async def _turn(incoming, transcript, brain, tool_host, ctx, *, fakes, tiers=None):
    fake_audio_source, fake_stt, fake_tts = fakes
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    tts = fake_tts(chunks=[b"\x01"])
    await run_turn(
        source,
        fake_stt(events=[FinalTranscript(text=transcript)]),
        brain,
        tts,
        tool_host,
        tools_schema=_SCHEMA,
        system_prompt="you manage a calendar and a home",
        max_tool_rounds=3,
        timings=TurnTimings(),
        handoff_context=ctx,
        tiers=tiers,
    )
    return source.follow_up.requested, tts


async def test_an_amended_chain_into_a_google_clarification_never_reaches_ha_call_service(
    fake_audio_source, fake_stt, fake_tts, fake_ha
):
    """The reviewer's proof of concept, asserting the safe outcome: the
    Home Assistant unlock is never offered and never executed."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    pending_actions = FakePendingActionRepository()
    created = await _stored_create(pending_actions)
    tool_host = tpa._GoogleToolHost(
        _two_accounts_no_default(), ha=fake_ha, policy=Policy.from_config(None), google_client=FakeGoogle().client
    )
    ctx = _context(tool_host, pending_actions, tpa._RecordingConfirmationBrain("cancel", amended=True))

    # Turn 1: the amended continuation re-proposes with no account named.
    brain1 = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(name="calendar_propose_event", arguments={"title": "Dentist", "start": "2026-10-02T16:00"})
                ]
            )
        ]
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    requested, _ = await _turn(incoming, "no, make it 4 on some other account", brain1, tool_host, ctx, fakes=fakes)
    assert requested is not None and requested.kind == "clarification"
    assert requested.proposals_only is True

    # Turn 2: the clarification window (no wake word) asks for an unlock.
    brain2 = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_UNLOCK]), BrainReply(text="done")])
    _, tts = await _turn(requested, "home, and unlock the front door", brain2, tool_host, ctx, fakes=fakes)

    offered = {entry["function"]["name"] for entry in brain2.calls[0].tools}
    assert "ha_call_service" not in offered
    assert fake_ha.requests == []
    assert tts.received_text == [AMENDED_CONTINUATION_REFUSAL]


def _triage_clarifier(fake_envelope_client) -> brain_race.TierBrain:
    reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=("light.example_lamp", "light.example_desk_lamp"),
    )
    return brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=reply, delay_s=0.0),
        calls_tools=False,
    )


async def test_a_triage_tier_never_runs_on_an_amended_turn(
    fake_audio_source, fake_stt, fake_tts, fake_brain, fake_envelope_client
):
    """The model-driven path: a triage tier's `needs_clarification` reply
    would win the race and open a clarification window. On a restricted
    turn, no triage tier runs at all."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    pending_actions = FakePendingActionRepository()
    created = await _stored_create(pending_actions)
    tool_host = tpa._GoogleToolHost((tpa._home_account(),), google_client=FakeGoogle().client)
    ctx = _context(tool_host, pending_actions, tpa._RecordingConfirmationBrain("cancel", amended=True))
    triage = _triage_clarifier(fake_envelope_client)
    top = brain_race.TierBrain(
        index=1,
        model="top-model",
        brain=fake_brain(replies=[BrainReply(text="what time did you want instead?")], delay_s=0.05),
        envelope_client=None,
        calls_tools=True,
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )

    requested, tts = await _turn(
        incoming, "which light, the kitchen or the hall?", top.brain, tool_host, ctx, fakes=fakes, tiers=[triage, top]
    )

    assert triage.envelope_client.calls == []
    assert requested is None
    assert tts.received_text == ["what time did you want instead?"]


async def test_a_proposals_only_clarification_skips_triage_and_refuses_other_tools(
    fake_audio_source, fake_stt, fake_tts, fake_ha, fake_envelope_client
):
    """The clarification-continuation branch honors the restriction the
    request carries: the schema is narrowed, a named `ha_call_service` is
    refused at dispatch, and no triage tier runs."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = tpa._GoogleToolHost(
        _two_accounts_no_default(), ha=fake_ha, policy=Policy.from_config(None), google_client=FakeGoogle().client
    )
    ctx = _context(tool_host, FakePendingActionRepository(), tpa._RecordingConfirmationBrain("confirm"))
    triage = _triage_clarifier(fake_envelope_client)
    top_brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_UNLOCK])])
    top = brain_race.TierBrain(index=1, model="top-model", brain=top_brain, envelope_client=None, calls_tools=True)
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=2,
        original_transcript="no, make it 4 on some other account",
        question="which account -- home or work?",
        proposals_only=True,
    )

    _, tts = await _turn(incoming, "unlock the front door", top_brain, tool_host, ctx, fakes=fakes, tiers=[triage, top])

    assert triage.envelope_client.calls == []
    assert "ha_call_service" not in {entry["function"]["name"] for entry in top_brain.calls[0].tools}
    assert fake_ha.requests == []
    assert tts.received_text == [AMENDED_CONTINUATION_REFUSAL]


async def test_a_proposals_only_clarification_passes_the_restriction_to_its_confirmation(
    fake_audio_source, fake_stt, fake_tts
):
    """Transitive: the confirmation a restricted clarification's answer
    requests carries the restriction forward too."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    pending_actions = FakePendingActionRepository()
    tool_host = tpa._GoogleToolHost(_two_accounts_no_default(), google_client=FakeGoogle().client)
    ctx = _context(tool_host, pending_actions, tpa._RecordingConfirmationBrain("confirm"))
    brain = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_event",
                        arguments={"title": "Dentist", "start": "2026-10-02T16:00", "account": "home"},
                    )
                ]
            )
        ]
    )
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=2,
        original_transcript="no, make it 4 on some other account",
        question="which account -- home or work?",
        proposals_only=True,
    )

    requested, _ = await _turn(incoming, "home", brain, tool_host, ctx, fakes=fakes)

    assert requested is not None and requested.kind == "confirmation"
    assert requested.proposals_only is True
    assert requested.chain_depth == 3


async def test_an_unrestricted_clarification_stays_unrestricted(fake_audio_source, fake_stt, fake_tts):
    """The restriction never leaks the other way: a wake turn's own
    clarification still offers the full schema on its answer."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = tpa._GoogleToolHost(_two_accounts_no_default(), google_client=FakeGoogle().client)
    ctx = _context(tool_host, FakePendingActionRepository(), tpa._RecordingConfirmationBrain("confirm"))
    brain = RecordingFakeBrain(replies=[BrainReply(text="ok")])
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="turn on the lamp",
        question="which one -- the example lamp or the desk lamp?",
    )

    await _turn(incoming, "the desk lamp", brain, tool_host, ctx, fakes=fakes)

    assert {entry["function"]["name"] for entry in brain.calls[0].tools} == {e["function"]["name"] for e in _SCHEMA}


# --- R2-WR-04: the allowlist is the Google plugin's own offered names --------


class _RecordingToolHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name, arguments):
        from types import SimpleNamespace

        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])


def _restricted_clarification() -> FollowUpRequest:
    return FollowUpRequest(
        kind="clarification",
        chain_depth=2,
        original_transcript="no, make it 4",
        question="which account -- home or work?",
        proposals_only=True,
    )


async def test_another_plugins_same_named_proposal_tool_is_never_offered_or_dispatched(
    fake_audio_source, fake_stt, fake_tts
):
    """Two plugins publish `calendar_propose_event`, so the naming pre-pass
    prefixes both. Only the Google plugin's own offered name passes -- a
    suffix match would let the other plugin's tool through."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _RecordingToolHost()
    ctx = HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=FakePendingActionRepository(),
        brain=None,
        now=tpa._NOW,
        proposal_tool_names=frozenset({"google__calendar_propose_event", "google__calendar_propose_delete"}),
    )
    schema = [
        {"type": "function", "function": {"name": "google__calendar_propose_event"}},
        {"type": "function", "function": {"name": "other__calendar_propose_event"}},
        {"type": "function", "function": {"name": "x__calendar_propose_delete"}},
    ]
    brain = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(name="other__calendar_propose_event", arguments={}),
                    ToolCall(name="x__calendar_propose_delete", arguments={}),
                ]
            ),
            BrainReply(text="sorry"),
        ]
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=_restricted_clarification())

    await run_turn(
        source,
        fake_stt(events=[FinalTranscript(text="home")]),
        brain,
        fake_tts(chunks=[b"\x01"]),
        tool_host,
        tools_schema=schema,
        system_prompt="s",
        max_tool_rounds=3,
        timings=TurnTimings(),
        handoff_context=ctx,
    )

    assert {entry["function"]["name"] for entry in brain.calls[0].tools} == {"google__calendar_propose_event"}
    assert tool_host.calls == []


def test_naming_result_lists_only_the_owning_plugins_offered_names():
    from atlas_mcp.google_tools import CALENDAR_PROPOSAL_TOOL_NAMES

    from atlas.plugins.naming import PluginTool, PluginTools, rename_collisions

    result = rename_collisions(
        [
            PluginTools(
                slug="google",
                display_name="Google",
                tools=(PluginTool("calendar_propose_event", ""), PluginTool("calendar_propose_delete", "")),
            ),
            PluginTools(
                slug="other",
                display_name="Other",
                tools=(PluginTool("calendar_propose_event", ""), PluginTool("x__calendar_propose_delete", "")),
            ),
        ]
    )

    assert result.offered_names_owned_by("google", CALENDAR_PROPOSAL_TOOL_NAMES) == frozenset(
        {"google__calendar_propose_event", "calendar_propose_delete"}
    )
    assert result.offered_names_owned_by("missing", CALENDAR_PROPOSAL_TOOL_NAMES) == frozenset()


def test_handoff_context_proposal_tool_names_defaults_to_empty():
    """R3-IN-03: a construction site that forgets `proposal_tool_names`
    must fail closed, not fall back to the bare names matched exactly --
    the exact R2-WR-04 case this field exists to prevent. Only
    `build_handoff_context` (tested below) is trusted to fill in the real
    set; every other caller gets nothing offered on a restricted turn."""
    ctx = HandoffContext(source_name="camera", tool_host=None, pending_actions=None, brain=None)

    assert ctx.proposal_tool_names == frozenset()


def test_build_handoff_context_reads_the_google_plugins_offered_names():
    from types import SimpleNamespace as NS

    from atlas_mcp.google_tools import CALENDAR_PROPOSAL_TOOL_NAMES, GOOGLE_PLUGIN_MODULE

    from atlas.google.turn_context import build_handoff_context

    asked: list[tuple[str, frozenset[str]]] = []

    def offered(module, bare_names):
        asked.append((module, bare_names))
        return frozenset({"google__calendar_propose_event"})

    app = NS(state=NS(plugin_manager=NS(offered_tool_names_for_module=offered)))
    ctx = build_handoff_context(app, "camera")

    assert ctx.proposal_tool_names == frozenset({"google__calendar_propose_event"})
    assert asked[0][0] == GOOGLE_PLUGIN_MODULE
    assert CALENDAR_PROPOSAL_TOOL_NAMES <= asked[0][1]
    # No plugin manager: nothing is allowed on a restricted turn.
    assert build_handoff_context(NS(state=NS()), "camera").proposal_tool_names == frozenset()


# --- R2-WR-05: an amended delete can look up the event it re-proposes -------


async def test_an_amended_delete_can_list_events_and_propose_the_right_one(fake_audio_source, fake_stt, fake_tts):
    """"no, the one on tuesday" needs an event id the continuation does not
    carry. The read-only `calendar_list_events` is allowed on a restricted
    turn, so the model can find it and re-propose -- a NEW pending action
    with its own readback (D-08)."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    fake_google = FakeGoogle()
    for event_id, day in (("evt-mon", "05"), ("evt-tue", "06")):
        fake_google.add_event(
            "at-work",
            "cal-work",
            {
                "id": event_id,
                "summary": "Standup",
                "start": {"dateTime": f"2026-10-{day}T09:00:00-04:00"},
                "end": {"dateTime": f"2026-10-{day}T09:15:00-04:00"},
            },
        )
    tool_host = tpa._GoogleToolHost((tpa._work_account(),), google_client=fake_google.client)
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="camera",
        action="calendar_delete",
        tool_name="calendar_delete_event",
        arguments={"account": "work", "calendar_id": "cal-work", "event_id": "evt-mon"},
        readback="delete Standup from the work calendar, monday october 5th at 9 am?",
        created_at=tpa._NOW,
        expires_at=tpa._NOW + timedelta(seconds=60),
    )
    ctx = _context(tool_host, pending_actions, tpa._RecordingConfirmationBrain("cancel", amended=True))
    schema = [
        {"type": "function", "function": {"name": "calendar_list_events"}},
        {"type": "function", "function": {"name": "calendar_propose_delete"}},
        {"type": "function", "function": {"name": "gmail_search"}},
    ]
    brain = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_list_events",
                        arguments={"start": "2026-10-06T00:00:00-04:00", "end": "2026-10-07T00:00:00-04:00"},
                    )
                ]
            ),
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_propose_delete",
                        arguments={"account": "work", "calendar_id": "cal-work", "event_id": "evt-tue"},
                    )
                ]
            ),
        ]
    )
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="delete standup",
        question=created.readback,
        pending_action_id=created.id,
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    tts = fake_tts(chunks=[b"\x01"])

    await run_turn(
        source,
        fake_stt(events=[FinalTranscript(text="no, the one on tuesday")]),
        brain,
        tts,
        tool_host,
        tools_schema=schema,
        system_prompt="s",
        max_tool_rounds=3,
        timings=TurnTimings(),
        handoff_context=ctx,
    )

    assert {entry["function"]["name"] for entry in brain.calls[0].tools} == {
        "calendar_list_events",
        "calendar_propose_delete",
    }
    assert (await pending_actions.get(created.id)).status == "superseded"
    new_rows = [row for row in pending_actions._rows.values() if row.id != created.id]
    assert len(new_rows) == 1 and new_rows[0].arguments["event_id"] == "evt-tue"
    requested = source.follow_up.requested
    assert requested is not None and requested.kind == "confirmation" and requested.proposals_only is True
    assert tts.received_text == ["delete Standup from the work calendar, tuesday october 6th at 9 am?"]
    assert fake_google.deleted_event_ids == []
