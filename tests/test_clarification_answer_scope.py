"""R3-IN-05 regression (D-24 of phase 9): the answer to a clarifying
question, heard in the no-wake-word window, reaches only what the
clarifying turn itself asked about.

Before this fix, a clarification that an ordinary wake turn asked gave
the next no-wake-word turn the full tool set. "The desk one, and unlock
the front door" then reached `ha_call_service` for the lock with no wake
word. Now the request carries an `AnswerScope`:

- A tool's own `needs_clarification` handoff scopes the answer to that
  one tool.
- A triage tier's clarification between entity ids scopes the answer to
  `ha_call_service`, for those entity ids only.
- Any other triage clarification (plugins, scheduled runs) scopes the
  answer to no tools at all.

The scope belongs to the chain: every follow-up the answer turn requests
carries it, and a later clarification can only narrow it.

Every account, calendar, and entity name below is invented.
"""

from __future__ import annotations

from datetime import timedelta

from atlas_mcp.safety import Policy

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.providers.tier_reply import FillerPhrase, TierReply
from atlas.timing import TurnTimings
from atlas.turn import brain_race
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import AnswerScope, FollowUpChannel, FollowUpRequest
from atlas.turn.handoff import AMENDED_CONTINUATION_REFUSAL, HandoffContext

import test_pending_action as tpa
from brain_fakes import RecordingFakeBrain
from google_fakes import FakeGoogle
from pending_action_fakes import FakePendingActionRepository

_SCHEMA = [
    {"type": "function", "function": {"name": "calendar_propose_event"}},
    {"type": "function", "function": {"name": "ha_call_service"}},
    {"type": "function", "function": {"name": "gmail_draft_reply"}},
]

_LAMPS = ("light.example_lamp", "light.example_desk_lamp")

_DESK_LAMP_ON = ToolCall(
    name="ha_call_service",
    arguments={"domain": "light", "service": "turn_on", "entity_id": "light.example_desk_lamp"},
)
_UNLOCK = ToolCall(
    name="ha_call_service",
    arguments={"domain": "lock", "service": "unlock", "entity_id": "lock.example_front_door"},
)


def _triage(candidates, fake_envelope_client) -> brain_race.TierBrain:
    reply = TierReply(
        answer="",
        confident=False,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        needs_clarification=True,
        candidates=candidates,
    )
    return brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=reply, delay_s=0.0),
        calls_tools=False,
    )


def _context(tool_host) -> HandoffContext:
    return HandoffContext(
        source_name="camera",
        tool_host=tool_host,
        pending_actions=FakePendingActionRepository(),
        brain=tpa._RecordingConfirmationBrain("confirm"),
        now=tpa._NOW + timedelta(seconds=5),
    )


def _ha_host(fake_ha):
    return tpa._GoogleToolHost(
        (tpa._account("home"), tpa._account("work")),
        ha=fake_ha,
        policy=Policy.from_config(None),
        google_client=FakeGoogle().client,
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


def _service_posts(fake_ha) -> list[str]:
    return [request.url.path for request in fake_ha.requests if request.method == "POST"]


# --- (a) an HA clarification: only the named light -------------------------


async def test_an_ha_clarification_answer_reaches_only_the_light_it_asked_about(
    fake_audio_source, fake_stt, fake_tts, fake_ha, fake_envelope_client
):
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _ha_host(fake_ha)
    ctx = _context(tool_host)

    # Turn 1 (wake word): "which light?"
    requested, _ = await _turn(
        None, "turn on the lamp", None, tool_host, ctx, fakes=fakes, tiers=[_triage(_LAMPS, fake_envelope_client)]
    )
    assert requested is not None and requested.kind == "clarification"
    assert requested.answer_scope == AnswerScope(
        tool_names=frozenset({"ha_call_service"}), entity_ids=frozenset(_LAMPS)
    )

    # Turn 2 (no wake word): the answer tries to add an unlock.
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_DESK_LAMP_ON, _UNLOCK])])
    _, tts = await _turn(requested, "the desk one, and unlock the front door", brain, tool_host, ctx, fakes=fakes)

    assert {entry["function"]["name"] for entry in brain.calls[0].tools} == {"ha_call_service"}
    assert [name for name, _ in tool_host.calls] == ["ha_call_service"]
    assert tool_host.calls[0][1]["entity_id"] == "light.example_desk_lamp"
    assert _service_posts(fake_ha) == ["/api/services/light/turn_on"]
    assert AMENDED_CONTINUATION_REFUSAL in tts.received_text[-1]


async def test_an_ha_clarification_answer_cannot_widen_its_target_by_area(
    fake_audio_source, fake_stt, fake_tts, fake_ha
):
    """An area, device, or label target expands to entities the question
    never named -- refused, even with the right tool."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _ha_host(fake_ha)
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="turn on the lamp",
        question="which one -- the example lamp or the desk lamp?",
        answer_scope=AnswerScope(tool_names=frozenset({"ha_call_service"}), entity_ids=frozenset(_LAMPS)),
    )
    by_area = ToolCall(
        name="ha_call_service",
        arguments={"domain": "light", "service": "turn_on", "entity_id": "light.example_lamp", "area_id": "example"},
    )
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[by_area])])

    _, tts = await _turn(incoming, "all of them", brain, tool_host, _context(tool_host), fakes=fakes)

    assert tool_host.calls == []
    assert fake_ha.requests == []
    assert AMENDED_CONTINUATION_REFUSAL in tts.received_text[-1]


# --- a Google tool's own clarification: only that tool ----------------------


async def test_a_tool_clarification_answer_is_offered_only_the_tool_that_asked(
    fake_audio_source, fake_stt, fake_tts, fake_ha
):
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _ha_host(fake_ha)
    ctx = _context(tool_host)

    # Turn 1 (wake word): two accounts and no default -- "which account?"
    brain1 = RecordingFakeBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(name="calendar_propose_event", arguments={"title": "Dentist", "start": "2026-10-02T15:00"})
                ]
            )
        ]
    )
    requested, _ = await _turn(None, "add dentist friday at 3", brain1, tool_host, ctx, fakes=fakes)
    assert requested is not None and requested.kind == "clarification"
    assert requested.answer_scope == AnswerScope(tool_names=frozenset({"calendar_propose_event"}))

    # Turn 2 (no wake word): the answer names an unlock.
    brain2 = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_UNLOCK])])
    _, tts = await _turn(requested, "home, and unlock the front door", brain2, tool_host, ctx, fakes=fakes)

    assert {entry["function"]["name"] for entry in brain2.calls[0].tools} == {"calendar_propose_event"}
    assert fake_ha.requests == []
    assert tts.received_text == [AMENDED_CONTINUATION_REFUSAL]


# --- (b) a chained second clarification stays restricted --------------------


async def test_a_chained_triage_clarification_only_narrows_the_scope(
    fake_audio_source, fake_stt, fake_tts, fake_ha, fake_envelope_client, fake_brain
):
    """The second question names the lock as a candidate. The chain's
    scope never grows: the lock was not in the first question, so the
    third turn still cannot reach it."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _ha_host(fake_ha)
    ctx = _context(tool_host)
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="turn on the lamp",
        question="which one -- the example lamp or the desk lamp?",
        answer_scope=AnswerScope(tool_names=frozenset({"ha_call_service"}), entity_ids=frozenset(_LAMPS)),
    )
    top = brain_race.TierBrain(
        index=1,
        model="top-model",
        brain=fake_brain(replies=[BrainReply(text="unused")], delay_s=0.05),
        envelope_client=None,
        calls_tools=True,
    )
    second_question = _triage(("light.example_desk_lamp", "lock.example_front_door"), fake_envelope_client)

    requested, _ = await _turn(
        incoming, "the one by the door", top.brain, tool_host, ctx, fakes=fakes, tiers=[second_question, top]
    )

    assert requested is not None and requested.kind == "clarification" and requested.chain_depth == 2
    assert requested.answer_scope == AnswerScope(
        tool_names=frozenset({"ha_call_service"}), entity_ids=frozenset({"light.example_desk_lamp"})
    )

    # Turn 3 (no wake word): "the lock" -- still refused.
    brain3 = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_UNLOCK])])
    _, tts = await _turn(requested, "the lock", brain3, tool_host, ctx, fakes=fakes)

    assert {entry["function"]["name"] for entry in brain3.calls[0].tools} == {"ha_call_service"}
    assert fake_ha.requests == []
    assert tts.received_text == [AMENDED_CONTINUATION_REFUSAL]


async def test_a_chained_tool_clarification_keeps_the_scope_and_passes_it_to_the_confirmation(
    fake_audio_source, fake_stt, fake_tts, fake_ha
):
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _ha_host(fake_ha)
    ctx = _context(tool_host)
    scope = AnswerScope(tool_names=frozenset({"calendar_propose_event"}))
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="add dentist friday at 3",
        question="i'm not sure which one you mean -- home, work?",
        answer_scope=scope,
    )
    propose = ToolCall(name="calendar_propose_event", arguments={"title": "Dentist", "start": "2026-10-02T15:00"})

    # Turn 2: the answer still names no account -- asked again.
    brain2 = RecordingFakeBrain(replies=[BrainReply(tool_calls=[propose])])
    requested, _ = await _turn(incoming, "the usual one", brain2, tool_host, ctx, fakes=fakes)
    assert requested is not None and requested.kind == "clarification" and requested.chain_depth == 2
    assert requested.answer_scope == scope

    # Turn 3: answered -- the confirmation carries the same scope.
    named = ToolCall(name="calendar_propose_event", arguments={**propose.arguments, "account": "home"})
    confirmation, _ = await _turn(
        requested, "home", RecordingFakeBrain(replies=[BrainReply(tool_calls=[named])]), tool_host, ctx, fakes=fakes
    )
    assert confirmation is not None and confirmation.kind == "confirmation"
    assert confirmation.answer_scope == scope
    assert confirmation.proposals_only is False


# --- (c) the happy path still completes -------------------------------------


async def test_an_ha_clarification_answer_still_turns_on_the_named_light(
    fake_audio_source, fake_stt, fake_tts, fake_ha
):
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _ha_host(fake_ha)
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="turn on the lamp",
        question="which one -- the example lamp or the desk lamp?",
        answer_scope=AnswerScope(tool_names=frozenset({"ha_call_service"}), entity_ids=frozenset(_LAMPS)),
    )
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_DESK_LAMP_ON])])

    requested, tts = await _turn(incoming, "the desk one", brain, tool_host, _context(tool_host), fakes=fakes)

    assert _service_posts(fake_ha) == ["/api/services/light/turn_on"]
    assert tts.received_text == ["done"]
    assert requested is None


async def test_a_plugin_clarification_answer_is_offered_no_tools(
    fake_audio_source, fake_stt, fake_tts, fake_ha, fake_envelope_client
):
    """A triage clarification between plugins or scheduled runs names no
    entity -- its answer gets no tools, so an action needs the wake word."""
    fakes = (fake_audio_source, fake_stt, fake_tts)
    tool_host = _ha_host(fake_ha)
    ctx = _context(tool_host)

    requested, _ = await _turn(
        None,
        "what's the forecast",
        None,
        tool_host,
        ctx,
        fakes=fakes,
        tiers=[_triage(("Weather", "Garden Sensors"), fake_envelope_client)],
    )
    assert requested is not None and requested.answer_scope == AnswerScope(tool_names=frozenset())

    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_UNLOCK])])
    _, tts = await _turn(requested, "weather, and unlock the front door", brain, tool_host, ctx, fakes=fakes)

    assert brain.calls[0].tools == []
    assert fake_ha.requests == []
    assert tts.received_text == [AMENDED_CONTINUATION_REFUSAL]
