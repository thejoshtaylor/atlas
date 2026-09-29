"""D-15 (11-06-PLAN.md Task 2): a speaker label is untrusted, the same as
transcribed text, and it never reaches anything that authorizes. Two
layers of proof:

- Structural: no parameter of `allow_call`, `handle_confirmation_reply`,
  `run_confirmation_round`, or `dispatch_handoff` has "speaker" in its
  name; no field of `Policy`, `FollowUpRequest`, `AnswerScope`, or
  `PendingProposal` has "speaker" in its name. `mcp/atlas_mcp/safety.py`
  already uses the word "speaker" for the person who HEARS a reply
  (unrelated to this project's own `speaker_id` subsystem) -- these checks
  read signatures and fields, never the file's own text, so that unrelated
  usage can never make this suite lie.
- Behavioral: an identified wake turn whose brain calls `ha_call_service`
  sends tool arguments carrying neither the member's name nor the word
  "speaker"; a confirmation follow-up from an identified member sends
  confirmation-round messages carrying neither the name nor the hint text.
"""

from __future__ import annotations

import dataclasses
import inspect
import json
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import test_turn_speaker_gate as ttsg
from atlas_mcp.safety import Policy, allow_call
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.speaker_id.matching import MatchResult
from atlas.speaker_id.tracker import SpeakerMeasurement
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext, compose_speaker_hint
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import AnswerScope, FollowUpChannel, FollowUpRequest
from atlas.turn.handoff import HandoffContext, dispatch_handoff
from atlas.turn.pending_action import PendingProposal, handle_confirmation_reply, run_confirmation_round

from tests.pending_action_fakes import FakePendingActionRepository

_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


def _no_name_carries_speaker(names) -> bool:
    return any("speaker" in name.lower() for name in names)


# ---------------------------------------------------------------------------
# Structural: signatures and fields, never the file's own text
# ---------------------------------------------------------------------------


def test_allow_call_signature_carries_no_speaker_parameter():
    params = inspect.signature(allow_call).parameters
    assert not _no_name_carries_speaker(params.keys())


def test_handle_confirmation_reply_signature_carries_no_speaker_parameter():
    params = inspect.signature(handle_confirmation_reply).parameters
    assert not _no_name_carries_speaker(params.keys())


def test_run_confirmation_round_signature_carries_no_speaker_parameter():
    params = inspect.signature(run_confirmation_round).parameters
    assert not _no_name_carries_speaker(params.keys())


def test_dispatch_handoff_signature_carries_no_speaker_parameter():
    params = inspect.signature(dispatch_handoff).parameters
    assert not _no_name_carries_speaker(params.keys())


def test_policy_fields_carry_no_speaker_field():
    field_names = [f.name for f in dataclasses.fields(Policy)]
    assert not _no_name_carries_speaker(field_names)


def test_follow_up_request_fields_carry_no_speaker_field():
    field_names = [f.name for f in dataclasses.fields(FollowUpRequest)]
    assert not _no_name_carries_speaker(field_names)


def test_answer_scope_fields_carry_no_speaker_field():
    field_names = [f.name for f in dataclasses.fields(AnswerScope)]
    assert not _no_name_carries_speaker(field_names)


def test_pending_proposal_fields_carry_no_speaker_field():
    # A pydantic BaseModel, not a dataclass -- its own field registry.
    field_names = list(PendingProposal.model_fields.keys())
    assert not _no_name_carries_speaker(field_names)


# ---------------------------------------------------------------------------
# Behavioral: an identified turn's own tool arguments
# ---------------------------------------------------------------------------


class _RecordingBrain:
    def __init__(self, replies) -> None:
        self._replies = list(replies)
        self.calls: "list[list[dict]]" = []

    async def chat(self, messages, tools=None) -> BrainReply:
        self.calls.append([dict(message) for message in messages])
        reply = self._replies[len(self.calls) - 1]
        return reply


class _RecordingToolHost:
    """Records every `call_tool`'s exact `name`/`arguments` and answers
    with a bare `{"changed": [...]}` Home Assistant reply -- enough for
    `_run_tool_rounds`'s own done shortcut to end the round with no second
    `brain.chat` call needed."""

    def __init__(self) -> None:
        self.calls: "list[tuple[str, dict]]" = []

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append((name, dict(arguments)))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps({"changed": []}))])


def _identified_context(name: str = "Member A") -> SpeakerIdTurnContext:
    references = ttsg._reference_set()
    match = MatchResult(
        best_speaker_id=1, best_name=name, best_score=0.9, second_score=0.1, margin=0.8, scores={1: 0.9, 2: 0.1},
    )
    measurement = SpeakerMeasurement(
        match=match, speech_ms=1000.0, window_count=2, ready_at=100.0, speaker_id_ms=42.0, detail=None,
    )
    span = ttsg._StubSpan(measurement)
    tracker = ttsg._StubTracker(span)
    return SpeakerIdTurnContext(
        tracker=tracker, references=references, mode="enforce", threshold=0.5, model_id="campplus", worker=object(),
    )


async def test_an_identified_turns_tool_arguments_never_carry_the_name_or_the_word_speaker(
    fake_audio_source, fake_stt, fake_tts
):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = _RecordingBrain(
        [
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"},
                    )
                ]
            ),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    tool_host = _RecordingToolHost()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=_identified_context(),
    )

    assert len(tool_host.calls) == 1
    name, arguments = tool_host.calls[0]
    assert name == "ha_call_service"
    serialized = json.dumps(arguments)
    assert "Member A" not in serialized
    assert "speaker" not in serialized.lower()
    # Only one brain round -- the done shortcut, proving the hint (present
    # in round 1's own messages) never became a second round's content either.
    assert len(brain.calls) == 1
    assert any(m.get("content") == compose_speaker_hint("Member A") for m in brain.calls[0])


# ---------------------------------------------------------------------------
# Behavioral: a confirmation follow-up's own restricted round
# ---------------------------------------------------------------------------


async def test_a_confirmation_rounds_messages_never_carry_the_name_or_the_hint_text(
    fake_audio_source, fake_stt, fake_tts
):
    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="edge",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={"title": "Dentist"},
        readback="add dentist to the home calendar, friday at 3 pm, for an hour?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )

    confirmation_brain = _RecordingBrain([BrainReply(tool_calls=[ToolCall(name="confirm", arguments={})])])
    tool_host = _RecordingToolHost()
    handoff_context = HandoffContext(
        source_name="edge", tool_host=tool_host, pending_actions=pending_actions, brain=confirmation_brain, now=_NOW,
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question=created.readback,
        pending_action_id=created.id,
    )
    # `window_opens_at`/`window_s` stay `None` (the default) -- given
    # instead, `_drain_to_final_transcript`'s `onset_deadline` would compare
    # a small relative number against a real `time.monotonic()` reading and
    # give up before the scripted "yes" is ever read.
    source.follow_up = FollowUpChannel(incoming=incoming)

    stt = fake_stt(events=[FinalTranscript(text="yes")])
    # The confirmation branch never reaches the ordinary tier race -- this
    # brain is never called, proving the assertion below is about the
    # RESTRICTED round's own brain, not a stray ordinary-turn call.
    turn_brain = _RecordingBrain([])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        turn_brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you manage a calendar",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=_identified_context(),
        handoff_context=handoff_context,
    )

    assert timings.turn_outcome == "confirmed"
    assert turn_brain.calls == []
    assert len(confirmation_brain.calls) == 1
    for message in confirmation_brain.calls[0]:
        serialized = json.dumps(message)
        assert "Member A" not in serialized
        assert "Untrusted hint" not in serialized
