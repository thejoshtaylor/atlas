"""D-14 (11-06-PLAN.md Task 2): the brain gets a member's name as one
fixed, explicitly untrusted system message on an identified wake turn --
never on a turn the gate did not identify, and never on any follow-up
(confirmation, amendment, or clarification answer), however identified the
speaker is.
"""

from __future__ import annotations

import test_turn_speaker_gate as ttsg
from atlas.providers.base import BrainReply, FinalTranscript
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.tracker import SpeakerMeasurement
from atlas.speaker_id.turn_gate import SPEAKER_HINT_TEMPLATE, SpeakerIdTurnContext, compose_speaker_hint
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import FollowUpChannel, FollowUpRequest


class _RecordingBrain:
    """Records every `chat()` call's own `messages`, as an independent
    snapshot (so later mutation of the caller's own list cannot retroactively
    change what a test asserts), then returns the next scripted reply."""

    def __init__(self, replies) -> None:
        self._replies = list(replies)
        self.calls: "list[list[dict]]" = []

    async def chat(self, messages, tools=None) -> BrainReply:
        self.calls.append([dict(message) for message in messages])
        reply = self._replies[len(self.calls) - 1]
        return reply


def _identified_context(name: str = "Member A", *, threshold: float = 0.5) -> SpeakerIdTurnContext:
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
        tracker=tracker, references=references, mode="enforce", threshold=threshold, model_id="campplus", worker=object(),
    )


def _below_threshold_context() -> SpeakerIdTurnContext:
    references = ReferenceSet()
    references.upsert_speaker(1, "Member A", [(1.0, 0.0)])
    match = MatchResult(
        best_speaker_id=1, best_name="Member A", best_score=0.2, second_score=0.1, margin=0.1, scores={1: 0.2},
    )
    measurement = SpeakerMeasurement(
        match=match, speech_ms=750.0, window_count=1, ready_at=10.0, speaker_id_ms=20.0, detail=None,
    )
    span = ttsg._StubSpan(measurement)
    tracker = ttsg._StubTracker(span)
    return SpeakerIdTurnContext(
        tracker=tracker, references=references, mode="record", threshold=0.5, model_id="campplus", worker=object(),
    )


def _off_context() -> SpeakerIdTurnContext:
    return SpeakerIdTurnContext(tracker=None, references=None, mode="off", threshold=0.5, model_id=None, worker=None)


def _hint_messages(messages: "list[dict]", name: str) -> "list[dict]":
    return [m for m in messages if m.get("content") == compose_speaker_hint(name)]


# ---------------------------------------------------------------------------
# compose_speaker_hint / SPEAKER_HINT_TEMPLATE
# ---------------------------------------------------------------------------


def test_compose_speaker_hint_holds_the_name_once_and_the_required_phrases():
    hint = compose_speaker_hint("Member A")
    assert hint.count("Member A") == 1
    assert "not proof" in hint
    assert "never grants permission" in hint


def test_compose_speaker_hint_uses_the_fixed_template():
    assert compose_speaker_hint("Member A") == SPEAKER_HINT_TEMPLATE.format(name="Member A")


# ---------------------------------------------------------------------------
# An identified wake turn gets exactly one hint, after state, before user
# ---------------------------------------------------------------------------


async def test_identified_wake_turn_sends_exactly_one_hint_message(fake_audio_source, fake_stt, fake_tts):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = _RecordingBrain([BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    context = _identified_context()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=context,
    )

    assert len(brain.calls) == 1
    messages = brain.calls[0]
    hint_messages = _hint_messages(messages, "Member A")
    assert len(hint_messages) == 1
    assert hint_messages[0]["role"] == "system"

    # No state fetch was given -- the hint lands right after system_prompt
    # and right before the user message.
    assert messages[0] == {"role": "system", "content": "you control a home"}
    assert messages[1]["content"] == compose_speaker_hint("Member A")
    assert messages[2]["role"] == "user"


async def test_identified_wake_turn_hint_lands_after_the_state_message(fake_audio_source, fake_stt, fake_tts):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = _RecordingBrain([BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    context = _identified_context()

    async def _state_fetch():
        return []

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=context,
        state_fetch=_state_fetch,
    )

    messages = brain.calls[0]
    assert messages[0]["content"] == "you control a home"
    assert messages[1]["role"] == "system"
    assert messages[1]["content"] != compose_speaker_hint("Member A"), "the state message, not the hint"
    assert messages[2]["content"] == compose_speaker_hint("Member A")
    assert messages[3]["role"] == "user"


# ---------------------------------------------------------------------------
# No hint when not identified, or with speaker id off/absent entirely
# ---------------------------------------------------------------------------


async def test_below_threshold_wake_turn_sends_no_hint(fake_audio_source, fake_stt, fake_tts):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = _RecordingBrain([BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=_below_threshold_context(),
    )

    messages = brain.calls[0]
    assert _hint_messages(messages, "Member A") == []


async def test_speaker_id_off_wake_turn_sends_no_hint(fake_audio_source, fake_stt, fake_tts):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = _RecordingBrain([BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=_off_context(),
    )

    messages = brain.calls[0]
    assert not any(m["role"] == "system" and "Untrusted hint" in m.get("content", "") for m in messages)


async def test_camera_or_browser_turn_with_no_speaker_context_sends_no_hint(fake_audio_source, fake_stt, fake_tts):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = _RecordingBrain([BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=None,
    )

    messages = brain.calls[0]
    assert not any(m["role"] == "system" and "Untrusted hint" in m.get("content", "") for m in messages)


# ---------------------------------------------------------------------------
# No hint on any follow-up, even when identified
# ---------------------------------------------------------------------------


async def test_a_clarification_answer_from_an_identified_member_sends_no_hint(fake_audio_source, fake_stt, fake_tts):
    source = fake_audio_source(frames=[b"\x00\x01"])
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="turn off the light",
        question="which one -- kitchen or hallway?",
    )
    # `window_opens_at`/`window_s` stay `None` (the default) -- given
    # instead, `_drain_to_final_transcript`'s `onset_deadline` would compare
    # a small relative number against a real `time.monotonic()` reading and
    # give up before the scripted transcript is ever read, which is not
    # what this test is about.
    source.follow_up = FollowUpChannel(incoming=incoming)

    stt = fake_stt(events=[FinalTranscript(text="the kitchen one")])
    brain = _RecordingBrain([BrainReply(text="done")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        speaker_id=_identified_context(),
    )

    assert len(brain.calls) == 1
    messages = brain.calls[0]
    assert _hint_messages(messages, "Member A") == []
