"""A follow-up turn (no wake word) goes through the same speaker gate a
wake turn does (D-11, 11-04-PLAN.md Task 2): an unenrolled voice's
confirmation or clarification reply never reaches
`handle_confirmation_reply`/the clarification branch at all, speaks
nothing, and leaves any stored pending action untouched; an enrolled
voice's follow-up reaches exactly the same code path this project already
has.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone

from atlas.config import SessionConfig
from atlas.session.recorder import SessionRecorder
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.tracker import SPEAKER_DECISION_TIMEOUT_S, SpeakerMeasurement
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext
from atlas.timing import TurnTimings
from atlas.turn import controller as controller_module
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import FollowUpChannel, FollowUpRequest
from atlas.turn.pending_action import CANCELLED_REPLY, ConfirmationOutcome

from tests.pending_action_fakes import FakePendingActionRepository

_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


class _StubSpan:
    def __init__(self, measurement: SpeakerMeasurement) -> None:
        self._measurement = measurement
        self.decide_calls: "list[dict]" = []

    async def decide(self, *, end_of_speech_at, references, timeout_s=SPEAKER_DECISION_TIMEOUT_S):
        self.decide_calls.append({"end_of_speech_at": end_of_speech_at, "references": references})
        return self._measurement


class _StubTracker:
    def __init__(self, span: "_StubSpan") -> None:
        self._span = span
        self.open_turn_calls: "list[float | None]" = []

    def open_turn(self, *, start_after=None):
        self.open_turn_calls.append(start_after)
        return self._span


def _enforce_context(match: "MatchResult | None") -> SpeakerIdTurnContext:
    references = ReferenceSet()
    references.upsert_speaker(1, "Alice", [(1.0, 0.0)])
    measurement = SpeakerMeasurement(
        match=match, speech_ms=500.0, window_count=1, ready_at=10.0, speaker_id_ms=30.0, detail=None,
    )
    span = _StubSpan(measurement)
    tracker = _StubTracker(span)
    return SpeakerIdTurnContext(
        tracker=tracker, references=references, mode="enforce", threshold=0.5, model_id="campplus", worker=object(),
    )


def _identified_match() -> MatchResult:
    return MatchResult(best_speaker_id=1, best_name="Alice", best_score=0.9, second_score=0.1, margin=0.8, scores={1: 0.9})


def _unidentified_match() -> MatchResult:
    return MatchResult(best_speaker_id=1, best_name="Alice", best_score=0.1, second_score=0.05, margin=0.05, scores={1: 0.1})


def _read_events(recorder: SessionRecorder) -> "list[dict]":
    lines = (recorder.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _find_speaker_result(events: "list[dict]") -> dict:
    for event in events:
        if event.get("type") == "speaker.result":
            return event
    raise AssertionError(f"no speaker.result event found among {[e.get('type') for e in events]}")


class _ConfirmationSpy:
    """Replaces `atlas.turn.controller.handle_confirmation_reply` --
    records every call and returns the scripted outcome, so a test can
    prove the speaker gate blocks a follow-up BEFORE this function is ever
    reached, with no real pending-action/Google machinery needed."""

    def __init__(self, outcome: ConfirmationOutcome) -> None:
        self.calls: "list[tuple]" = []
        self._outcome = outcome

    async def __call__(self, ctx, incoming, transcript, *, timeout_s):
        self.calls.append((ctx, incoming, transcript, timeout_s))
        return self._outcome


async def test_an_unenrolled_confirmation_reply_never_reaches_handle_confirmation_reply(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    from atlas.providers.base import BrainReply, FinalTranscript

    pending_actions = FakePendingActionRepository()
    created = await pending_actions.create(
        source="edge",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={"title": "Dentist"},
        readback="add dentist to the calendar friday at 3?",
        created_at=_NOW,
        expires_at=_NOW + timedelta(seconds=60),
    )

    spy = _ConfirmationSpy(ConfirmationOutcome(reply_text="confirmed", turn_outcome="confirmed"))
    monkeypatch.setattr(controller_module, "handle_confirmation_reply", spy)

    source = fake_audio_source(frames=[b"\x00\x01"])
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="add dentist to the home calendar, friday at 3 pm, for an hour?",
        pending_action_id=created.id,
    )
    source.follow_up = FollowUpChannel(incoming=incoming, window_opens_at=0.0, window_s=10.0)

    stt = fake_stt(events=[FinalTranscript(text="yeah")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    recorder = SessionRecorder(SessionConfig(dir=str(tmp_path)), timings)

    context = _enforce_context(_unidentified_match())

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
        session_recorder=recorder,
        speaker_id=context,
    )

    assert spy.calls == [], "an unenrolled follow-up must never reach handle_confirmation_reply"
    assert tts.received_text == [], "an unenrolled follow-up must speak nothing"
    assert source.sent_audio == []
    assert timings.turn_outcome == "unknown_speaker"
    assert source.follow_up.requested is None, "a blocked follow-up must never request a new one"

    row = await pending_actions.get(created.id)
    assert row.status == "awaiting", "a blocked follow-up must leave the stored pending action untouched"

    result = _find_speaker_result(_read_events(recorder))
    assert result["status"] == "unknown"
    assert result["blocked"] is True
    assert result["reason"] == "unknown_speaker"


async def test_an_enrolled_confirmation_reply_reaches_handle_confirmation_reply_as_before(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    from atlas.providers.base import BrainReply, FinalTranscript

    spy = _ConfirmationSpy(ConfirmationOutcome(reply_text="confirmed", turn_outcome="confirmed"))
    monkeypatch.setattr(controller_module, "handle_confirmation_reply", spy)

    source = fake_audio_source(frames=[b"\x00\x01"])
    incoming = FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="add dentist to the home calendar, friday at 3 pm, for an hour?",
        pending_action_id=1,
    )
    source.follow_up = FollowUpChannel(incoming=incoming, window_opens_at=0.0, window_s=10.0)

    stt = fake_stt(events=[FinalTranscript(text="yeah")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    recorder = SessionRecorder(SessionConfig(dir=str(tmp_path)), timings)

    context = _enforce_context(_identified_match())

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
        session_recorder=recorder,
        speaker_id=context,
    )

    assert len(spy.calls) == 1
    assert tts.received_text == ["confirmed"]
    assert timings.turn_outcome == "confirmed"

    result = _find_speaker_result(_read_events(recorder))
    assert result["status"] == "identified"
    assert result["speaker_id"] == 1
    assert result["speaker_name"] == "Alice"
    assert result["blocked"] is False


async def test_an_unenrolled_clarification_reply_ends_silently_never_with_cancelled_reply(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    from atlas.providers.base import BrainReply, FinalTranscript

    source = fake_audio_source(frames=[b"\x00\x01"])
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="turn off the light",
        question="which one -- kitchen or hallway?",
    )
    source.follow_up = FollowUpChannel(incoming=incoming, window_opens_at=0.0, window_s=10.0)

    stt = fake_stt(events=[FinalTranscript(text="kitchen")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    recorder = SessionRecorder(SessionConfig(dir=str(tmp_path)), timings)

    context = _enforce_context(_unidentified_match())

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
        session_recorder=recorder,
        speaker_id=context,
    )

    assert tts.received_text == []
    assert CANCELLED_REPLY not in tts.received_text
    assert timings.turn_outcome == "unknown_speaker"
    assert source.follow_up.requested is None

    result = _find_speaker_result(_read_events(recorder))
    assert result["blocked"] is True
    assert result["reason"] == "unknown_speaker"
