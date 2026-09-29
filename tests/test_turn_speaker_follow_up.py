"""A follow-up turn (no wake word) goes through the same speaker gate a
wake turn does (D-11, 11-04-PLAN.md Task 2): an unenrolled voice's
confirmation or clarification reply never reaches
`handle_confirmation_reply`/the clarification branch at all, speaks
nothing, and leaves any stored pending action untouched; an enrolled
voice's follow-up reaches exactly the same code path this project already
has.
"""

from __future__ import annotations

import asyncio
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
from atlas.turn.turn_context import TurnContext, follow_up_speaker_mismatch

from tests.pending_action_fakes import FakePendingActionRepository

_NOW = datetime(2026, 9, 28, 12, 0, tzinfo=timezone.utc)


class _StubSpan:
    def __init__(self, measurement: SpeakerMeasurement) -> None:
        self._measurement = measurement
        self.decide_calls: "list[dict]" = []
        # 11-06-PLAN.md Task 1 (D-12): `run_turn` reads `speaker_span.
        # split_event` unconditionally off every span it opens.
        self.split_event = asyncio.Event()

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


def _enforce_context(match: "MatchResult | None", mode: str = "enforce") -> SpeakerIdTurnContext:
    references = ReferenceSet()
    references.upsert_speaker(1, "Alice", [(1.0, 0.0)])
    measurement = SpeakerMeasurement(
        match=match, speech_ms=500.0, window_count=1, ready_at=10.0, speaker_id_ms=30.0, detail=None,
    )
    span = _StubSpan(measurement)
    tracker = _StubTracker(span)
    return SpeakerIdTurnContext(
        tracker=tracker, references=references, mode=mode, threshold=0.5, model_id="campplus", worker=object(),
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


# --- D-10 (12-08-PLAN.md task 2): only the asker answers a follow-up ---------


def _asked_context(answer_only_from: "str | None") -> TurnContext:
    return TurnContext(
        turn_key="src:1",
        group_id="group-1",
        order_frame=0,
        follow_up=True,
        answer_only_from=answer_only_from,
    )


def _confirmation_request(pending_action_id: int = 1) -> FollowUpRequest:
    return FollowUpRequest(
        kind="confirmation",
        chain_depth=1,
        original_transcript="add dentist on friday at 3",
        question="add dentist to the home calendar, friday at 3 pm, for an hour?",
        pending_action_id=pending_action_id,
    )


async def _run_follow_up(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, *, incoming, turn_context, mode, transcript
):
    from atlas.providers.base import FinalTranscript

    spy = _ConfirmationSpy(ConfirmationOutcome(reply_text="confirmed", turn_outcome="confirmed"))
    monkeypatch.setattr(controller_module, "handle_confirmation_reply", spy)
    source = fake_audio_source(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming, window_opens_at=0.0, window_s=10.0)
    stt = fake_stt(events=[FinalTranscript(text=transcript)])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    recorder = SessionRecorder(SessionConfig(dir=str(tmp_path)), timings)
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
        speaker_id=_enforce_context(_identified_match(), mode),
        turn_context=turn_context,
    )
    return spy, source, tts, timings


async def test_enforce_mode_blocks_a_confirmation_answered_by_someone_else_and_leaves_the_row(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
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

    # The stub identifies speaker 1. Atlas asked speaker 2.
    spy, source, tts, timings = await _run_follow_up(
        monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        incoming=_confirmation_request(created.id), turn_context=_asked_context("2"), mode="enforce", transcript="yeah",
    )

    assert spy.calls == []
    assert tts.received_text == []
    assert source.sent_audio == []
    assert timings.turn_outcome == "follow_up_wrong_speaker"
    assert source.follow_up.requested is None
    assert (await pending_actions.get(created.id)).status == "awaiting"


async def test_enforce_mode_lets_the_asker_answer_a_confirmation(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    spy, _source, tts, timings = await _run_follow_up(
        monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        incoming=_confirmation_request(), turn_context=_asked_context("1"), mode="enforce", transcript="yeah",
    )

    assert len(spy.calls) == 1
    assert tts.received_text == ["confirmed"]
    assert timings.turn_outcome == "confirmed"


async def test_enforce_mode_blocks_a_clarification_answered_by_someone_else(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="turn off the light",
        question="which one -- kitchen or hallway?",
    )

    spy, source, tts, timings = await _run_follow_up(
        monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        incoming=incoming, turn_context=_asked_context("2"), mode="enforce", transcript="kitchen",
    )

    assert tts.received_text == []
    assert CANCELLED_REPLY not in tts.received_text
    assert timings.turn_outcome == "follow_up_wrong_speaker"
    assert source.follow_up.requested is None


async def test_record_mode_never_restricts_a_follow_up_by_speaker(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    spy, _source, tts, timings = await _run_follow_up(
        monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        incoming=_confirmation_request(), turn_context=_asked_context("2"), mode="record", transcript="yeah",
    )

    assert len(spy.calls) == 1
    assert timings.turn_outcome == "confirmed"


async def test_an_asking_turn_with_no_identified_speaker_does_not_restrict_the_follow_up(
    monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    spy, _source, _tts, timings = await _run_follow_up(
        monkeypatch, tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        incoming=_confirmation_request(), turn_context=_asked_context(None), mode="enforce", transcript="yeah",
    )

    assert len(spy.calls) == 1
    assert timings.turn_outcome == "confirmed"


def test_the_mismatch_helper_only_fires_for_a_follow_up_in_enforce_mode_with_a_different_speaker():
    incoming = _confirmation_request()
    asked = _asked_context("m-1")

    def check(*, incoming=incoming, context=asked, speaker="m-2", mode="enforce"):
        return follow_up_speaker_mismatch(
            context, incoming=incoming, speaker_event={"speaker_id": speaker}, effective_mode=mode
        )

    assert check() is True
    assert check(speaker="m-1") is False
    assert check(speaker=None) is True
    assert check(mode="record") is False
    assert check(mode="off") is False
    assert check(incoming=None) is False
    assert check(context=_asked_context(None)) is False
    assert follow_up_speaker_mismatch(None, incoming=incoming, speaker_event={}, effective_mode="enforce") is False
