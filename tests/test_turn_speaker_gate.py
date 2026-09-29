"""The `vad_end` speaker gate as `run_turn` actually calls it (11-04-PLAN.md
Task 2): every mode, a `None` context (D-07), and a context with no
tracker (mode `"off"`) all record their own `speaker.result` event, and
only `enforce` with an identified-below-threshold or unenrolled voice ever
blocks.

Driven with a stub span/tracker (`_StubSpan`/`_StubTracker`) -- these tests
prove the *gate's* own behavior at every mode/context combination, not the
live `SpeakerTracker`'s streaming embedding pipeline (`tests/
test_speaker_tracer.py` already proves that end to end, and `tests/
test_speaker_tracker.py` covers the tracker's own unit behavior).
"""

from __future__ import annotations

import asyncio
import json

from atlas.config import SessionConfig
from atlas.session.recorder import SessionRecorder
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.tracker import SPEAKER_DECISION_TIMEOUT_S, SpeakerMeasurement
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn


class _StubSpan:
    """A `TurnSpeakerSpan`-shaped double: `decide()` returns whatever
    `SpeakerMeasurement` the test scripted, with no real tracker, no real
    embedding worker, and no real windows behind it."""

    def __init__(self, measurement: SpeakerMeasurement) -> None:
        self._measurement = measurement
        self.decide_calls: "list[dict]" = []
        # 11-06-PLAN.md Task 1 (D-12): `run_turn` reads `speaker_span.
        # split_event` unconditionally off every span it opens -- never set
        # by these tests, which drive the gate's own decision, not the
        # drain's live-detection race.
        self.split_event = asyncio.Event()

    async def decide(self, *, end_of_speech_at, references, timeout_s=SPEAKER_DECISION_TIMEOUT_S):
        self.decide_calls.append(
            {"end_of_speech_at": end_of_speech_at, "references": references, "timeout_s": timeout_s}
        )
        return self._measurement


class _StubTracker:
    """A `SpeakerTracker`-shaped double: `open_turn()` always returns the
    one span the test built, and records every call's `start_after`."""

    def __init__(self, span: "_StubSpan") -> None:
        self._span = span
        self.open_turn_calls: "list[float | None]" = []

    def open_turn(self, *, start_after=None):
        self.open_turn_calls.append(start_after)
        return self._span


def _reference_set(*, alice_vector=(1.0, 0.0), bob_vector=(0.0, 1.0)) -> ReferenceSet:
    references = ReferenceSet()
    references.upsert_speaker(1, "Alice", [alice_vector])
    if bob_vector is not None:
        references.upsert_speaker(2, "Bob", [bob_vector])
    return references


def _measurement_for(match: "MatchResult | None", *, detail=None) -> SpeakerMeasurement:
    return SpeakerMeasurement(
        match=match, speech_ms=750.0, window_count=2, ready_at=100.0, speaker_id_ms=42.0, detail=detail,
    )


def _read_events(recorder: SessionRecorder) -> "list[dict]":
    lines = (recorder.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _find_speaker_result(events: "list[dict]") -> dict:
    for event in events:
        if event.get("type") == "speaker.result":
            return event
    raise AssertionError(f"no speaker.result event found among {[e.get('type') for e in events]}")


async def _run_turn_with_speaker_id(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, *, speaker_id, final_text="turn on the fan"
):
    from atlas.providers.base import BrainReply, FinalTranscript

    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text=final_text)])
    brain = fake_brain(replies=[BrainReply(text="the fan is on")])
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
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
        speaker_id=speaker_id,
    )
    return source, tts, timings, recorder


async def test_enforce_blocks_a_below_threshold_match(tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts):
    references = _reference_set()
    match = MatchResult(best_speaker_id=1, best_name="Alice", best_score=0.2, second_score=0.1, margin=0.1, scores={1: 0.2, 2: 0.1})
    span = _StubSpan(_measurement_for(match))
    tracker = _StubTracker(span)
    context = SpeakerIdTurnContext(
        tracker=tracker, references=references, mode="enforce", threshold=0.5, model_id="campplus", worker=object(),
    )

    source, tts, timings, recorder = await _run_turn_with_speaker_id(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, speaker_id=context,
    )

    assert timings.turn_outcome == "unknown_speaker"
    assert tts.received_text == []
    assert source.sent_audio == []

    events = _read_events(recorder)
    result = _find_speaker_result(events)
    assert result["status"] == "unknown"
    assert result["blocked"] is True
    assert result["reason"] == "unknown_speaker"
    assert result["detail"] == "below_threshold"
    assert result["best_speaker_id"] == 1
    assert result["speaker_id"] is None
    assert result["speaker_name"] is None


async def test_record_mode_runs_the_turn_and_records_an_unblocked_below_threshold_match(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    references = _reference_set()
    match = MatchResult(best_speaker_id=1, best_name="Alice", best_score=0.2, second_score=0.1, margin=0.1, scores={1: 0.2, 2: 0.1})
    span = _StubSpan(_measurement_for(match))
    tracker = _StubTracker(span)
    context = SpeakerIdTurnContext(
        tracker=tracker, references=references, mode="record", threshold=0.5, model_id="campplus", worker=object(),
    )

    source, tts, timings, recorder = await _run_turn_with_speaker_id(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, speaker_id=context,
    )

    assert timings.turn_outcome == "completed"
    assert tts.received_text == ["the fan is on"]

    result = _find_speaker_result(_read_events(recorder))
    assert result["blocked"] is False
    assert result["effective_mode"] == "record"
    assert result["status"] == "unknown"
    assert result["detail"] == "below_threshold"
    # 11-09's tuning script reads every enrolled member's own score, and
    # the best raw candidate id, regardless of whether it gated anything.
    assert result["scores"] == {"1": 0.2, "2": 0.1}
    assert result["best_speaker_id"] == 1


async def test_enforce_with_zero_enrolled_members_degrades_to_record(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    references = ReferenceSet()  # nobody enrolled.
    span = _StubSpan(_measurement_for(None))
    tracker = _StubTracker(span)
    context = SpeakerIdTurnContext(
        tracker=tracker, references=references, mode="enforce", threshold=0.5, model_id="campplus", worker=object(),
    )

    source, tts, timings, recorder = await _run_turn_with_speaker_id(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, speaker_id=context,
    )

    assert timings.turn_outcome == "completed"
    assert tts.received_text == ["the fan is on"]

    result = _find_speaker_result(_read_events(recorder))
    assert result["blocked"] is False
    assert result["effective_mode"] == "record"
    assert result["status"] == "unknown"
    assert result["detail"] == "no_members_enrolled"


async def test_mode_off_context_runs_unblocked_and_awaits_no_span(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    context = SpeakerIdTurnContext(tracker=None, references=None, mode="off", threshold=0.5, model_id=None, worker=None)

    source, tts, timings, recorder = await _run_turn_with_speaker_id(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, speaker_id=context,
    )

    assert timings.turn_outcome == "completed"
    assert tts.received_text == ["the fan is on"]

    result = _find_speaker_result(_read_events(recorder))
    assert result["status"] == "unknown"
    assert result["blocked"] is False
    assert result["effective_mode"] == "off"
    assert result["detail"] == "speaker_id_off"


async def test_no_speaker_id_context_runs_unblocked_and_records_not_edge_source(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    source, tts, timings, recorder = await _run_turn_with_speaker_id(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, speaker_id=None,
    )

    assert timings.turn_outcome == "completed"
    assert tts.received_text == ["the fan is on"]

    result = _find_speaker_result(_read_events(recorder))
    assert result["status"] == "unknown"
    assert result["blocked"] is False
    assert result["effective_mode"] == "off"
    assert result["detail"] == "not_edge_source"
