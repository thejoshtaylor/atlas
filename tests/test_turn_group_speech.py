"""`run_turn` and `_speak` inside a reply group (Phase 12, plan 12-07).

Two concurrent `run_turn` calls share one `GroupSpeaker`. Each gets a
`TurnContext` with its own reply handle, its own `order_frame`, and a stub
speaker span. These tests drive the real `run_turn` and `_speak`, with the
fakes from `conftest.py`, and prove that the group speaks through one TTS
call while each turn keeps its own session record.

Every id, name, and phrase here is a generic placeholder.
"""

from __future__ import annotations

import asyncio
import json
import time

from atlas.config import SessionConfig
from atlas.providers.base import BrainReply, FinalTranscript
from atlas.session.recorder import SessionRecorder
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.tracker import SPEAKER_DECISION_TIMEOUT_S, SpeakerMeasurement
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.reply_group import GroupSpeaker, current_reply_route
from atlas.turn.turn_context import TurnContext

_TIMEOUT_S = 5.0


class _StubSpan:
    """A `TurnSpeakerSpan`-shaped double that identifies one scripted member."""

    def __init__(self, measurement: SpeakerMeasurement) -> None:
        self._measurement = measurement
        self.split_event = asyncio.Event()
        self.decide_calls = 0
        self.close_calls = 0

    async def decide(self, *, end_of_speech_at, references, timeout_s=SPEAKER_DECISION_TIMEOUT_S):
        self.decide_calls += 1
        return self._measurement

    def close(self) -> None:
        self.close_calls += 1


class _StubTracker:
    def __init__(self, span: _StubSpan | None = None) -> None:
        self._span = span
        self.open_turn_calls: list = []

    def open_turn(self, *, start_after=None):
        self.open_turn_calls.append(start_after)
        return self._span


def _references() -> ReferenceSet:
    references = ReferenceSet()
    references.upsert_speaker(1, "Josh", [(1.0, 0.0)])
    references.upsert_speaker(2, "Sam", [(0.0, 1.0)])
    return references


def _span_for(speaker_id: int, name: str) -> _StubSpan:
    match = MatchResult(
        best_speaker_id=speaker_id,
        best_name=name,
        best_score=0.9,
        second_score=0.1,
        margin=0.8,
        scores={speaker_id: 0.9},
    )
    return _StubSpan(
        SpeakerMeasurement(match=match, speech_ms=750.0, window_count=2, ready_at=100.0, speaker_id_ms=42.0, detail=None)
    )


def _speaker_context(tracker: _StubTracker) -> SpeakerIdTurnContext:
    return SpeakerIdTurnContext(
        tracker=tracker, references=_references(), mode="record", threshold=0.5, model_id="model", worker=object()
    )


def _events(recorder: SessionRecorder) -> list[dict]:
    lines = (recorder.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _of_type(events: list[dict], event_type: str) -> list[dict]:
    return [event for event in events if event.get("type") == event_type]


class _Turn:
    """One turn's doubles, and a coroutine that runs it inside the group."""

    def __init__(
        self,
        tmp_path,
        speaker: GroupSpeaker,
        *,
        key: str,
        order_frame: int,
        answer: str,
        live_tts,
        fake_audio_source,
        fake_stt,
        fake_brain,
        member: tuple[int, str] | None = None,
        transcript: str = "turn something on",
        group_id: str = "group-1",
        with_handle: bool = True,
    ) -> None:
        self.key = key
        self.source = fake_audio_source(frames=[b"\x00\x01"] * 3)
        self.stt = fake_stt(events=[FinalTranscript(text=transcript)])
        self.brain = fake_brain(replies=[BrainReply(text=answer)])
        self.tts = live_tts
        self.timings = TurnTimings()
        self.recorder = SessionRecorder(SessionConfig(dir=str(tmp_path / key)), self.timings)
        self.span = _span_for(*member) if member is not None else None
        self.tracker = _StubTracker(self.span)
        self.handle = speaker.register(key, order_frame, group_id=group_id) if with_handle else None
        self.context = TurnContext(
            turn_key=key,
            group_id=group_id,
            order_frame=order_frame,
            speaker_span=self.span,
            reply_group=self.handle,
        )

    async def run(self, **extra) -> None:
        try:
            await run_turn(
                self.source,
                self.stt,
                self.brain,
                self.tts,
                None,
                tools_schema=[],
                system_prompt="you control a home",
                max_tool_rounds=3,
                timings=self.timings,
                session_recorder=self.recorder,
                speaker_id=_speaker_context(self.tracker),
                turn_context=self.context,
                **extra,
            )
        finally:
            if self.handle is not None:
                self.handle.finish()

    @property
    def events(self) -> list[dict]:
        return _events(self.recorder)


async def _run_both(*turns: _Turn, **extra) -> None:
    await asyncio.wait_for(asyncio.gather(*(turn.run(**extra) for turn in turns)), _TIMEOUT_S)


async def test_two_turns_in_one_group_speak_one_merged_reply_in_speech_order(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(live_tts=live_tts, fake_audio_source=fake_audio_source, fake_stt=fake_stt, fake_brain=fake_brain)
    josh = _Turn(tmp_path, speaker, key="src:1", order_frame=100, answer="the fan is on", member=(1, "Josh"), **common)
    sam = _Turn(tmp_path, speaker, key="src:2", order_frame=300, answer="the door is locked", member=(2, "Sam"), **common)

    # Sam is started first: speech order comes from `order_frame`, not arrival.
    await _run_both(sam, josh)

    assert live_tts.received_text == ["Josh, the fan is on. Sam, the door is locked."]
    assert len(josh.source.sent_audio) + len(sam.source.sent_audio) == 1


async def test_each_turn_keeps_its_own_reply_text_and_group_events_with_no_name(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(live_tts=live_tts, fake_audio_source=fake_audio_source, fake_stt=fake_stt, fake_brain=fake_brain)
    josh = _Turn(tmp_path, speaker, key="src:1", order_frame=100, answer="the fan is on", member=(1, "Josh"), **common)
    sam = _Turn(tmp_path, speaker, key="src:2", order_frame=300, answer="the door is locked", member=(2, "Sam"), **common)

    await _run_both(josh, sam)

    for turn, own_answer in ((josh, "the fan is on"), (sam, "the door is locked")):
        events = turn.events
        replies = _of_type(events, "reply.text")
        assert [event["text"] for event in replies] == [own_answer]
        (turn_group,) = _of_type(events, "turn.group")
        (reply_group,) = _of_type(events, "reply.group")
        assert turn_group["turn_key"] == turn.key
        assert reply_group["turn_key"] == turn.key
        assert reply_group["group_id"] == "group-1"
        assert reply_group["merged_count"] == 2
        assert reply_group["prefixed"] is True
        # T-12-21: no group event carries a speaker name.
        for event in (turn_group, reply_group):
            assert "Josh" not in json.dumps(event)
            assert "Sam" not in json.dumps(event)
    roles = sorted(_of_type(turn.events, "reply.group")[0]["role"] for turn in (josh, sam))
    assert roles == ["follow", "lead"]


async def test_both_turns_get_answer_audio_at_from_the_shared_write(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(live_tts=live_tts, fake_audio_source=fake_audio_source, fake_stt=fake_stt, fake_brain=fake_brain)
    josh = _Turn(tmp_path, speaker, key="src:1", order_frame=100, answer="the fan is on", member=(1, "Josh"), **common)
    sam = _Turn(tmp_path, speaker, key="src:2", order_frame=300, answer="the door is locked", member=(2, "Sam"), **common)

    await _run_both(josh, sam)

    assert josh.timings.answer_audio_at is not None
    assert sam.timings.answer_audio_at is not None
    assert abs(josh.timings.answer_audio_at - sam.timings.answer_audio_at) < 0.05
    assert josh.timings.first_audio_at is not None and sam.timings.first_audio_at is not None


async def test_the_runner_opened_span_is_used_and_closed_and_the_tracker_opens_none(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(live_tts=live_tts, fake_audio_source=fake_audio_source, fake_stt=fake_stt, fake_brain=fake_brain)
    josh = _Turn(tmp_path, speaker, key="src:1", order_frame=100, answer="the fan is on", member=(1, "Josh"), **common)
    sam = _Turn(tmp_path, speaker, key="src:2", order_frame=300, answer="the door is locked", member=(2, "Sam"), **common)

    await _run_both(josh, sam)

    for turn in (josh, sam):
        assert turn.tracker.open_turn_calls == []
        assert turn.span.decide_calls == 1
        assert turn.span.close_calls == 1


async def test_note_speaker_gets_the_identified_id_and_name_before_the_reply(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(live_tts=live_tts, fake_audio_source=fake_audio_source, fake_stt=fake_stt, fake_brain=fake_brain)
    josh = _Turn(tmp_path, speaker, key="src:1", order_frame=100, answer="the fan is on", member=(1, "Josh"), **common)
    sam = _Turn(tmp_path, speaker, key="src:2", order_frame=300, answer="the door is locked", member=(2, "Sam"), **common)

    await _run_both(josh, sam)

    assert josh.context.speaker_label == "Josh"
    assert josh.context.speaker_id == 1
    assert sam.context.speaker_label == "Sam"
    assert sam.context.speaker_id == 2
    # The merged text names both, so both labels reached their handles.
    assert live_tts.received_text[0].startswith("Josh, ")
    assert "Sam, " in live_tts.received_text[0]


async def test_a_turn_alone_on_the_parallel_path_speaks_its_own_text_with_its_own_tts(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    solo = _Turn(
        tmp_path,
        speaker,
        key="src:1",
        order_frame=100,
        answer="the fan is on",
        member=(1, "Josh"),
        live_tts=live_tts,
        fake_audio_source=fake_audio_source,
        fake_stt=fake_stt,
        fake_brain=fake_brain,
    )

    await _run_both(solo)

    assert live_tts.received_text == ["the fan is on"]
    (reply_group,) = _of_type(solo.events, "reply.group")
    assert reply_group["role"] == "alone"
    assert [event["text"] for event in _of_type(solo.events, "reply.text")] == ["the fan is on"]


async def test_a_turn_context_with_no_group_and_no_span_opens_the_tracker_span_and_speaks_directly(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    turn = _Turn(
        tmp_path,
        speaker,
        key="src:1",
        order_frame=100,
        answer="the fan is on",
        live_tts=live_tts,
        fake_audio_source=fake_audio_source,
        fake_stt=fake_stt,
        fake_brain=fake_brain,
        with_handle=False,
    )
    tracker_span = _span_for(1, "Josh")
    turn.tracker = _StubTracker(tracker_span)

    await _run_both(turn)

    assert turn.tracker.open_turn_calls == [None]
    assert tracker_span.close_calls == 1
    assert live_tts.received_text == ["the fan is on"]
    assert _of_type(turn.events, "reply.group") == []
    assert len(_of_type(turn.events, "turn.group")) == 1


async def test_the_reply_route_is_reset_when_the_turn_ends(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    speaker = GroupSpeaker(merge_wait_s=0.2)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    turn = _Turn(
        tmp_path,
        speaker,
        key="src:1",
        order_frame=100,
        answer="the fan is on",
        member=(1, "Josh"),
        live_tts=live_tts,
        fake_audio_source=fake_audio_source,
        fake_stt=fake_stt,
        fake_brain=fake_brain,
    )

    # Awaited in this task, so a leaked ContextVar would show here.
    await asyncio.wait_for(turn.run(), _TIMEOUT_S)

    assert current_reply_route.get() is None


async def test_a_turn_with_no_turn_context_is_unchanged(tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    started = time.monotonic()

    await run_turn(
        source,
        fake_stt(events=[FinalTranscript(text="turn something on")]),
        fake_brain(replies=[BrainReply(text="the fan is on")]),
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == ["the fan is on"]
    assert timings.answer_audio_at is not None and timings.answer_audio_at >= started
    assert current_reply_route.get() is None
