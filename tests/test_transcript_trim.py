"""D-04, D-12 (12-08-PLAN.md, task 3): a turn that a speaker change ended
early transcribes again from its own first frame to the split frame, so its
text holds none of the next speaker's words.

Every id, name, and phrase here is a generic placeholder.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from atlas.transports.base import SourceFormat
from atlas.config import SessionConfig
from atlas.providers.base import BrainReply, FinalTranscript, PartialTranscript
from atlas.session.recorder import SessionRecorder
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.tracker import SPEAKER_DECISION_TIMEOUT_S, SpeakerMeasurement
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.transcript_trim import TRANSCRIPT_TRIM_EVENT, retranscribe
from atlas.turn.turn_context import TurnContext
from tests.conftest import FakeBrain

_FORMAT = SourceFormat("pcm", 16000)
_FRAMES = [b"\x00\x01", b"\x00\x02", b"\x00\x03", b"\x00\x04"]
_LONG_TEXT = "hey atlas turn on the lamp and open the door"
_TRIMMED_TEXT = "hey atlas turn on the lamp"


class _ScriptedStt:
    """Answers each `stream` call with the next scripted behavior and records
    the frames it read and whether `finalize` was set once the frames ended."""

    def __init__(self, *texts: str, hang: bool = False, raises: bool = False, no_final: bool = False) -> None:
        self._texts = list(texts)
        self._hang = hang
        self._raises = raises
        self._no_final = no_final
        self.calls: list[list[bytes]] = []
        self.finalize_set_after_frames: list[bool] = []
        self.closed = 0

    async def stream(self, frames, source_format=None, *, finalize: "asyncio.Event | None" = None):
        received: list[bytes] = []
        self.calls.append(received)
        try:
            async for frame in frames:
                received.append(frame)
            self.finalize_set_after_frames.append(finalize is not None and finalize.is_set())
            if self._raises:
                raise RuntimeError("stream broke")
            if self._hang:
                await asyncio.Event().wait()
            yield PartialTranscript(text="partial")
            if not self._no_final:
                yield FinalTranscript(text=self._texts.pop(0))
        finally:
            self.closed += 1


async def test_retranscribe_feeds_exactly_the_frames_sets_finalize_after_the_last_and_returns_the_text():
    stt = _ScriptedStt("turn on the lamp")

    text = await retranscribe(stt, list(_FRAMES[:2]), _FORMAT, timeout_s=2.0)

    assert text == "turn on the lamp"
    assert stt.calls == [_FRAMES[:2]]
    assert stt.finalize_set_after_frames == [True]
    assert stt.closed == 1


@pytest.mark.parametrize(
    "kwargs",
    [{"raises": True}, {"no_final": True}, {"hang": True}],
    ids=["raises", "no_final", "timeout"],
)
async def test_retranscribe_returns_none_when_the_stream_fails_ends_without_a_final_or_times_out(kwargs):
    stt = _ScriptedStt("never", **kwargs)

    text = await retranscribe(stt, list(_FRAMES), _FORMAT, timeout_s=0.1)

    assert text is None
    assert stt.closed == 1


class _StubSpan:
    def __init__(self, *, split: bool, split_frame_index: "int | None") -> None:
        match = MatchResult(
            best_speaker_id=1, best_name="Josh", best_score=0.9, second_score=0.1, margin=0.8, scores={1: 0.9}
        )
        self._measurement = SpeakerMeasurement(
            match=match, speech_ms=750.0, window_count=2, ready_at=100.0, speaker_id_ms=42.0, detail=None
        )
        self.split_event = asyncio.Event()
        if split:
            self.split_event.set()
        self.split_frame_index = split_frame_index

    async def decide(self, *, end_of_speech_at, references, timeout_s=SPEAKER_DECISION_TIMEOUT_S):
        return self._measurement

    def close(self) -> None:
        pass


class _RecordingBrain(FakeBrain):
    def __init__(self, replies) -> None:
        super().__init__(replies=replies)
        self.received: list[list[dict]] = []

    async def chat(self, messages, tools=None):
        self.received.append(list(messages))
        return await super().chat(messages, tools)


def _speaker_context() -> SpeakerIdTurnContext:
    references = ReferenceSet()
    references.upsert_speaker(1, "Josh", [(1.0, 0.0)])
    return SpeakerIdTurnContext(
        tracker=object(), references=references, mode="record", threshold=0.5, model_id="model", worker=object()
    )


async def _run(tmp_path, fake_audio_source, fake_tts, stt, *, span, replay_until, with_context=True):
    source = fake_audio_source(frames=list(_FRAMES))
    brain = _RecordingBrain([BrainReply(text="done")])
    timings = TurnTimings()
    recorder = SessionRecorder(SessionConfig(dir=str(tmp_path)), timings)
    context = (
        TurnContext(
            turn_key="src:1",
            group_id="group-1",
            order_frame=0,
            speaker_span=span,
            replay_until=replay_until,
        )
        if with_context
        else None
    )
    await asyncio.wait_for(
        run_turn(
            source,
            stt,
            brain,
            fake_tts(chunks=[b"\x01\x02"]),
            None,
            tools_schema=[],
            system_prompt="you control a home",
            max_tool_rounds=3,
            timings=timings,
            session_recorder=recorder,
            speaker_id=_speaker_context() if with_context else None,
            turn_context=context,
            wake_phrase="hey atlas",
        ),
        5.0,
    )
    lines = (recorder.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    events = [json.loads(line) for line in lines if line.strip()]
    return brain, [event for event in events if event.get("type") == TRANSCRIPT_TRIM_EVENT]


def _user_text(brain: _RecordingBrain) -> str:
    return brain.received[0][-1]["content"]


async def test_a_split_turn_answers_only_its_own_words_and_records_one_trim_event(
    tmp_path, fake_audio_source, fake_tts
):
    stt = _ScriptedStt(_LONG_TEXT, _TRIMMED_TEXT)
    asked_for: list[int] = []

    def replay_until(end: int):
        asked_for.append(end)
        return list(_FRAMES[:end])

    brain, trims = await _run(
        tmp_path,
        fake_audio_source,
        fake_tts,
        stt,
        span=_StubSpan(split=True, split_frame_index=2),
        replay_until=replay_until,
    )

    assert asked_for == [2]
    assert stt.calls[1] == _FRAMES[:2]
    # The wake phrase strip ran on the new text.
    assert _user_text(brain) == "turn on the lamp"
    (trim,) = trims
    assert trim["method"] == "retranscribe"
    assert trim["frames"] == 2
    assert isinstance(trim["duration_ms"], float) and trim["duration_ms"] >= 0.0


async def test_a_turn_keeps_its_first_transcript_when_the_frames_left_the_history(
    tmp_path, fake_audio_source, fake_tts
):
    stt = _ScriptedStt(_LONG_TEXT)

    brain, trims = await _run(
        tmp_path,
        fake_audio_source,
        fake_tts,
        stt,
        span=_StubSpan(split=True, split_frame_index=2),
        replay_until=lambda end: None,
    )

    assert len(stt.calls) == 1
    assert _user_text(brain) == "turn on the lamp and open the door"
    (trim,) = trims
    assert trim["method"] == "unavailable"


async def test_a_turn_keeps_its_first_transcript_when_the_second_transcription_fails(
    tmp_path, fake_audio_source, fake_tts
):
    class _FailingSecond(_ScriptedStt):
        async def stream(self, frames, source_format=None, *, finalize=None):
            if self.calls:
                raise RuntimeError("stream broke")
            async for event in super().stream(frames, source_format, finalize=finalize):
                yield event

    stt = _FailingSecond(_LONG_TEXT)

    brain, _trims = await _run(
        tmp_path,
        fake_audio_source,
        fake_tts,
        stt,
        span=_StubSpan(split=True, split_frame_index=2),
        replay_until=lambda end: list(_FRAMES[:end]),
    )

    assert _user_text(brain) == "turn on the lamp and open the door"


async def test_a_turn_with_no_split_never_transcribes_again(tmp_path, fake_audio_source, fake_tts):
    stt = _ScriptedStt(_LONG_TEXT)

    brain, trims = await _run(
        tmp_path,
        fake_audio_source,
        fake_tts,
        stt,
        span=_StubSpan(split=False, split_frame_index=None),
        replay_until=lambda end: list(_FRAMES[:end]),
    )

    assert len(stt.calls) == 1
    assert trims == []
    assert _user_text(brain) == "turn on the lamp and open the door"


async def test_a_turn_with_no_turn_context_never_transcribes_again(tmp_path, fake_audio_source, fake_tts):
    stt = _ScriptedStt(_LONG_TEXT)

    _brain, trims = await _run(
        tmp_path, fake_audio_source, fake_tts, stt, span=None, replay_until=None, with_context=False
    )

    assert len(stt.calls) == 1
    assert trims == []
