"""D-12 (11-06-PLAN.md Task 1): a speaker change inside one segment splits
the audio at the change point. Only the part that holds the wake phrase
decides the turn; every other part is recorded as a `speaker.split` event
and dropped.

Three layers, matching the plan's own `<behavior>`:

- `SpeakerTracker`/`TurnSpeakerSpan` unit behavior -- live detection
  (`_check_for_split`, hooked into the existing window-done callback, no
  extra task) and the final split (`decide()`'s own `dropped_parts`).
  Driven the same way `tests/test_speaker_tracker.py` drives the tracker:
  a real `SpeechWindowAccumulator` behind a synchronous, controllable
  embedding worker.
- The drain (`turn/controller.py::_drain_to_final_transcript`): a
  `split_event` finalizes the utterance the same way a real `vad.end`
  already does, once a word has been heard.
- `run_turn` wiring: one `speaker.split` event per dropped part, the kept
  part's own `speaker.result` event still recorded, and the D-16 fallback
  chain for `speaker_id_ms`'s own `end_of_speech_at`.
"""

from __future__ import annotations

import asyncio
import logging
import struct
import time

import pytest

import test_turn_speaker_gate as ttsg
from atlas.providers.base import FinalTranscript, PartialTranscript
from atlas.speaker_id.matching import MatchResult
from atlas.speaker_id.tracker import SPEAKER_DECISION_TIMEOUT_S, DroppedPart, SpeakerMeasurement, SpeakerTracker
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext, evaluate_turn_speaker
from atlas.timing import TurnTimings
from atlas.transports.base import SourceFormat
from atlas.transports.edge import SpeechSignals

from tests.speaker_fakes import FakeEmbedder

_SAMPLE_RATE = 16000


def _voice_frame(value: int, ms: int = 1000) -> bytes:
    """A `ms`-long frame at a constant PCM16 value -- exactly one window's
    worth of samples at this file's own `window_ms=1000`, so a single
    `on_frame` call closes exactly one window. Two different `value`s land
    in different `FakeEmbedder` buckets (`tests/speaker_fakes.py`), i.e.
    two different "voices"."""
    samples = int(_SAMPLE_RATE * ms / 1000)
    return struct.pack(f"<{samples}h", *([value] * samples))


class _SyncWorker:
    """A synchronous stand-in for `EmbeddingWorker`: resolves each
    submission immediately with `FakeEmbedder`'s own deterministic vector
    -- no thread, so live change-point detection (the tracker's own
    window-done callback) can be driven and observed one
    `asyncio.sleep(0)` at a time, the same way `_CountingWorker` in
    `tests/test_speaker_tracker.py` drives cadence assertions."""

    def __init__(self, embedder: FakeEmbedder) -> None:
        self._embedder = embedder

    def submit(self, pcm16_mono: bytes) -> "asyncio.Future":
        future = asyncio.get_event_loop().create_future()
        future.set_result(self._embedder.embed(pcm16_mono))
        return future


def _tracker(worker, *, similarity_floor: float = 0.3) -> SpeakerTracker:
    return SpeakerTracker(
        channels=1,
        asr_channel=0,
        sample_rate=_SAMPLE_RATE,
        window_ms=1000,
        min_window_ms=100,
        speech_rms_floor=0.01,
        change_similarity_floor=similarity_floor,
        worker=worker,
    )


# ---------------------------------------------------------------------------
# SpeakerTracker / TurnSpeakerSpan: live detection and the final split
# ---------------------------------------------------------------------------


async def test_a_second_voice_after_the_wake_part_sets_split_event_and_is_dropped():
    """Voice A (holding the wake frame), then voice B: `split_event` is set
    the moment B's window finishes embedding, and `decide()` identifies A
    from A's own windows only, with B recorded as the one dropped part."""
    worker = _SyncWorker(FakeEmbedder(dim=8))
    tracker = _tracker(worker)
    tracker.on_vad_start(1, 0.0)

    tracker.on_frame(_voice_frame(2000), 0, 10.0)  # A's window: started_at=10.0
    tracker.on_wake_hit(0)
    span = tracker.open_turn()
    await asyncio.sleep(0)
    assert not span.split_event.is_set(), "one window alone can never show a change"

    tracker.on_frame(_voice_frame(20000), 1, 11.0)  # B's window: started_at=11.0
    await asyncio.sleep(0)
    assert span.split_event.is_set(), "B's window should trip live detection the instant it embeds"
    assert span.split_detected_at is not None

    tracker.on_vad_end(2, 12.0)
    measurement = await span.decide(end_of_speech_at=None, references=None)

    assert measurement.detail is None
    assert measurement.window_count == 1
    assert measurement.speech_ms == pytest.approx(1000.0, abs=1.0)
    assert len(measurement.dropped_parts) == 1
    dropped = measurement.dropped_parts[0]
    assert dropped.speech_ms == pytest.approx(1000.0, abs=1.0)
    assert dropped.started_at == 11.0


async def test_a_voice_before_the_wake_part_never_sets_split_event():
    """B first, then A (holding the wake frame): the only change point is
    BEFORE the kept part, so `split_event` is never set -- nothing after
    the wake phrase needs to stop. `decide()` still identifies A alone and
    drops B."""
    worker = _SyncWorker(FakeEmbedder(dim=8))
    tracker = _tracker(worker)
    tracker.on_vad_start(1, 0.0)

    tracker.on_frame(_voice_frame(20000), 0, 10.0)  # B's window
    await asyncio.sleep(0)

    tracker.on_wake_hit(1)  # the wake frame will land in A's window, index 1
    span = tracker.open_turn()

    tracker.on_frame(_voice_frame(2000), 1, 11.0)  # A's window, holds the wake frame
    await asyncio.sleep(0)
    assert not span.split_event.is_set(), "the change is before the kept part -- nothing to stop for"

    tracker.on_vad_end(2, 12.0)
    measurement = await span.decide(end_of_speech_at=None, references=None)

    assert measurement.window_count == 1
    assert len(measurement.dropped_parts) == 1
    assert measurement.dropped_parts[0].started_at == 10.0


async def test_one_voice_throughout_never_splits():
    """No change at all: `split_event` is never set, and `dropped_parts`
    is empty -- byte-identical to every measurement before this plan."""
    worker = _SyncWorker(FakeEmbedder(dim=8))
    tracker = _tracker(worker)
    tracker.on_vad_start(1, 0.0)

    tracker.on_frame(_voice_frame(2000), 0, 10.0)
    tracker.on_wake_hit(0)
    span = tracker.open_turn()
    await asyncio.sleep(0)

    tracker.on_frame(_voice_frame(2000), 1, 11.0)
    await asyncio.sleep(0)
    assert not span.split_event.is_set()

    tracker.on_vad_end(2, 12.0)
    measurement = await span.decide(end_of_speech_at=None, references=None)

    assert measurement.dropped_parts == ()
    assert measurement.window_count == 2
    assert measurement.speech_ms == pytest.approx(2000.0, abs=1.0)


async def test_follow_up_span_kept_part_is_the_first_part_after_start_after():
    """D-11: a follow-up span has no wake frame at all -- the kept part is
    simply the first part chronologically after `start_after`, whichever
    voice it happens to be."""
    worker = _SyncWorker(FakeEmbedder(dim=8))
    tracker = _tracker(worker)
    tracker.on_vad_start(1, 0.0)

    tracker.on_frame(_voice_frame(2000), 0, 0.0)  # before start_after -- excluded entirely
    tracker.on_frame(_voice_frame(20000), 1, 1.0)  # first part in scope
    tracker.on_frame(_voice_frame(2000), 2, 2.0)  # a second part in scope
    tracker.on_vad_end(3, 3.0)

    span = tracker.open_turn(start_after=0.5)
    measurement = await span.decide(end_of_speech_at=None, references=None)

    assert measurement.window_count == 1
    assert measurement.speech_ms == pytest.approx(1000.0, abs=1.0)
    assert len(measurement.dropped_parts) == 1
    assert measurement.dropped_parts[0].started_at == 2.0


# ---------------------------------------------------------------------------
# The drain: split_event finalizes like a real vad.end, once a word is heard
# ---------------------------------------------------------------------------


class _SpeechSignalsSource:
    def __init__(self, speech_signals: SpeechSignals) -> None:
        self.speech_signals = speech_signals

    async def frames(self):
        while True:
            yield b"\x00\x00"
            await asyncio.sleep(0.01)

    def source_format(self) -> SourceFormat:
        return SourceFormat("pcm", 16000)


class _WordThenGatedStt:
    """Yields one partial carrying a word, then hangs until finalized."""

    async def stream(self, frames, source_format, *, finalize=None):
        yield PartialTranscript(text="turn on the fan")
        await finalize.wait()
        yield FinalTranscript(text="turn on the fan")


async def test_split_event_finalizes_before_a_real_vad_end_once_a_word_is_heard():
    """The real scenario this plan exists for: a second voice's own change
    point is detected before the segment's genuine `vad.end` ever arrives.
    `finalize` is set, `timings.speaker_split_at` is recorded, and
    `timings.vad_end_at` stays `None` -- the `vad.end` watch never got
    anywhere near its own event."""
    from atlas.turn.controller import _drain_to_final_transcript

    signals = SpeechSignals(hangover_s=0.0)
    signals.publish({"type": "vad.start", "seq": 1})  # in_speech True; no vad.end ever published.
    source = _SpeechSignalsSource(signals)
    stt = _WordThenGatedStt()
    timings = TurnTimings()
    split_event = asyncio.Event()

    async def _split_soon() -> None:
        await asyncio.sleep(0.02)
        split_event.set()

    asyncio.ensure_future(_split_soon())

    final = await asyncio.wait_for(
        _drain_to_final_transcript(
            source,
            stt,
            15.0,
            timings,
            clock=time.monotonic,
            poll_interval_s=0.01,
            speech_signals=signals,
            split_event=split_event,
        ),
        timeout=2.0,
    )

    assert final is not None
    assert final.text == "turn on the fan"
    assert timings.speaker_split_at is not None
    assert timings.vad_end_at is None


async def test_split_event_none_leaves_the_drain_unchanged():
    """`split_event=None` (the default, and every caller that predates
    this plan) starts no second watch task -- a real `vad.end` finalizes
    exactly as it always has."""
    from atlas.turn.controller import _drain_to_final_transcript

    signals = SpeechSignals(hangover_s=0.0)
    signals.publish({"type": "vad.start", "seq": 1})
    source = _SpeechSignalsSource(signals)
    stt = _WordThenGatedStt()
    timings = TurnTimings()

    async def _end_speech_soon() -> None:
        await asyncio.sleep(0.02)
        signals.publish({"type": "vad.end", "seq": 2})

    asyncio.ensure_future(_end_speech_soon())

    final = await asyncio.wait_for(
        _drain_to_final_transcript(
            source,
            stt,
            15.0,
            timings,
            clock=time.monotonic,
            poll_interval_s=0.01,
            speech_signals=signals,
        ),
        timeout=2.0,
    )

    assert final is not None
    assert timings.vad_end_at is not None
    assert timings.speaker_split_at is None


# ---------------------------------------------------------------------------
# speaker_id_ms's own end_of_speech_at: vad_end_at, else speaker_split_at,
# else stt_final_at
# ---------------------------------------------------------------------------


class _RecordingSpan:
    """Records the `end_of_speech_at` `evaluate_turn_speaker` actually
    passed to `decide()`, and returns a fixed, otherwise-uninteresting
    measurement."""

    def __init__(self, measurement: SpeakerMeasurement) -> None:
        self._measurement = measurement
        self.end_of_speech_at: "float | None" = "unset"

    async def decide(self, *, end_of_speech_at, references, timeout_s: float = SPEAKER_DECISION_TIMEOUT_S):
        self.end_of_speech_at = end_of_speech_at
        return self._measurement


def _no_speech_measurement() -> SpeakerMeasurement:
    return SpeakerMeasurement(
        match=None, speech_ms=0.0, window_count=0, ready_at=None, speaker_id_ms=None, detail="no_speech_measured",
    )


async def test_end_of_speech_at_prefers_vad_end_at_then_speaker_split_at_then_stt_final_at():
    context = SpeakerIdTurnContext(
        tracker=object(), references=None, mode="off", threshold=0.5, model_id=None, worker=None,
    )

    span = _RecordingSpan(_no_speech_measurement())
    timings = TurnTimings()
    timings.vad_end_at = 10.0
    timings.speaker_split_at = 20.0
    timings.stt_final_at = 30.0
    await evaluate_turn_speaker(context, span, timings=timings)
    assert span.end_of_speech_at == 10.0

    span = _RecordingSpan(_no_speech_measurement())
    timings = TurnTimings()
    timings.speaker_split_at = 20.0
    timings.stt_final_at = 30.0
    await evaluate_turn_speaker(context, span, timings=timings)
    assert span.end_of_speech_at == 20.0

    span = _RecordingSpan(_no_speech_measurement())
    timings = TurnTimings()
    timings.stt_final_at = 30.0
    await evaluate_turn_speaker(context, span, timings=timings)
    assert span.end_of_speech_at == 30.0


# ---------------------------------------------------------------------------
# run_turn: one speaker.split event per dropped part, the kept part's own
# speaker.result event still records, and the log line names no speaker
# ---------------------------------------------------------------------------


def _measurement_with_two_dropped_parts() -> SpeakerMeasurement:
    kept_match = MatchResult(
        best_speaker_id=1, best_name="Alice", best_score=0.9, second_score=0.2, margin=0.7,
        scores={1: 0.9, 2: 0.85, 3: 0.2},
    )
    identified_drop_match = MatchResult(
        best_speaker_id=2, best_name="Bob", best_score=0.85, second_score=0.2, margin=0.65,
        scores={1: 0.2, 2: 0.85, 3: 0.1},
    )
    unknown_drop_match = MatchResult(
        best_speaker_id=3, best_name="Carol", best_score=0.2, second_score=0.15, margin=0.05,
        scores={1: 0.1, 2: 0.15, 3: 0.2},
    )
    identified_drop = DroppedPart(
        first_frame_index=10, last_frame_index=20, started_at=1.5, speech_ms=500.0, match=identified_drop_match,
    )
    unknown_drop = DroppedPart(
        first_frame_index=30, last_frame_index=40, started_at=3.0, speech_ms=300.0, match=unknown_drop_match,
    )
    return SpeakerMeasurement(
        match=kept_match, speech_ms=1000.0, window_count=2, ready_at=100.0, speaker_id_ms=42.0, detail=None,
        dropped_parts=(identified_drop, unknown_drop),
    )


async def test_run_turn_records_one_speaker_split_event_per_dropped_part(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, caplog
):
    measurement = _measurement_with_two_dropped_parts()
    span = ttsg._StubSpan(measurement)
    tracker = ttsg._StubTracker(span)
    context = SpeakerIdTurnContext(
        tracker=tracker, references=ttsg._reference_set(), mode="enforce", threshold=0.5,
        model_id="campplus", worker=object(),
    )

    with caplog.at_level(logging.INFO, logger="atlas.turn.controller"):
        source, tts, timings, recorder = await ttsg._run_turn_with_speaker_id(
            tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, speaker_id=context,
        )

    assert timings.turn_outcome == "completed"
    assert tts.received_text == ["the fan is on"]

    events = ttsg._read_events(recorder)
    result = ttsg._find_speaker_result(events)
    assert result["status"] == "identified"
    assert result["speaker_name"] == "Alice"

    split_events = [e for e in events if e.get("type") == "speaker.split"]
    assert len(split_events) == 2
    for event in split_events:
        assert event["kept"] is False
        assert event["offset_s"] is not None

    by_speaker_id = {e["best_speaker_id"]: e for e in split_events}
    # D-14/D-15-shaped: a dropped part's own name is shown only when its
    # own best score reaches the gate's threshold -- exactly the bar an
    # identified turn itself must clear, never a looser one.
    assert by_speaker_id[2]["speaker_name"] == "Bob"
    assert by_speaker_id[2]["speech_ms"] == 500.0
    assert by_speaker_id[3]["speaker_name"] is None
    assert by_speaker_id[3]["score"] == 0.2

    # D-15/D-21: the log line names the count, never a name.
    split_logs = [r.message for r in caplog.records if "speaker part" in r.message]
    assert len(split_logs) == 1
    assert "2" in split_logs[0]
    assert "Bob" not in split_logs[0]
    assert "Carol" not in split_logs[0]
