"""Behavior tests for the speaker-id core: windows, matching, and the gate
(Phase 11 plan 02, Task 1). A synthetic voice travels frames -> windows ->
a fake per-window embedding -> a turn embedding -> a match -> a gate
decision, the same chain the live pipeline (plan 11-04) will run.
"""

from __future__ import annotations

import typing

import numpy as np
import pytest

from atlas.speaker_id.gate import evaluate_speaker_gate
from atlas.speaker_id.matching import ReferenceSet, mean_embedding
from atlas.speaker_id.windows import SpeechWindowAccumulator
from atlas.wake.gate import BlockReason

SAMPLE_RATE = 16000
FRAME_SAMPLES = 256
LOUD_AMPLITUDE = 20000
SPEECH_RMS_FLOOR = 0.1  # LOUD_AMPLITUDE / 32768 ~= 0.61, well above this floor


def _frame(amplitude: int) -> bytes:
    return np.full(FRAME_SAMPLES, amplitude, dtype="<i2").tobytes()


def _push_loud_frames(acc: SpeechWindowAccumulator, count: int, start_index: int = 0) -> "list":
    windows = []
    for i in range(count):
        window = acc.push(_frame(LOUD_AMPLITUDE), frame_index=start_index + i, at=float(start_index + i))
        if window is not None:
            windows.append(window)
    return windows


# ---------------------------------------------------------------------------
# SpeechWindowAccumulator
# ---------------------------------------------------------------------------


def test_1_2s_of_loud_frames_produces_two_windows_of_500ms():
    # 75 frames * 256 samples = 19200 samples = 1.2s at 16kHz.
    acc = SpeechWindowAccumulator(sample_rate=SAMPLE_RATE, window_ms=500, min_window_ms=150, speech_rms_floor=SPEECH_RMS_FLOOR)
    acc.start_segment(1)
    windows = _push_loud_frames(acc, 75)
    assert len(windows) == 2
    assert windows[0].speech_ms == pytest.approx(500.0)
    assert windows[1].speech_ms == pytest.approx(500.0)
    assert windows[0].segment_seq == 1
    assert len(windows[0].pcm) == int(500 / 1000 * SAMPLE_RATE) * 2


def test_end_segment_returns_a_200ms_tail_when_min_window_ms_is_150():
    acc = SpeechWindowAccumulator(sample_rate=SAMPLE_RATE, window_ms=500, min_window_ms=150, speech_rms_floor=SPEECH_RMS_FLOOR)
    acc.start_segment(1)
    _push_loud_frames(acc, 75)
    tail = acc.end_segment()
    assert tail is not None
    assert tail.speech_ms == pytest.approx(200.0)


def test_end_segment_returns_none_when_min_window_ms_is_250():
    acc = SpeechWindowAccumulator(sample_rate=SAMPLE_RATE, window_ms=500, min_window_ms=250, speech_rms_floor=SPEECH_RMS_FLOOR)
    acc.start_segment(1)
    _push_loud_frames(acc, 75)
    tail = acc.end_segment()
    assert tail is None


def test_frames_below_floor_never_enter_a_window_or_advance_the_count():
    acc = SpeechWindowAccumulator(sample_rate=SAMPLE_RATE, window_ms=500, min_window_ms=150, speech_rms_floor=SPEECH_RMS_FLOOR)
    acc.start_segment(1)
    # 50 silent frames precede the speech -- pre-roll silence.
    for i in range(50):
        assert acc.push(_frame(0), frame_index=i, at=0.0) is None
    windows = _push_loud_frames(acc, 75, start_index=50)
    assert len(windows) == 2
    tail = acc.end_segment()
    assert tail is not None
    assert tail.speech_ms == pytest.approx(200.0)
    # None of the emitted PCM contains the silent pre-roll bytes -- every
    # sample in every emitted window is the loud amplitude, never zero.
    for window in [*windows, tail]:
        samples = np.frombuffer(window.pcm, dtype="<i2")
        assert np.all(samples == LOUD_AMPLITUDE)


def test_start_segment_discards_previous_segments_partial_buffer():
    acc = SpeechWindowAccumulator(sample_rate=SAMPLE_RATE, window_ms=500, min_window_ms=150, speech_rms_floor=SPEECH_RMS_FLOOR)
    acc.start_segment(1)
    _push_loud_frames(acc, 10)  # 10 * 256 samples = 160ms, well under window_ms -- never closes
    acc.start_segment(2)
    tail = acc.end_segment()
    assert tail is None  # segment 1's partial buffer never carried over


# ---------------------------------------------------------------------------
# ReferenceSet / matching
# ---------------------------------------------------------------------------


def test_reference_set_matches_the_correct_member_with_margin():
    rng = np.random.default_rng(42)
    unit_a = np.array([1.0, 0.0, 0.0])
    unit_b = np.array([0.0, 1.0, 0.0])

    def near(vector: "np.ndarray", scale: float = 0.02) -> "np.ndarray":
        return vector + rng.normal(scale=scale, size=vector.shape)

    rows = [{"speaker_id": 1, "display_name": "Alice", "vector": near(unit_a)} for _ in range(3)]
    rows += [{"speaker_id": 2, "display_name": "Bob", "vector": near(unit_b)} for _ in range(3)]
    reference_set = ReferenceSet.load(rows)

    result = reference_set.match(near(unit_a))
    assert result.best_speaker_id == 1
    assert result.best_score is not None and result.best_score > 0.9
    assert result.margin is not None and result.margin > 0.0


# ---------------------------------------------------------------------------
# Tracer: frames -> windows -> fake embeddings -> mean -> match -> gate
# ---------------------------------------------------------------------------


def test_tracer_frames_become_a_gate_decision():
    unit_a = np.array([1.0, 0.0, 0.0])
    unit_b = np.array([0.0, 1.0, 0.0])
    unit_c = np.array([0.0, 0.0, 1.0])  # orthogonal to both enrolled members
    threshold = 0.5

    rows = [{"speaker_id": 1, "display_name": "Alice", "vector": unit_a} for _ in range(3)]
    rows += [{"speaker_id": 2, "display_name": "Bob", "vector": unit_b} for _ in range(3)]
    reference_set = ReferenceSet.load(rows)

    # Voice A: real frames through the real windowing accumulator.
    acc = SpeechWindowAccumulator(sample_rate=SAMPLE_RATE, window_ms=500, min_window_ms=150, speech_rms_floor=SPEECH_RMS_FLOOR)
    acc.start_segment(1)
    windows = _push_loud_frames(acc, 75)
    tail = acc.end_segment()
    if tail is not None:
        windows.append(tail)
    assert len(windows) >= 1

    # Fake per-window embeddings: every window of "voice A" maps to a
    # vector near the enrolled A reference -- the embedder itself (real
    # sherpa-onnx inference) is plan 11-01's concern, not this one's.
    fake_window_vectors = [unit_a for _ in windows]
    turn_embedding = mean_embedding(fake_window_vectors)

    match_a = reference_set.match(turn_embedding)
    decision_a = evaluate_speaker_gate(
        mode="enforce", enrolled_count=reference_set.enrolled_count, best_score=match_a.best_score, threshold=threshold
    )
    assert match_a.best_speaker_id == 1
    assert decision_a.allowed
    assert decision_a.identified

    # Voice C: orthogonal to every enrolled member -- blocked.
    match_c = reference_set.match(unit_c)
    decision_c = evaluate_speaker_gate(
        mode="enforce", enrolled_count=reference_set.enrolled_count, best_score=match_c.best_score, threshold=threshold
    )
    assert not decision_c.allowed
    assert decision_c.reason == "unknown_speaker"


# ---------------------------------------------------------------------------
# The wake gate's BlockReason is unchanged
# ---------------------------------------------------------------------------


def test_wake_gate_block_reason_is_unchanged():
    assert typing.get_args(BlockReason) == ("below_threshold", "refractory", "media_playing")
