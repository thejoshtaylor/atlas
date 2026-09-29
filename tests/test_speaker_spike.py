"""Behavior tests for the D-05/D-09 spike scorer (11-05-PLAN.md): every
`question_q*` rule tested against known inputs, and `score_corpus` proved
end to end on a synthetic corpus written through the real
`edge/spike/speaker_corpus.py::write_clip` (`FakeEmbedder` standing in for
sherpa-onnx, since no real model or hardware is available in this
environment -- RESEARCH.md's own Environment Availability table).

`scripts/speaker_spike.py` and `edge/spike/speaker_corpus.py` are both
loaded by file path (`scripts/` and `edge/spike/` are not on this
project's `pythonpath`), the same technique `tests/test_score_wake_engines.py`
already uses for `scripts/score_wake_engines.py`.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

from tests.speaker_fakes import FakeEmbedder

_SPIKE_PATH = Path(__file__).resolve().parent.parent / "scripts" / "speaker_spike.py"
_spike_spec = importlib.util.spec_from_file_location("speaker_spike", _SPIKE_PATH)
assert _spike_spec is not None and _spike_spec.loader is not None
speaker_spike = importlib.util.module_from_spec(_spike_spec)
sys.modules["speaker_spike"] = speaker_spike
_spike_spec.loader.exec_module(speaker_spike)

_CORPUS_TOOL_PATH = Path(__file__).resolve().parent.parent / "edge" / "spike" / "speaker_corpus.py"
_corpus_spec = importlib.util.spec_from_file_location("speaker_corpus_contract", _CORPUS_TOOL_PATH)
assert _corpus_spec is not None and _corpus_spec.loader is not None
speaker_corpus = importlib.util.module_from_spec(_corpus_spec)
sys.modules["speaker_corpus_contract"] = speaker_corpus
_corpus_spec.loader.exec_module(speaker_corpus)

SAMPLE_RATE = 16000
FRAME_SAMPLES = 256


# --- fixtures ----------------------------------------------------------------


def _constant_pcm16(amplitude: int, seconds: float, sample_rate: int = SAMPLE_RATE) -> bytes:
    n = int(seconds * sample_rate)
    return np.full(n, amplitude, dtype="<i2").tobytes()


def _full_intervals(seconds: float) -> "list[list[float]]":
    return [[0.0, seconds * 1000.0]]


def _write_full_speech_clip(root, label, kind, prompt, amplitude, seconds=1.0, sample_rate=SAMPLE_RATE):
    n = int(seconds * sample_rate)
    samples = np.full((n, 2), amplitude, dtype=np.int16)
    speech_flags = [True] * (n // FRAME_SAMPLES)
    return speaker_corpus.write_clip(root, label, kind, prompt, samples, speech_flags, sample_rate)


def _fake_transcriber(mapping: dict) -> "callable":
    return lambda pcm16: mapping.get(pcm16, "")


def _fake_embedder_factory(model: str) -> FakeEmbedder:
    return FakeEmbedder()


# --- contract: the two enrollment-phrase lists never drift apart -----------


def test_enrollment_prompts_pinned_to_atlas_phrases():
    from atlas.speaker_id.phrases import ENROLLMENT_PHRASES

    assert speaker_corpus.ENROLLMENT_PROMPTS == ENROLLMENT_PHRASES


# --- Q1 (D-08) ---------------------------------------------------------------


def test_question_q1_correct_channel0_dropped_word_channel1_picks_0():
    prompt = "hey atlas turn on the kitchen light"
    ch0, ch1 = b"ch0-audio", b"ch1-audio"
    transcriber = _fake_transcriber({ch0: prompt, ch1: "hey atlas turn on the kitchen"})
    result = speaker_spike.question_q1([(prompt, ch0, ch1)] * 10, transcriber)
    assert result["result"] == "PASS"
    assert result["asr_channel"] == 0


def test_question_q1_reverse_picks_channel_1():
    prompt = "hey atlas turn on the kitchen light"
    ch0, ch1 = b"ch0-audio", b"ch1-audio"
    transcriber = _fake_transcriber({ch0: "hey atlas turn on the kitchen", ch1: prompt})
    result = speaker_spike.question_q1([(prompt, ch0, ch1)] * 10, transcriber)
    assert result["asr_channel"] == 1


def test_question_q1_difference_under_one_point_keeps_channel_0():
    prompt = "hey atlas turn on the kitchen light"
    ch0, ch1 = b"ch0-audio", b"ch1-audio"
    transcriber = _fake_transcriber({ch0: prompt, ch1: prompt})
    result = speaker_spike.question_q1([(prompt, ch0, ch1)] * 10, transcriber)
    assert result["asr_channel"] == 0


def test_question_q1_nine_clips_fails():
    prompt = "hey atlas turn on the kitchen light"
    result = speaker_spike.question_q1([(prompt, b"a", b"b")] * 9, lambda pcm16: prompt)
    assert result["result"].startswith("FAIL")


# --- Q6 -----------------------------------------------------------------


def test_question_q6_rms_0_004_gives_floor_0_006():
    result = speaker_spike.question_q6([0.004] * 100)
    assert result["result"] == "PASS"
    assert result["speech_rms_floor"] == pytest.approx(0.006)


def test_question_q6_fewer_than_100_frames_fails():
    result = speaker_spike.question_q6([0.004] * 99)
    assert result["result"].startswith("FAIL")


# --- Q3 -----------------------------------------------------------------


def test_question_q3_three_windows_tie_picks_500():
    enrollment_by_label = {
        "member-a": [(_constant_pcm16(5000, 2.0), _full_intervals(2.0))],
        "member-b": [(_constant_pcm16(-5000, 2.0), _full_intervals(2.0))],
    }
    command_clips = [("member-a", _constant_pcm16(5000, 2.0), _full_intervals(2.0))] * 10 + [
        ("member-b", _constant_pcm16(-5000, 2.0), _full_intervals(2.0))
    ] * 10
    other_clips = [(_constant_pcm16(15000, 2.0), _full_intervals(2.0))] * 2

    result = speaker_spike.question_q3(
        winning_model="campplus",
        embedder_factory=_fake_embedder_factory,
        enrollment_by_label=enrollment_by_label,
        command_clips=command_clips,
        other_clips=other_clips,
        speech_rms_floor=0.001,
    )
    assert result["result"] == "PASS"
    assert result["winner_window_ms"] == 500
    for window_ms in (500, 750, 1000):
        assert result["windows"][window_ms]["eer"] == pytest.approx(0.0)


# --- Q4 -----------------------------------------------------------------


def test_question_q4_threshold_rounded_and_between_groups():
    result = speaker_spike.question_q4(genuine_scores=[0.9, 0.95], impostor_scores=[0.1, 0.2])
    assert result["result"] == "PASS"
    assert result["threshold"] == round(result["threshold"], 2)
    assert 0.2 < result["threshold"] < 0.9


def test_question_q4_no_scores_fails():
    result = speaker_spike.question_q4(genuine_scores=[], impostor_scores=[])
    assert result["result"].startswith("FAIL")


# --- Q5 -----------------------------------------------------------------


def test_question_q5_detects_diff_label_joins_never_splits_same_label():
    fake = FakeEmbedder()
    vec_a = fake.embed(_constant_pcm16(5000, 0.1))
    vec_b = fake.embed(_constant_pcm16(-5000, 0.1))
    labels = (["member-a", "member-b"] * 6)  # 6 of each -> 30 diff pairs, 60 same pairs
    clip_windows = [(label, [vec_a if label == "member-a" else vec_b] * 3) for label in labels]

    result = speaker_spike.question_q5(clip_windows=clip_windows, window_ms=500)
    assert result["result"] == "PASS"
    assert result["chosen_floor"] is not None
    assert result["mean_lag_ms"] is not None
    for floor, report in result["floors"].items():
        if floor > 0.0:
            assert report["detection_rate"] == pytest.approx(1.0)
            assert report["false_split_rate"] == pytest.approx(0.0)


def test_question_q5_too_few_pairs_fails():
    fake = FakeEmbedder()
    vec_a = fake.embed(_constant_pcm16(5000, 0.1))
    vec_b = fake.embed(_constant_pcm16(-5000, 0.1))
    # 4 member-a clips, 1 member-b clip: 8 different-label pairs, below the floor of 10.
    labels = ["member-a"] * 4 + ["member-b"]
    clip_windows = [(label, [vec_a if label == "member-a" else vec_b] * 3) for label in labels]

    result = speaker_spike.question_q5(clip_windows=clip_windows, window_ms=500)
    assert result["result"].startswith("FAIL")


# --- Q7 -----------------------------------------------------------------


def test_question_q7_single_interval_clips_give_share_1_and_gap_800():
    intervals = [[[0.0, 100.0]]] * 5
    result = speaker_spike.question_q7(intervals)
    assert result["result"] == "PASS"
    assert result["single_interval_share"] == 1.0
    assert result["enrollment_gap_ms"] == 800


def test_question_q7_900ms_gap_raises_enrollment_gap_to_1100():
    intervals = [[[0.0, 100.0]]] * 4 + [[[0.0, 100.0], [1000.0, 1100.0]]]
    result = speaker_spike.question_q7(intervals)
    assert result["enrollment_gap_ms"] == 1100


def test_question_q7_fewer_than_5_clips_fails():
    result = speaker_spike.question_q7([[[0.0, 100.0]]] * 4)
    assert result["result"].startswith("FAIL")


# --- Q8 / Q9: informational, never gate the overall verdict -----------------


def test_overall_verdict_ignores_q8_and_q9():
    results = {
        "q1": {"result": "PASS"},
        "q2": {"result": "PASS"},
        "q5": {"result": "PASS"},
        "q6": {"result": "PASS"},
        "q7": {"result": "PASS"},
        "q8": {"result": "FAIL (only 2 reply clip(s), need at least 5 -- informational only)"},
        "q9": {"result": "FAIL (no embedding time measurements)"},
    }
    assert speaker_spike.overall_verdict(results) == "PASS"


def test_overall_verdict_fails_naming_each_failed_question():
    results = {
        "q1": {"result": "PASS"},
        "q2": {"result": "FAIL (too few trials)"},
        "q5": {"result": "FAIL (no floor keeps false splits at or below 10%)"},
        "q6": {"result": "PASS"},
        "q7": {"result": "PASS"},
    }
    verdict = speaker_spike.overall_verdict(results)
    assert verdict.startswith("FAIL (")
    assert "Q2:" in verdict
    assert "Q5:" in verdict
    assert "Q1:" not in verdict


# --- score_corpus: end to end on a synthetic corpus ------------------------


def test_score_corpus_two_members_and_other_gives_q2_pass_with_enough_trials(tmp_path):
    root = tmp_path / "speakers"
    for i in range(10):
        _write_full_speech_clip(root, "member-a", "command", f"hey atlas, command {i}", amplitude=5000)
        _write_full_speech_clip(root, "member-b", "command", f"hey atlas, command {i}", amplitude=-5000)
    _write_full_speech_clip(root, "member-a", "enrollment", "the morning light", amplitude=5000)
    _write_full_speech_clip(root, "member-b", "enrollment", "the morning light", amplitude=-5000)
    for i in range(2):
        _write_full_speech_clip(root, "other", "other", None, amplitude=15000)

    report = speaker_spike.score_corpus(root, embedder_factory=_fake_embedder_factory, transcriber=lambda pcm16: "")

    q2 = report["q2"]
    assert q2["result"] == "PASS"
    for model, model_result in q2["models"].items():
        assert model_result["genuine_trials"] >= 20
        assert model_result["impostor_trials"] >= 20
        assert model_result["eer"] == pytest.approx(0.0)


def test_score_corpus_one_member_no_other_gives_q2_fail_naming_impostor(tmp_path):
    root = tmp_path / "speakers"
    for i in range(20):
        _write_full_speech_clip(root, "member-a", "command", f"hey atlas, command {i}", amplitude=5000)
    _write_full_speech_clip(root, "member-a", "enrollment", "the morning light", amplitude=5000)

    report = speaker_spike.score_corpus(root, embedder_factory=_fake_embedder_factory, transcriber=lambda pcm16: "")

    assert report["q2"]["result"].startswith("FAIL")
    assert "impostor" in report["q2"]["result"]


def test_score_corpus_report_ends_with_keys_and_verdict(tmp_path):
    root = tmp_path / "speakers"
    for i in range(10):
        _write_full_speech_clip(root, "member-a", "command", f"hey atlas, command {i}", amplitude=5000)
        _write_full_speech_clip(root, "member-b", "command", f"hey atlas, command {i}", amplitude=-5000)
    _write_full_speech_clip(root, "member-a", "enrollment", "the morning light", amplitude=5000)
    _write_full_speech_clip(root, "member-b", "enrollment", "the morning light", amplitude=-5000)

    report = speaker_spike.score_corpus(root, embedder_factory=_fake_embedder_factory, transcriber=lambda pcm16: "")

    assert "keys" in report
    assert "verdict" in report
    for key in (
        "asr_channel", "model", "window_ms", "threshold", "change_similarity_floor",
        "speech_rms_floor", "enrollment_gap_ms", "short_reply_acceptance", "embed_ms_p95",
    ):
        assert key in report["keys"]
