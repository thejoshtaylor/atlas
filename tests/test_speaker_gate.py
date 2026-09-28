"""Boundary tests for `atlas.speaker_id.gate` (Phase 11 plan 02, Task 2)."""

from __future__ import annotations

import math

from atlas.speaker_id.gate import evaluate_speaker_gate


def test_score_equal_to_threshold_is_identified():
    decision = evaluate_speaker_gate(mode="enforce", enrolled_count=1, best_score=0.5, threshold=0.5)
    assert decision.identified
    assert decision.allowed


def test_score_one_float_step_below_threshold_is_not_identified():
    below_threshold = math.nextafter(0.5, -math.inf)
    decision = evaluate_speaker_gate(mode="enforce", enrolled_count=1, best_score=below_threshold, threshold=0.5)
    assert not decision.identified
    assert not decision.allowed
    assert decision.reason == "unknown_speaker"


def test_mode_off_never_blocks_even_with_no_score():
    decision = evaluate_speaker_gate(mode="off", enrolled_count=0, best_score=None, threshold=0.5)
    assert decision.allowed
    assert decision.effective_mode == "off"
    assert not decision.identified


def test_record_mode_never_blocks():
    decision = evaluate_speaker_gate(mode="record", enrolled_count=3, best_score=0.1, threshold=0.5)
    assert decision.allowed
    assert decision.effective_mode == "record"


def test_enforce_with_zero_enrolled_acts_as_record():
    decision = evaluate_speaker_gate(mode="enforce", enrolled_count=0, best_score=None, threshold=0.5)
    assert decision.allowed
    assert decision.effective_mode == "record"


def test_enforce_with_best_score_none_blocks():
    decision = evaluate_speaker_gate(mode="enforce", enrolled_count=2, best_score=None, threshold=0.5)
    assert not decision.allowed
    assert decision.reason == "unknown_speaker"
    assert decision.effective_mode == "enforce"


def test_enforce_with_identified_score_allows():
    decision = evaluate_speaker_gate(mode="enforce", enrolled_count=2, best_score=0.9, threshold=0.5)
    assert decision.allowed
    assert decision.identified
    assert decision.reason is None
