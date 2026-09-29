"""Behavior tests for `atlas.speaker_id.evaluation` (Phase 11 plan 05,
Task 2): word error rate, the EER sweep, and the FAR-threshold lookup.
Pure numpy against known, hand-computed examples -- no audio, no model.
"""

from __future__ import annotations

import pytest

from atlas.speaker_id.evaluation import (
    equal_error_rate,
    normalize_words,
    percentile,
    threshold_at_far,
    word_error_rate,
)


def test_normalize_words_drops_punctuation_and_lowercases():
    assert normalize_words("The quick, brown fox!") == ["the", "quick", "brown", "fox"]


def test_word_error_rate_known_example():
    # One deletion ("brown") out of four reference words.
    assert word_error_rate("the quick brown fox", "the quick fox") == pytest.approx(0.25)


def test_word_error_rate_exact_match_is_zero():
    assert word_error_rate("hey atlas turn on the light", "hey atlas turn on the light") == 0.0


def test_word_error_rate_empty_reference_and_hypothesis_is_zero():
    assert word_error_rate("", "") == 0.0


def test_word_error_rate_empty_reference_nonempty_hypothesis_is_one():
    assert word_error_rate("", "hello") == 1.0


def test_percentile_requires_at_least_one_value():
    with pytest.raises(ValueError):
        percentile([], 95)


def test_percentile_known_values():
    assert percentile([1.0, 2.0, 3.0, 4.0], 50) == pytest.approx(2.5)


def test_equal_error_rate_requires_both_groups_nonempty():
    with pytest.raises(ValueError):
        equal_error_rate([], [0.5])
    with pytest.raises(ValueError):
        equal_error_rate([0.5], [])


def test_equal_error_rate_fully_separated_is_zero_and_threshold_between_groups():
    genuine = [0.8, 0.9, 0.85]
    impostor = [0.1, 0.2, 0.15]
    eer, threshold = equal_error_rate(genuine, impostor)
    assert eer == pytest.approx(0.0)
    assert max(impostor) < threshold < min(genuine)


def test_equal_error_rate_hand_computed_overlapping_example():
    # genuine=[0.6, 0.8], impostor=[0.4, 0.7]. At threshold 0.65:
    #   far = mean(impostor >= 0.65) = 0.5 (0.7 counts, 0.4 doesn't)
    #   frr = mean(genuine < 0.65)   = 0.5 (0.6 counts, 0.8 doesn't)
    # No other midpoint candidate reaches |far - frr| == 0.
    genuine = [0.6, 0.8]
    impostor = [0.4, 0.7]
    eer, threshold = equal_error_rate(genuine, impostor)
    assert eer == pytest.approx(0.5)
    assert threshold == pytest.approx(0.65)


def test_threshold_at_far_requires_impostor_scores():
    with pytest.raises(ValueError):
        threshold_at_far([], 0.1)


def test_threshold_at_far_returns_lowest_threshold_meeting_target():
    # impostor scores: 0.1, 0.5, 0.9 -- a far of 1/3 must accept exactly the
    # top score (0.9) as a false accept, so the lowest threshold that keeps
    # far <= 1/3 sits strictly between 0.5 and 0.9.
    impostor = [0.1, 0.5, 0.9]
    threshold = threshold_at_far(impostor, 1 / 3)
    assert 0.5 < threshold < 0.9
