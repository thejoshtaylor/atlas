"""Boundary tests for `atlas.speaker_id.matching` (Phase 11 plan 02, Task 2)."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.speaker_id.matching import MatchResult, ReferenceSet, l2_normalize


def test_l2_normalize_rejects_zero_vector():
    with pytest.raises(ValueError):
        l2_normalize(np.zeros(3))


def test_l2_normalize_rejects_non_finite_vector():
    with pytest.raises(ValueError):
        l2_normalize(np.array([1.0, float("nan"), 0.0]))


def test_empty_reference_set_returns_every_field_none():
    reference_set = ReferenceSet()
    result = reference_set.match(np.array([1.0, 0.0, 0.0]))
    assert result == MatchResult(None, None, None, None, None, {})


def test_one_member_gives_margin_none():
    reference_set = ReferenceSet()
    reference_set.upsert_speaker(1, "Alice", [np.array([1.0, 0.0, 0.0])])
    result = reference_set.match(np.array([1.0, 0.0, 0.0]))
    assert result.margin is None
    assert result.second_score is None


def test_two_members_with_equal_scores_give_margin_zero():
    reference_set = ReferenceSet()
    reference_set.upsert_speaker(1, "Alice", [np.array([1.0, 1.0, 0.0])])
    reference_set.upsert_speaker(2, "Bob", [np.array([1.0, -1.0, 0.0])])
    result = reference_set.match(np.array([1.0, 0.0, 0.0]))
    assert result.margin == pytest.approx(0.0)


def test_non_finite_embedding_raises():
    reference_set = ReferenceSet()
    reference_set.upsert_speaker(1, "Alice", [np.array([1.0, 0.0, 0.0])])
    with pytest.raises(ValueError):
        reference_set.match(np.array([1.0, float("nan"), 0.0]))


def test_dimension_mismatch_on_match_raises():
    reference_set = ReferenceSet()
    reference_set.upsert_speaker(1, "Alice", [np.array([1.0, 0.0, 0.0])])
    with pytest.raises(ValueError):
        reference_set.match(np.array([1.0, 0.0]))


def test_dimension_mismatch_on_upsert_raises():
    reference_set = ReferenceSet()
    reference_set.upsert_speaker(1, "Alice", [np.array([1.0, 0.0, 0.0])])
    with pytest.raises(ValueError):
        reference_set.upsert_speaker(2, "Bob", [np.array([1.0, 0.0])])


def test_remove_speaker_of_the_only_member_leaves_zero_enrolled():
    reference_set = ReferenceSet()
    reference_set.upsert_speaker(1, "Alice", [np.array([1.0, 0.0, 0.0])])
    reference_set.remove_speaker(1)
    assert reference_set.enrolled_count == 0
    assert reference_set.name_for(1) is None


def test_upsert_speaker_replaces_never_appends():
    reference_set = ReferenceSet()
    reference_set.upsert_speaker(1, "Alice", [np.array([1.0, 0.0, 0.0])])
    reference_set.upsert_speaker(1, "Alice", [np.array([0.0, 1.0, 0.0])])
    result = reference_set.match(np.array([0.0, 1.0, 0.0]))
    # If the second call had appended rather than replaced, the reference
    # would be a blend of both vectors and could never score a perfect 1.0
    # against a pure [0, 1, 0] turn embedding.
    assert result.best_score == pytest.approx(1.0)


def test_remove_speaker_is_a_no_op_for_an_unknown_id():
    reference_set = ReferenceSet()
    reference_set.remove_speaker(999)
    assert reference_set.enrolled_count == 0
