"""Boundary tests for `atlas.speaker_id.change_point` (Phase 11 plan 02, Task 2)."""

from __future__ import annotations

import numpy as np
import pytest

from atlas.speaker_id.change_point import find_change_points, part_index_for_frame, split_parts
from atlas.speaker_id.windows import SpeechWindow

UNIT_A = np.array([1.0, 0.0, 0.0])
UNIT_B = np.array([0.0, 1.0, 0.0])


def _near(vector: "np.ndarray", rng: "np.random.Generator", scale: float = 0.02) -> "np.ndarray":
    return vector + rng.normal(scale=scale, size=vector.shape)


def test_find_change_points_five_a_then_four_b_returns_five():
    rng = np.random.default_rng(1)
    embeddings = [_near(UNIT_A, rng) for _ in range(5)] + [_near(UNIT_B, rng) for _ in range(4)]
    assert find_change_points(embeddings, similarity_floor=0.5) == [5]


def test_find_change_points_nine_a_returns_empty():
    rng = np.random.default_rng(2)
    embeddings = [_near(UNIT_A, rng) for _ in range(9)]
    assert find_change_points(embeddings, similarity_floor=0.5) == []


def test_find_change_points_a_b_a_returns_two():
    rng = np.random.default_rng(3)
    embeddings = (
        [_near(UNIT_A, rng) for _ in range(4)]
        + [_near(UNIT_B, rng) for _ in range(4)]
        + [_near(UNIT_A, rng) for _ in range(4)]
    )
    change_points = find_change_points(embeddings, similarity_floor=0.5)
    assert len(change_points) == 2
    assert change_points == [4, 8]


def test_find_change_points_empty_list_returns_empty():
    assert find_change_points([], similarity_floor=0.5) == []


def test_find_change_points_first_window_is_never_a_change_point():
    rng = np.random.default_rng(4)
    embeddings = [_near(UNIT_B, rng)] + [_near(UNIT_A, rng) for _ in range(3)]
    change_points = find_change_points(embeddings, similarity_floor=0.5)
    assert 0 not in change_points


def test_split_parts_turns_one_change_point_into_two_ranges():
    parts = split_parts(9, [5])
    assert list(parts[0]) == list(range(0, 5))
    assert list(parts[1]) == list(range(5, 9))


def _window(first: int, last: int) -> SpeechWindow:
    return SpeechWindow(
        segment_seq=1,
        first_frame_index=first,
        last_frame_index=last,
        started_at=0.0,
        speech_ms=500.0,
        pcm=b"",
    )


def test_part_index_for_frame_inside_each_part():
    windows = [_window(0, 9), _window(10, 19), _window(20, 29), _window(30, 39)]
    parts = split_parts(4, [2])  # part 0: windows 0-1, part 1: windows 2-3
    assert part_index_for_frame(parts, windows, 5) == 0
    assert part_index_for_frame(parts, windows, 25) == 1


def test_part_index_for_frame_before_the_first_window():
    windows = [_window(10, 19), _window(20, 29)]
    parts = split_parts(2, [1])
    assert part_index_for_frame(parts, windows, 0) == 0


def test_part_index_for_frame_after_the_last_window():
    windows = [_window(0, 9), _window(10, 19)]
    parts = split_parts(2, [1])
    assert part_index_for_frame(parts, windows, 100) == 1


def test_part_index_for_frame_between_two_parts_belongs_to_the_later_part():
    windows = [_window(0, 9), _window(20, 29)]  # gap: frames 10-19 held no window
    parts = split_parts(2, [1])  # part 0: window 0, part 1: window 1
    assert part_index_for_frame(parts, windows, 15) == 1


def test_part_index_for_frame_raises_with_no_windows():
    with pytest.raises(ValueError):
        part_index_for_frame([], [], 0)
