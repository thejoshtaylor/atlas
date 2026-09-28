"""Where one segment's audio holds more than one speaker (D-12).

This rule is this project's own design (11-RESEARCH.md Assumption A1), not
a library feature -- sherpa-onnx ships no streaming change-point primitive,
only an offline diarization pipeline measured at 2.4s for 20s of audio
(too slow for a turn's latency budget) and the plain extractor/manager pair
this phase already uses for matching. The Phase 11 spike (plan 11-08) tunes
`similarity_floor` against real house recordings and may replace this rule
entirely if it proves too noisy on real audio.

The rule: walk the windows in order, keeping a running reference -- the
normalized mean of the current part's windows so far. When a window's
cosine similarity to that reference drops below `similarity_floor`, a new
part starts at that window, and the reference resets to that window alone.
The first window is never a change point; it defines the first part.

This module is pure: no I/O, no imports from `turn/`, `providers/`, `db/`,
or `transports/`.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np

from atlas.speaker_id.matching import l2_normalize, mean_embedding


def find_change_points(
    embeddings: "Sequence[np.ndarray]",
    *,
    similarity_floor: float,
) -> "list[int]":
    """Return the indices in `embeddings` where a new part starts.

    An empty `embeddings` returns `[]`. The first window (index 0) is
    never itself a change point.
    """
    if not embeddings:
        return []

    change_points: "list[int]" = []
    current_part: "list[np.ndarray]" = [np.asarray(embeddings[0], dtype=np.float64)]
    reference = l2_normalize(current_part[0])

    for index in range(1, len(embeddings)):
        vector = np.asarray(embeddings[index], dtype=np.float64)
        similarity = float(np.dot(l2_normalize(vector), reference))
        if similarity < similarity_floor:
            change_points.append(index)
            current_part = [vector]
        else:
            current_part.append(vector)
        reference = mean_embedding(current_part)

    return change_points


def split_parts(window_count: int, change_points: "Sequence[int]") -> "list[range]":
    """Turn `change_points` (window indices where a new part starts) into
    contiguous window-index ranges covering `0..window_count`."""
    boundaries = [0, *change_points, window_count]
    return [range(boundaries[i], boundaries[i + 1]) for i in range(len(boundaries) - 1)]


def part_index_for_frame(parts: "Sequence[range]", windows: "Sequence", frame_index: int) -> int:
    """Return the index into `parts` whose windows span `frame_index`.

    A frame index that falls between two windows (never covered by any
    window's `first_frame_index..last_frame_index` span -- e.g. a silent
    frame no window kept) belongs to the *later* part: the first window
    whose own span reaches or passes `frame_index` decides it. A frame
    index before the first window belongs to the first part; one after the
    last window belongs to the last part.
    """
    if not windows or not parts:
        raise ValueError("part_index_for_frame: no windows/parts to search")

    window_index = len(windows) - 1  # default: after every window -- last window's part
    for candidate_index, window in enumerate(windows):
        if frame_index <= window.last_frame_index:
            window_index = candidate_index
            break

    for part_index, part_range in enumerate(parts):
        if window_index in part_range:
            return part_index
    return len(parts) - 1
