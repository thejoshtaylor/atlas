"""Evaluation math for the Phase 11 spike scorer and plan 11-09's threshold
tuning script (11-05-PLAN.md).

Pure numpy and stdlib only: no I/O, no `sherpa_onnx`, no other
`atlas.speaker_id` import. Every function here takes plain scores or word
lists, never a corpus path or a config object, so both scripts can reuse it
without sharing any other state.
"""

from __future__ import annotations

import re
from typing import Sequence

import numpy as np

_WORD_RE = re.compile(r"[^\w']+")


def normalize_words(text: str) -> "list[str]":
    """Lowercase `text` and split it into words, dropping punctuation-only
    tokens. `"The quick, brown fox!"` becomes `["the", "quick", "brown",
    "fox"]`."""
    return [w for w in _WORD_RE.sub(" ", text.lower()).split() if w]


def word_error_rate(reference: str, hypothesis: str) -> float:
    """Word-level Levenshtein distance between `reference` and
    `hypothesis`, divided by the reference's word count. An empty
    reference is `0.0` if the hypothesis is also empty, `1.0` otherwise --
    there is no reference length to divide by."""
    ref = normalize_words(reference)
    hyp = normalize_words(hypothesis)
    if not ref:
        return 0.0 if not hyp else 1.0

    n, m = len(ref), len(hyp)
    row = list(range(m + 1))
    for i in range(1, n + 1):
        prev = row[0]
        row[0] = i
        for j in range(1, m + 1):
            temp = row[j]
            if ref[i - 1] == hyp[j - 1]:
                row[j] = prev
            else:
                row[j] = 1 + min(prev, row[j], row[j - 1])
            prev = temp
    return row[m] / n


def percentile(values: Sequence[float], q: float) -> float:
    """The `q`th percentile of `values` (0-100). Raises `ValueError` for an
    empty sequence -- there is no percentile of nothing."""
    if not values:
        raise ValueError("percentile: at least one value is required")
    return float(np.percentile(np.asarray(values, dtype=np.float64), q))


def _candidate_thresholds(scores: "np.ndarray") -> "np.ndarray":
    """Midpoints between every pair of consecutive sorted unique scores,
    plus one candidate below the lowest score and one above the highest --
    so a fully separated pair of score groups always has a candidate
    strictly between them, never only at a group's own boundary value."""
    unique = np.unique(scores)
    if unique.size == 1:
        return np.array([unique[0] - 1e-9, unique[0] + 1e-9])
    midpoints = (unique[:-1] + unique[1:]) / 2.0
    return np.concatenate([[unique[0] - 1e-9], midpoints, [unique[-1] + 1e-9]])


def equal_error_rate(genuine: Sequence[float], impostor: Sequence[float]) -> "tuple[float, float]":
    """The equal error rate and its threshold, from a sweep over candidate
    thresholds between every observed score. False accept is the share of
    `impostor` scores at or above the threshold; false reject is the share
    of `genuine` scores below it. Returns `(eer, threshold)` at the
    candidate minimizing `|far - frr|`; `eer` there is `(far + frr) / 2`.

    On two fully separated groups, the winning threshold falls strictly
    between them (a real midpoint, not either group's own edge value), and
    `eer` is `0.0`. Raises `ValueError` if either sequence is empty.
    """
    if not genuine or not impostor:
        raise ValueError("equal_error_rate: both genuine and impostor need at least one score")
    genuine_arr = np.asarray(genuine, dtype=np.float64)
    impostor_arr = np.asarray(impostor, dtype=np.float64)

    best_diff = None
    best_eer = 0.0
    best_threshold = 0.0
    for t in _candidate_thresholds(np.concatenate([genuine_arr, impostor_arr])):
        far = float(np.mean(impostor_arr >= t))
        frr = float(np.mean(genuine_arr < t))
        diff = abs(far - frr)
        eer = (far + frr) / 2.0
        if best_diff is None or diff < best_diff or (diff == best_diff and eer < best_eer):
            best_diff, best_eer, best_threshold = diff, eer, float(t)
    return best_eer, best_threshold


def threshold_at_far(impostor: Sequence[float], far: float) -> float:
    """The lowest threshold whose false-accept share (the fraction of
    `impostor` scores at or above it) is at or below `far`. Raises
    `ValueError` for an empty `impostor`."""
    if not impostor:
        raise ValueError("threshold_at_far: impostor needs at least one score")
    impostor_arr = np.asarray(impostor, dtype=np.float64)
    candidates = sorted(_candidate_thresholds(impostor_arr).tolist())
    for t in candidates:
        if float(np.mean(impostor_arr >= t)) <= far:
            return t
    return candidates[-1]
