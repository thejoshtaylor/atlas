"""Cosine matching with a runner-up margin (D-06).

Each member's reference is the L2-normalized mean of that member's
L2-normalized phrase embeddings. A turn's embedding is the L2-normalized,
speech-weighted mean of its window embeddings (`mean_embedding`, called by
the live pipeline, not by this module). The match score is plain cosine
similarity; the margin is the best score minus the second-best member's
score.

`sherpa_onnx.SpeakerEmbeddingManager.search()` is not used here -- it
returns a match or nothing, and hides the runner-up score D-06 needs for
the margin (11-RESEARCH.md Pitfall 1). This module computes both directly
against Postgres-loaded reference embeddings instead; at household scale
(a handful of members, a handful of phrases each) that is a handful of dot
products, nowhere near a cost worth hiding behind a vector-search library.

This module is pure: no I/O, no imports from `turn/`, `providers/`, `db/`,
or `transports/`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


def l2_normalize(vector: "np.ndarray | Sequence[float]") -> "np.ndarray":
    """Return `vector` scaled to unit L2 norm, as `float32`.

    Raises `ValueError` for a vector that is all-zero (nothing to scale
    toward) or that carries a NaN/inf component (a corrupted embedding
    must never silently become a direction).
    """
    arr = np.asarray(vector, dtype=np.float64)
    if not np.all(np.isfinite(arr)):
        raise ValueError("l2_normalize: vector contains a non-finite value")
    norm = float(np.linalg.norm(arr))
    if norm == 0.0:
        raise ValueError("l2_normalize: vector has zero norm")
    return (arr / norm).astype(np.float32)


def mean_embedding(
    vectors: "Sequence[np.ndarray | Sequence[float]]",
    weights: "Sequence[float] | None" = None,
) -> "np.ndarray":
    """Return the L2-normalized, optionally weighted, mean of `vectors`.

    Raises `ValueError` for an empty `vectors`, a `weights` of the wrong
    length, or a non-positive total weight.
    """
    arr = np.asarray(vectors, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[0] == 0:
        raise ValueError("mean_embedding: at least one vector is required")
    if weights is None:
        weight_arr = np.ones(arr.shape[0], dtype=np.float64)
    else:
        weight_arr = np.asarray(weights, dtype=np.float64)
        if weight_arr.shape[0] != arr.shape[0]:
            raise ValueError("mean_embedding: weights length must match vectors length")
    total_weight = float(weight_arr.sum())
    if total_weight <= 0.0:
        raise ValueError("mean_embedding: total weight must be positive")
    mean = (arr * weight_arr[:, None]).sum(axis=0) / total_weight
    return l2_normalize(mean)


@dataclass(frozen=True)
class MatchResult:
    """The outcome of scoring one turn embedding against a `ReferenceSet`.

    An empty reference set (`enrolled_count == 0`) returns every field
    `None` and `scores` empty -- there is nothing to match against, not a
    zero score.
    """

    best_speaker_id: "int | None"
    best_name: "str | None"
    best_score: "float | None"
    second_score: "float | None"
    margin: "float | None"
    scores: "dict[int, float]"


class ReferenceSet:
    """Enrolled members' reference embeddings, and the cosine match against
    them (D-06). One global threshold decides identified/unknown elsewhere
    (`speaker_id/gate.py`) -- this class only produces the score and the
    margin, never a threshold decision.
    """

    def __init__(self) -> None:
        self._names: "dict[int, str]" = {}
        self._references: "dict[int, np.ndarray]" = {}
        self._dim: "int | None" = None

    @classmethod
    def load(cls, rows: "Sequence[dict]") -> "ReferenceSet":
        """Build a `ReferenceSet` from `rows`, each carrying `speaker_id`,
        `display_name`, and `vector` (one row per stored phrase embedding).
        Rows for the same `speaker_id` are grouped before their reference
        is computed."""
        grouped: "dict[int, dict]" = {}
        for row in rows:
            speaker_id = row["speaker_id"]
            entry = grouped.setdefault(speaker_id, {"display_name": row["display_name"], "vectors": []})
            entry["vectors"].append(row["vector"])
        reference_set = cls()
        for speaker_id, entry in grouped.items():
            reference_set.upsert_speaker(speaker_id, entry["display_name"], entry["vectors"])
        return reference_set

    def upsert_speaker(
        self,
        speaker_id: int,
        display_name: str,
        vectors: "Sequence[np.ndarray | Sequence[float]]",
    ) -> None:
        """Replace `speaker_id`'s reference with the L2-normalized mean of
        `vectors` (each L2-normalized first, per D-06) -- this always
        replaces, it never appends to a member's existing vectors.

        Raises `ValueError` for an empty `vectors`, or a dimension that
        does not match every other enrolled member's dimension.
        """
        normalized = [l2_normalize(v) for v in vectors]
        if not normalized:
            raise ValueError("upsert_speaker: at least one vector is required")
        dim = normalized[0].shape[0]
        if self._dim is not None and dim != self._dim:
            raise ValueError(f"upsert_speaker: dimension {dim} does not match existing dimension {self._dim}")
        self._references[speaker_id] = mean_embedding(normalized)
        self._names[speaker_id] = display_name
        self._dim = dim

    def remove_speaker(self, speaker_id: int) -> None:
        """Drop `speaker_id`'s reference entirely. A no-op if it was never
        enrolled."""
        self._references.pop(speaker_id, None)
        self._names.pop(speaker_id, None)
        if not self._references:
            self._dim = None

    @property
    def enrolled_count(self) -> int:
        return len(self._references)

    def name_for(self, speaker_id: int) -> "str | None":
        return self._names.get(speaker_id)

    def match(self, embedding: "np.ndarray | Sequence[float]") -> MatchResult:
        """Score `embedding` against every enrolled member.

        Raises `ValueError` for a non-finite `embedding` or one whose
        dimension does not match the enrolled references.
        """
        if not self._references:
            return MatchResult(None, None, None, None, None, {})
        normalized = l2_normalize(embedding)
        if self._dim is not None and normalized.shape[0] != self._dim:
            raise ValueError(
                f"match: embedding dimension {normalized.shape[0]} does not match reference dimension {self._dim}"
            )
        scores = {speaker_id: float(np.dot(normalized, reference)) for speaker_id, reference in self._references.items()}
        ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
        best_id, best_score = ranked[0]
        second_score = ranked[1][1] if len(ranked) > 1 else None
        margin = (best_score - second_score) if second_score is not None else None
        return MatchResult(best_id, self._names[best_id], best_score, second_score, margin, scores)
