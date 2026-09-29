"""`Speaker`, `ReferenceEmbedding`, and the `SpeakerRepository` Protocol
(D-01, D-03, D-04, Phase 11).

Plan 11-03 builds the shape and the Postgres implementation
(`db/speaker_postgres.py`) together -- unlike `EdgeDeviceRepository`'s
Protocol-then-storage split, this repository has no earlier plan that
needed the shape alone.

There is no `revoke_*`/`disable_*` method here, on purpose: `delete_speaker`
is a real `DELETE`, not a flag flip. A speaker embedding is biometric data,
and D-03 requires it gone for good when a member is removed -- the
opposite of `EdgeDeviceRepository.revoke_device`'s "disable, never delete"
shape.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class Speaker:
    """One household member (D-01) -- never carries an embedding vector
    itself; that lives only in `ReferenceEmbedding` below."""

    id: int
    display_name: str
    linked_user_id: int | None
    created_at: datetime


@dataclass(frozen=True)
class ReferenceEmbedding:
    """One enrollment-phrase embedding for one member, under one model id
    (D-03) -- joined with the member's `display_name` so a caller (the
    `vad_end` matching gate, plan 11-04) never needs a second lookup per
    row."""

    speaker_id: int
    display_name: str
    phrase_index: int
    model_id: str
    vector: "list[float]"


class SpeakerRepository(Protocol):
    """What the admin routes (`routes/speakers.py`) and the live matching
    gate (plan 11-04, D-06) need. `delete_speaker` is a hard delete (D-03)
    -- no revoke/disable method exists on this Protocol."""

    async def create_speaker(
        self, *, display_name: str, linked_user_id: int | None, created_at: datetime
    ) -> Speaker: ...

    async def list_speakers(self) -> "list[Speaker]": ...

    async def get_speaker(self, speaker_id: int) -> Speaker | None: ...

    async def delete_speaker(self, speaker_id: int) -> bool:
        """One `DELETE`, its row count decides the result -- the foreign
        key cascade on `speaker_embeddings.speaker_id` removes every one of
        this member's embedding rows along with it (D-03). No `DELETE`
        column ever gates this; there is nothing to flip back."""
        ...

    async def upsert_embedding(
        self,
        *,
        speaker_id: int,
        phrase_index: int,
        model_id: str,
        vector: "list[float]",
        created_at: datetime,
    ) -> None:
        """Replaces the row for the same `(speaker_id, phrase_index,
        model_id)` -- re-enrolling one phrase, or re-embedding under a new
        model id, never leaves a stale duplicate row behind."""
        ...

    async def delete_embedding(self, *, speaker_id: int, phrase_index: int) -> int:
        """Remove the row for this phrase under every `model_id`, because
        the clip is gone. Return the number of rows removed."""
        ...

    async def list_reference_embeddings(self, model_id: str) -> "list[ReferenceEmbedding]":
        """Every embedding stored under `model_id`, across every member --
        the input to `speaker_id.matching.ReferenceSet.load` (plan 11-02)."""
        ...

    async def count_embeddings(self, model_id: str) -> "dict[int, int]":
        """`speaker_id -> number of phrases enrolled` under `model_id` --
        the enrollment-progress read plan 11-07's admin screen and plan
        11-04's gate (D-10, zero-enrolled degrades `enforce` to `record`)
        both need."""
        ...
