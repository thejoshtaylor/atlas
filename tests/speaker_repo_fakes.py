"""`FakeSpeakerRepository` -- an in-memory `SpeakerRepository`
(`atlas.db.speaker_repository`), the Postgres-free double
`tests/test_speakers_route.py` drives, matching `tests/edge_fakes.py`'s
own `FakeEdgeDeviceRepository` precedent.

Raises the same error type on a duplicate `display_name` that the real
`PostgresSpeakerRepository` would (`sqlalchemy.exc.IntegrityError`), so a
route test against this fake exercises the exact same except-clause the
route runs against real Postgres. Cascades embeddings on delete, matching
`speaker_embeddings.speaker_id`'s real `ondelete="CASCADE"` foreign key.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy.exc import IntegrityError

from atlas.db.speaker_repository import ReferenceEmbedding, Speaker


class FakeSpeakerRepository:
    """Structurally satisfies `SpeakerRepository` without inheriting from
    it, matching every other `Fake*Repository` in this project."""

    def __init__(self) -> None:
        self._speakers: dict[int, Speaker] = {}
        self._next_id = 1
        # Keyed by (speaker_id, phrase_index, model_id) -- the same triple
        # the real `uq_speaker_embeddings_speaker_phrase_model` constraint
        # covers.
        self._embeddings: "dict[tuple[int, int, str], ReferenceEmbedding]" = {}

    async def create_speaker(
        self, *, display_name: str, linked_user_id: "int | None", created_at: datetime
    ) -> Speaker:
        if any(s.display_name == display_name for s in self._speakers.values()):
            raise IntegrityError(
                "INSERT INTO speakers ...", {}, Exception("uq_speakers_display_name")
            )
        speaker = Speaker(
            id=self._next_id,
            display_name=display_name,
            linked_user_id=linked_user_id,
            created_at=created_at,
        )
        self._speakers[speaker.id] = speaker
        self._next_id += 1
        return speaker

    async def list_speakers(self) -> "list[Speaker]":
        return sorted(self._speakers.values(), key=lambda s: (s.created_at, s.id))

    async def get_speaker(self, speaker_id: int) -> "Speaker | None":
        return self._speakers.get(speaker_id)

    async def delete_speaker(self, speaker_id: int) -> bool:
        if speaker_id not in self._speakers:
            return False
        del self._speakers[speaker_id]
        for key in [k for k in self._embeddings if k[0] == speaker_id]:
            del self._embeddings[key]
        return True

    async def upsert_embedding(
        self,
        *,
        speaker_id: int,
        phrase_index: int,
        model_id: str,
        vector: "list[float]",
        created_at: datetime,
    ) -> None:
        speaker = self._speakers[speaker_id]
        key = (speaker_id, phrase_index, model_id)
        self._embeddings[key] = ReferenceEmbedding(
            speaker_id=speaker_id,
            display_name=speaker.display_name,
            phrase_index=phrase_index,
            model_id=model_id,
            vector=list(vector),
        )

    async def delete_embedding(self, *, speaker_id: int, phrase_index: int) -> int:
        keys = [k for k in self._embeddings if k[0] == speaker_id and k[1] == phrase_index]
        for key in keys:
            del self._embeddings[key]
        return len(keys)

    async def list_reference_embeddings(self, model_id: str) -> "list[ReferenceEmbedding]":
        return [e for e in self._embeddings.values() if e.model_id == model_id]

    async def count_embeddings(self, model_id: str) -> "dict[int, int]":
        counts: dict[int, int] = {}
        for (speaker_id, _phrase_index, embedding_model_id), _embedding in self._embeddings.items():
            if embedding_model_id == model_id:
                counts[speaker_id] = counts.get(speaker_id, 0) + 1
        return counts


def fake_speaker(
    *, speaker_id: int = 1, display_name: str = "Test Member", linked_user_id: "int | None" = None
) -> Speaker:
    """One `Speaker` shaped for a test, matching `edge_fakes.fake_edge_device`'s
    own convenience-constructor precedent."""
    return Speaker(
        id=speaker_id,
        display_name=display_name,
        linked_user_id=linked_user_id,
        created_at=datetime.now(timezone.utc),
    )
