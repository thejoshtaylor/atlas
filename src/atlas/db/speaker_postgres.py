"""`SpeakerRepository`, implemented against a real Postgres (D-01, D-03,
Phase 11, plan 11-03).

Structurally satisfies `atlas.db.speaker_repository.SpeakerRepository` (a
`typing.Protocol`) -- built from `PostgresEdgeDeviceRepository`'s own
shape (`db/edge_postgres.py`), reusing `db/postgres.py`'s
`_to_naive_utc`/`_to_aware_utc` naive-UTC boundary convention rather than
re-deriving it.

`upsert_embedding` uses a real `INSERT ... ON CONFLICT ON CONSTRAINT ...
DO UPDATE`, the same one-statement-not-two discipline
`PostgresProviderSelectionRepository.set_selection` already uses (WR-11):
a read-then-write shape here would let two concurrent enrollment captures
for the same phrase both see no row and both insert, and the unique
constraint would turn the second into an uncaught `IntegrityError`.
"""

from __future__ import annotations

import math
from datetime import datetime

from sqlalchemy import delete, func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import async_sessionmaker

from atlas.db.postgres import _to_aware_utc, _to_naive_utc
from atlas.db.speaker_models import SpeakerEmbeddingRow, SpeakerRow
from atlas.db.speaker_repository import ReferenceEmbedding, Speaker

_UQ_SPEAKER_EMBEDDINGS_SPEAKER_PHRASE_MODEL = "uq_speaker_embeddings_speaker_phrase_model"

# A household holds a handful of members with a handful of phrases each --
# 1024 floats is generously larger than any embedding model this project
# fetches (`scripts/fetch_models.py`'s largest dim is 512, CAM++). A vector
# past this length, or one that is empty or carries a non-finite value, is
# refused before it ever reaches a write -- never silently truncated or
# stored as `NaN`/`Infinity`, which Postgres's own `double precision[]`
# would otherwise happily accept.
_MAX_VECTOR_LENGTH = 1024


def _validate_vector(vector: "list[float]") -> None:
    if not vector:
        raise ValueError("vector must not be empty")
    if len(vector) > _MAX_VECTOR_LENGTH:
        raise ValueError(f"vector must not exceed {_MAX_VECTOR_LENGTH} values, got {len(vector)}")
    if not all(math.isfinite(value) for value in vector):
        raise ValueError("vector must not contain a non-finite value")


def _speaker_from_row(row: SpeakerRow) -> Speaker:
    return Speaker(
        id=row.id,
        display_name=row.display_name,
        linked_user_id=row.linked_user_id,
        created_at=_to_aware_utc(row.created_at),
        can_control_home=row.can_control_home,
    )


class PostgresSpeakerRepository:
    """`SpeakerRepository`, implemented against a real Postgres.

    There is no base class to inherit from, matching every
    `Postgres*Repository` class in `db/postgres.py`/`db/edge_postgres.py`.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def create_speaker(
        self, *, display_name: str, linked_user_id: int | None, created_at: datetime
    ) -> Speaker:
        async with self._sessionmaker() as session:
            row = SpeakerRow(
                display_name=display_name,
                linked_user_id=linked_user_id,
                created_at=_to_naive_utc(created_at),
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _speaker_from_row(row)

    async def list_speakers(self) -> "list[Speaker]":
        async with self._sessionmaker() as session:
            rows = (
                (await session.execute(select(SpeakerRow).order_by(SpeakerRow.created_at, SpeakerRow.id)))
                .scalars()
                .all()
            )
            return [_speaker_from_row(row) for row in rows]

    async def get_speaker(self, speaker_id: int) -> "Speaker | None":
        async with self._sessionmaker() as session:
            row = (
                await session.execute(select(SpeakerRow).where(SpeakerRow.id == speaker_id))
            ).scalar_one_or_none()
            return None if row is None else _speaker_from_row(row)

    async def set_can_control_home(self, speaker_id: int, can_control_home: bool) -> "Speaker | None":
        """One `UPDATE ... RETURNING` on the member row. `None` when no row
        matches."""
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    update(SpeakerRow)
                    .where(SpeakerRow.id == speaker_id)
                    .values(can_control_home=can_control_home)
                    .returning(SpeakerRow)
                )
            ).scalar_one_or_none()
            await session.commit()
            return None if row is None else _speaker_from_row(row)

    async def delete_speaker(self, speaker_id: int) -> bool:
        """One `DELETE ... WHERE id = :id` -- its row count decides the
        result, matching `PostgresEdgeDeviceRepository.revoke_device`'s own
        rowcount-decides-truth shape. The foreign key `ondelete="CASCADE"`
        on `speaker_embeddings.speaker_id` removes every one of this
        member's embedding rows in the same statement, at the database
        level -- not a second, application-issued `DELETE` this method
        would have to run itself."""
        async with self._sessionmaker() as session:
            result = await session.execute(delete(SpeakerRow).where(SpeakerRow.id == speaker_id))
            await session.commit()
            return bool(result.rowcount == 1)

    async def upsert_embedding(
        self,
        *,
        speaker_id: int,
        phrase_index: int,
        model_id: str,
        vector: "list[float]",
        created_at: datetime,
    ) -> None:
        _validate_vector(vector)
        naive_created_at = _to_naive_utc(created_at)
        statement = (
            pg_insert(SpeakerEmbeddingRow)
            .values(
                speaker_id=speaker_id,
                phrase_index=phrase_index,
                model_id=model_id,
                vector=vector,
                created_at=naive_created_at,
            )
            .on_conflict_do_update(
                constraint=_UQ_SPEAKER_EMBEDDINGS_SPEAKER_PHRASE_MODEL,
                set_=dict(vector=vector, created_at=naive_created_at),
            )
        )
        async with self._sessionmaker() as session:
            await session.execute(statement)
            await session.commit()

    async def delete_embedding(self, *, speaker_id: int, phrase_index: int) -> int:
        """One `DELETE` for the phrase under every model id. Return its row count."""
        async with self._sessionmaker() as session:
            result = await session.execute(
                delete(SpeakerEmbeddingRow).where(
                    SpeakerEmbeddingRow.speaker_id == speaker_id,
                    SpeakerEmbeddingRow.phrase_index == phrase_index,
                )
            )
            await session.commit()
            return int(result.rowcount)

    async def list_reference_embeddings(self, model_id: str) -> "list[ReferenceEmbedding]":
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(
                    select(
                        SpeakerEmbeddingRow.speaker_id,
                        SpeakerRow.display_name,
                        SpeakerEmbeddingRow.phrase_index,
                        SpeakerEmbeddingRow.model_id,
                        SpeakerEmbeddingRow.vector,
                    )
                    .join(SpeakerRow, SpeakerRow.id == SpeakerEmbeddingRow.speaker_id)
                    .where(SpeakerEmbeddingRow.model_id == model_id)
                    .order_by(SpeakerEmbeddingRow.speaker_id, SpeakerEmbeddingRow.phrase_index)
                )
            ).all()
            return [
                ReferenceEmbedding(
                    speaker_id=row.speaker_id,
                    display_name=row.display_name,
                    phrase_index=row.phrase_index,
                    model_id=row.model_id,
                    vector=list(row.vector),
                )
                for row in rows
            ]

    async def count_embeddings(self, model_id: str) -> "dict[int, int]":
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(
                    select(SpeakerEmbeddingRow.speaker_id, func.count(SpeakerEmbeddingRow.id))
                    .where(SpeakerEmbeddingRow.model_id == model_id)
                    .group_by(SpeakerEmbeddingRow.speaker_id)
                )
            ).all()
            return {speaker_id: count for speaker_id, count in rows}
