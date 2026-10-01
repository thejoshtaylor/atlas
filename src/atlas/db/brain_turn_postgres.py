"""`PostgresBrainTurnRepository`: `BrainTurnRepository` over a real Postgres.

It reuses the naive-UTC boundary helpers of `db/postgres.py`. The insert goes
through the ORM, so every value is a bound parameter. No SQL text is built
from a transcript or an argument (D-18).
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import async_sessionmaker

from atlas.db.brain_turn_models import BrainTurnRow
from atlas.db.brain_turn_repository import BrainTurn, BrainTurnEntry
from atlas.db.postgres import _to_aware_utc, _to_naive_utc


def _from_row(row: BrainTurnRow) -> BrainTurn:
    return BrainTurn(
        id=row.id,
        turn_id=row.turn_id,
        created_at=_to_aware_utc(row.created_at),
        transcript=row.transcript,
        normalized_transcript=row.normalized_transcript,
        transcript_fingerprint=row.transcript_fingerprint,
        continuation=row.continuation,
        tool_calls=tuple(row.tool_calls or ()),
        reply_text=row.reply_text,
        tier_index=row.tier_index,
        tier_model=row.tier_model,
        brain_latency_ms=row.brain_latency_ms,
        outcome=row.outcome,
    )


class PostgresBrainTurnRepository:
    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def record_brain_turn(self, entry: BrainTurnEntry) -> None:
        row = BrainTurnRow(
            turn_id=entry.turn_id,
            created_at=_to_naive_utc(entry.created_at),
            transcript=entry.transcript,
            normalized_transcript=entry.normalized_transcript,
            transcript_fingerprint=entry.transcript_fingerprint,
            continuation=entry.continuation,
            tool_calls=list(entry.tool_calls),
            reply_text=entry.reply_text,
            tier_index=entry.tier_index,
            tier_model=entry.tier_model,
            brain_latency_ms=entry.brain_latency_ms,
            outcome=entry.outcome,
        )
        async with self._sessionmaker() as session:
            session.add(row)
            await session.commit()

    async def list_brain_turns(self, limit: "int | None" = None) -> "list[BrainTurn]":
        statement = select(BrainTurnRow).order_by(BrainTurnRow.created_at.desc(), BrainTurnRow.id.desc())
        if limit is not None:
            statement = statement.limit(limit)
        async with self._sessionmaker() as session:
            rows = (await session.execute(statement)).scalars().all()
            return [_from_row(row) for row in rows]

    async def delete_brain_turns_before(self, cutoff: datetime) -> int:
        """One `DELETE`, returning the row count. `_to_naive_utc` on the way
        in, because the column is naive UTC."""
        statement = delete(BrainTurnRow).where(BrainTurnRow.created_at < _to_naive_utc(cutoff))
        async with self._sessionmaker() as session:
            result = await session.execute(statement)
            await session.commit()
            return int(result.rowcount or 0)
