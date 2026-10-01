"""`BrainTurnRow`: the `brain_turns` table that migration 0022 creates.

`alembic/env.py` imports this module so `Base.metadata` sees the table.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import Boolean, Index, Text, text
from sqlalchemy.dialects import postgresql
from sqlalchemy.orm import Mapped, mapped_column

from atlas.db.models import Base


class BrainTurnRow(Base):
    """One turn that reached the brain tier race. Datetimes are naive UTC,
    the project's database convention (`db/postgres.py`)."""

    __tablename__ = "brain_turns"
    __table_args__ = (
        Index("ix_brain_turns_created_at", "created_at"),
        Index("ix_brain_turns_fingerprint", "transcript_fingerprint"),
    )

    id: Mapped[int] = mapped_column(primary_key=True)
    turn_id: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(nullable=False)
    transcript: Mapped[str] = mapped_column(Text, nullable=False)
    normalized_transcript: Mapped[str] = mapped_column(Text, nullable=False)
    transcript_fingerprint: Mapped[str] = mapped_column(Text, nullable=False)
    continuation: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    tool_calls: Mapped[list] = mapped_column(
        postgresql.JSONB, nullable=False, server_default=text("'[]'::jsonb")
    )
    reply_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    tier_index: Mapped[int | None] = mapped_column(nullable=True)
    tier_model: Mapped[str | None] = mapped_column(Text, nullable=True)
    brain_latency_ms: Mapped[int | None] = mapped_column(nullable=True)
    outcome: Mapped[str] = mapped_column(Text, nullable=False)
