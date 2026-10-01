"""brain_turns -- the brain-turn intent log

Revision ID: 0022
Revises: 0021
Create Date: 2026-10-01

Quick task 261001-mp8 (D-14 to D-16): one row for every turn that reached the
brain tier race. It records what was said (raw transcript, wake-stripped
normalized transcript, and its sha256 fingerprint) and what the brain meant
(the ordered tool calls, each read or write). The table is new and empty.
Nothing is seeded.

The log records only. No turn reads it back, so it dispatches and matches
nothing. `RetentionScheduler` expires the rows on `debug.retain_days`
(D-19), because they hold transcripts derived from household audio.

`tool_calls` is JSONB because each call carries its own arguments object, and
a later tool wants to filter on tool name and argument keys without a join
table.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision: str = "0022"
down_revision: Union[str, Sequence[str], None] = "0021"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "brain_turns",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("turn_id", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("transcript", sa.Text(), nullable=False),
        sa.Column("normalized_transcript", sa.Text(), nullable=False),
        sa.Column("transcript_fingerprint", sa.Text(), nullable=False),
        sa.Column("continuation", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.Column(
            "tool_calls",
            postgresql.JSONB(),
            nullable=False,
            server_default=sa.text("'[]'::jsonb"),
        ),
        sa.Column("reply_text", sa.Text(), nullable=True),
        sa.Column("tier_index", sa.Integer(), nullable=True),
        sa.Column("tier_model", sa.Text(), nullable=True),
        sa.Column("brain_latency_ms", sa.Integer(), nullable=True),
        sa.Column("outcome", sa.Text(), nullable=False),
    )
    op.create_index("ix_brain_turns_created_at", "brain_turns", ["created_at"])
    op.create_index("ix_brain_turns_fingerprint", "brain_turns", ["transcript_fingerprint"])


def downgrade() -> None:
    op.drop_index("ix_brain_turns_fingerprint", table_name="brain_turns")
    op.drop_index("ix_brain_turns_created_at", table_name="brain_turns")
    op.drop_table("brain_turns")
