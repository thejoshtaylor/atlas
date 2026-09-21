"""wake_events table -- every wake hit the detector reports, allowed or blocked

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-21

D-13, D-14: the persistent wake-hit record this phase must build, because
it does not exist yet. `sources/runner.py::_record_blocked_hit` today
emits a structured log line and nothing else, and no session directory is
ever created for a hit the gate blocks (`SessionRecorder` is only
constructed after a hit is allowed). This table is what the tuning
screen's DBG-05 re-partitions arithmetic against instead of re-running an
engine over recorded audio.

Unlike `0011_provider_selection.py`, this migration seeds nothing: this is
new, empty, append-only state, not a cutover with legacy data to carry.
No unique constraint -- every row is a distinct event, not a per-key
singleton. Indexed on `recorded_at`, the column the newest-first read
(`list_wake_events`) orders by.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0012"
down_revision: Union[str, Sequence[str], None] = "0011"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX = "ix_wake_events_recorded_at"


def upgrade() -> None:
    op.create_table(
        "wake_events",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("source", sa.Text(), nullable=False),
        sa.Column("engine", sa.Text(), nullable=False),
        sa.Column("score", sa.Float(), nullable=True),
        sa.Column("allowed", sa.Boolean(), nullable=False),
        sa.Column("block_reason", sa.Text(), nullable=True),
        sa.Column("recorded_at", sa.DateTime(), nullable=False),
    )
    op.create_index(_INDEX, "wake_events", ["recorded_at"])


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="wake_events")
    op.drop_table("wake_events")
