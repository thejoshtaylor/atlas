"""timers -- one table for voice timers and alarms

Revision ID: 0020
Revises: 0019
Create Date: 2026-09-30

Quick task 260930-06x: the `timers` table holds both kinds of entry. A timer
is a countdown with a due time, or a paused remainder. An alarm is an HH:MM
wall-clock time in server.timezone with an optional weekday repeat mask. The
poller in `atlas/timers/scheduler.py` rings an entry when `due_at` passes.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0020"
down_revision: Union[str, Sequence[str], None] = "0019"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "timers",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("kind", sa.Text(), nullable=False),
        sa.Column("label", sa.Text(), nullable=False, server_default=""),
        sa.Column("due_at", sa.DateTime(), nullable=True),
        sa.Column("remaining_s", sa.Integer(), nullable=True),
        sa.Column("duration_s", sa.Integer(), nullable=True),
        sa.Column("time_of_day", sa.Text(), nullable=True),
        sa.Column("repeat_days", sa.SmallInteger(), nullable=False, server_default="0"),
        sa.Column("enabled", sa.Boolean(), nullable=False, server_default=sa.true()),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.CheckConstraint("kind IN ('timer', 'alarm')", name="ck_timers_kind"),
    )


def downgrade() -> None:
    op.drop_table("timers")
