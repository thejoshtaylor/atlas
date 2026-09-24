"""setup state

Revision ID: 0004
Revises: 0003
Create Date: 2026-09-18

Creates the three tables the first-run wizard's own server-side progress
needs (WEB-01, WEB-02, WEB-03, D-08): `setup_state` (a single row recording
whether the wizard has ever been finished), `setup_steps` (one row per
wizard step, seeded here so a fresh install reads every step as present and
incomplete rather than finding an empty table that reads as "nothing to
do"), and `settings` (the general operator-editable settings store the
wizard's own audio-source choice is the first, but not the only, writer
of).

Unlike `0001_policy_tables.py`, nothing here is seeded from an operator's
existing configuration file -- there is no prior "wizard progress" to carry
forward, because no wizard existed before this phase. `setup_steps` is
seeded with five named rows, each `completed_at IS NULL`; `setup_state` is
seeded with its one row, also `completed_at IS NULL`; `settings` starts
empty, exactly like `0002_accounts.py`'s `users` table starts empty.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0004"
down_revision: Union[str, Sequence[str], None] = "0003"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# The five step names `atlas.routes.wizard` enumerates -- named here,
# not imported from that module, because a migration must keep reading the
# same way years after the application code around it has changed; a
# migration that imports application code can silently start seeding a
# different set of rows the day that code is edited (03-PATTERNS.md's own
# "a migration is a historical record" convention, `0001`'s seed step sets
# the precedent).
_STEP_NAMES = ("admin_account", "hub", "provider_set", "audio_source", "room")


def upgrade() -> None:
    _create_tables()
    _seed_rows()


def _create_tables() -> None:
    op.create_table(
        "setup_state",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_table(
        "setup_steps",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("completed_at", sa.DateTime(), nullable=True),
        sa.Column("detail", sa.JSON(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("name", name="uq_setup_steps_name"),
    )
    op.create_table(
        "settings",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("value", sa.JSON(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("updated_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("key", name="uq_settings_key"),
        sa.ForeignKeyConstraint(
            ["updated_by_user_id"], ["users.id"], name="fk_settings_updated_by_user_id"
        ),
    )


def _seed_rows() -> None:
    conn = op.get_bind()
    metadata = sa.MetaData()
    setup_state = sa.Table("setup_state", metadata, autoload_with=conn)
    setup_steps = sa.Table("setup_steps", metadata, autoload_with=conn)

    conn.execute(setup_state.insert().values(id=1, completed_at=None))
    conn.execute(
        setup_steps.insert(),
        [{"name": name, "completed_at": None, "detail": None} for name in _STEP_NAMES],
    )


def downgrade() -> None:
    op.drop_table("settings")
    op.drop_table("setup_steps")
    op.drop_table("setup_state")
