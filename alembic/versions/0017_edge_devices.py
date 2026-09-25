"""edge_devices table -- one row per paired Raspberry Pi (D-03, Phase 10)

Revision ID: 0017
Revises: 0016
Create Date: 2026-09-25

Plan 10-04: the token store every deployed Pi's device token hash lives
in. `token_hash` is a SHA-256 hash of the bearer token an admin creates
through `POST /api/edge-devices` -- the plaintext is returned once, in
that response, and this table never stores it. `uq_edge_devices_token_hash`
backs `EdgeDeviceRepository.get_active_by_token_hash`'s own lookup and
also refuses two devices from ever sharing a hash at the database level,
not only in application code.

Like `0012_wake_events.py`, this migration seeds nothing: this is new,
empty, append-only state. `revoked_at` is nullable, never a `DELETE`
column -- a revoked device keeps its row, the same "disable, never
delete" discipline `users.disabled_at` already establishes, so audit
history survives a revoke.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0017"
down_revision: Union[str, Sequence[str], None] = "0016"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_UNIQUE = "uq_edge_devices_token_hash"


def upgrade() -> None:
    op.create_table(
        "edge_devices",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column(
            "created_by_user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False
        ),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("last_connected_at", sa.DateTime(), nullable=True),
        sa.UniqueConstraint("token_hash", name=_UNIQUE),
    )


def downgrade() -> None:
    op.drop_table("edge_devices")
