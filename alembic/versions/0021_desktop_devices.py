"""desktop_devices -- one table for paired Macs

Revision ID: 0021
Revises: 0020
Create Date: 2026-10-01

Phase 14, plan 14-04. D-01: a Mac is a separate table from `edge_devices`, so
a Mac token can never open `/ws/edge`. The token is stored as a SHA-256 hash,
and a Mac is revoked by `revoked_at`, never deleted. D-06: `edge_device_id`
maps a Mac to a room (the edge device that room's microphone belongs to).

D-14: at most one default Mac. The partial unique index
`uq_desktop_devices_single_default` makes the database refuse a second row
with `is_default` true, so a bug in application code cannot create two
defaults. D-28: active Mac names are unique, case-insensitive. The partial
unique index `uq_desktop_devices_active_name` covers `lower(name)` for rows
with `revoked_at IS NULL`, so a revoked Mac frees its name.

Unlike 0016, there is no data cleanup before the indexes: this table is new
and starts empty.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0021"
down_revision: Union[str, Sequence[str], None] = "0020"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_DEFAULT_INDEX = "uq_desktop_devices_single_default"
_NAME_INDEX = "uq_desktop_devices_active_name"


def upgrade() -> None:
    op.create_table(
        "desktop_devices",
        sa.Column("id", sa.Integer(), primary_key=True),
        sa.Column("name", sa.Text(), nullable=False),
        sa.Column("token_hash", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column(
            "created_by_user_id",
            sa.Integer(),
            sa.ForeignKey("users.id"),
            nullable=False,
        ),
        sa.Column("revoked_at", sa.DateTime(), nullable=True),
        sa.Column("last_seen_at", sa.DateTime(), nullable=True),
        sa.Column(
            "edge_device_id",
            sa.Integer(),
            sa.ForeignKey("edge_devices.id"),
            nullable=True,
        ),
        sa.Column("is_default", sa.Boolean(), nullable=False, server_default=sa.false()),
        sa.UniqueConstraint("token_hash", name="uq_desktop_devices_token_hash"),
    )
    op.create_index(
        _DEFAULT_INDEX,
        "desktop_devices",
        ["is_default"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )
    op.create_index(
        _NAME_INDEX,
        "desktop_devices",
        [sa.text("lower(name)")],
        unique=True,
        postgresql_where=sa.text("revoked_at IS NULL"),
    )


def downgrade() -> None:
    op.drop_index(_NAME_INDEX, table_name="desktop_devices")
    op.drop_index(_DEFAULT_INDEX, table_name="desktop_devices")
    op.drop_table("desktop_devices")
