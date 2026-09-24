"""provider_credentials

Revision ID: 0003
Revises: 0002
Create Date: 2026-09-18

Creates `provider_credentials` (PROV-04, D-07): one row per credential
slot in the closed set `atlas.crypto.credentials.CredentialSlot`
names, holding ciphertext only -- `atlas.crypto.credentials` is the
one module that ever turns a row here back into a usable value, and only
at startup (`app.py`'s `lifespan`), never from a route.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0003"
down_revision: Union[str, Sequence[str], None] = "0002"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.create_table(
        "provider_credentials",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slot", sa.Text(), nullable=False),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=False),
        sa.Column("key_version", sa.Integer(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("updated_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slot", name="uq_provider_credentials_slot"),
        sa.ForeignKeyConstraint(
            ["updated_by_user_id"],
            ["users.id"],
            name="fk_provider_credentials_updated_by_user_id",
        ),
    )


def downgrade() -> None:
    op.drop_table("provider_credentials")
