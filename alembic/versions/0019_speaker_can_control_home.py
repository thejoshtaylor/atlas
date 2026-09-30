"""speakers.can_control_home -- a per-member home control permission (D-A)

Revision ID: 0019
Revises: 0018
Create Date: 2026-09-29

Quick task 260929-p12: an admin can stop one household member from running
home writes by voice. The column is a boolean that is NOT NULL and defaults
to true on the server. Every existing member and every new enrollment keeps
today's behavior, so this migration changes nothing until an admin clears the
flag for a member (D-A).

The flag applies only in speaker_id enforce mode, to a voice the gate
identifies. `turn/home_control.py` explains the rule and its limits.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0019"
down_revision: Union[str, Sequence[str], None] = "0018"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "speakers",
        sa.Column("can_control_home", sa.Boolean(), nullable=False, server_default=sa.true()),
    )


def downgrade() -> None:
    op.drop_column("speakers", "can_control_home")
