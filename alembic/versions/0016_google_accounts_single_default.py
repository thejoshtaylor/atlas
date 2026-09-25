"""google_accounts: at most one row is_default at the database level

Revision ID: 0016
Revises: 0015
Create Date: 2026-09-25

B1-WR-04 (09-REVIEW.md): `google_repository.py::update_account`'s own
docstring promises "there is never a moment where two rows are both
default at once," but until this migration that invariant was enforced
only by application code inside a single request's transaction (clear
every other row's `is_default`, then set this row's) -- with no partial
unique index backing it at the database level. Two concurrent
`PATCH /api/google/accounts/{id}` requests each setting a *different*
account's `is_default=True` could, under Postgres's default read
committed isolation, both commit successfully: each transaction's own
`UPDATE ... WHERE id != account_id SET is_default=False` only sees state
as of its own execution, never the other transaction's still-uncommitted
write.

Like `0010_plugin_config_value_uniqueness.py`, any violation that already
exists is resolved before the index is added -- keeping the most
recently updated `is_default=true` row as the one true default (the row
`updated_at` names as the last write to actually land), and clearing
every other one. This table is Phase 9's own, and application code has
never allowed two simultaneous defaults by design, so no such row is
expected to exist in practice; the cleanup is defense in depth, the same
posture 0010's own docstring takes.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0016"
down_revision: Union[str, Sequence[str], None] = "0015"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_INDEX = "uq_google_accounts_single_default"


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            UPDATE google_accounts
            SET is_default = false
            WHERE is_default = true
              AND id NOT IN (
                  SELECT id FROM google_accounts
                  WHERE is_default = true
                  ORDER BY updated_at DESC, id DESC
                  LIMIT 1
              )
            """
        )
    )
    op.create_index(
        _INDEX,
        "google_accounts",
        ["is_default"],
        unique=True,
        postgresql_where=sa.text("is_default"),
    )


def downgrade() -> None:
    op.drop_index(_INDEX, table_name="google_accounts")
