"""plugin_config_values (plugin_id, key) uniqueness

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-19

IN-03 (06-REVIEW.md): `plugin_config_values` had no database-level
uniqueness on `(plugin_id, key)`, unlike `uq_plugins_slug` on its parent
table. `PostgresPluginRepository.set_config_values` is a read-then-upsert
with no row lock, so two concurrent configuration saves for the same
plugin could both miss the existing row and insert duplicates for one key
-- after which `_env_from_config_values` silently passed the child
whichever row the unordered `SELECT` happened to return last, and the
editor showed only one of them. Which of two values a plugin's child
actually received was decided by row order, invisibly.

Any duplicate that already exists is resolved before the constraint is
added, keeping the highest `id` for each `(plugin_id, key)` -- the most
recently inserted row, which is the one `set_config_values`' own
read-then-upsert would have been reading back as current. Deleting the
older twin is the only choice that can be made without decrypting
anything: the two rows are, by construction, two writes of the same key,
and the later one is what the operator last saved.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0010"
down_revision: Union[str, Sequence[str], None] = "0009"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_CONSTRAINT = "uq_plugin_config_values_plugin_id_key"


def upgrade() -> None:
    op.execute(
        sa.text(
            """
            DELETE FROM plugin_config_values a
            USING plugin_config_values b
            WHERE a.plugin_id = b.plugin_id
              AND a.key = b.key
              AND a.id < b.id
            """
        )
    )
    op.create_unique_constraint(_CONSTRAINT, "plugin_config_values", ["plugin_id", "key"])


def downgrade() -> None:
    op.drop_constraint(_CONSTRAINT, "plugin_config_values", type_="unique")
