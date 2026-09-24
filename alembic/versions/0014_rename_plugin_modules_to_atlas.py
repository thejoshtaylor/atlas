"""point stored plugin args at the renamed atlas_mcp package

Revision ID: 0014
Revises: 0013
Create Date: 2026-09-23

The project was renamed from spire-voice to ATLAS, and the builtin MCP
package moved from `spire_mcp` to `atlas_mcp`. `0008_plugin_tables.py`
seeds the builtin plugins with `["-m", "<module>"]` in `plugins.args`, so
an install seeded before the rename still stores `spire_mcp.ha` and
`spire_mcp.weather`. The child then fails at startup with
`No module named 'spire_mcp'`. This migration rewrites that prefix in
every stored `args` list. A fresh install already seeds `atlas_mcp.*`,
so on a fresh install the upgrade changes nothing.

The downgrade reverses the prefix, so an older image that still ships
`spire_mcp` can start its children again.
"""

from __future__ import annotations

from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0014"
down_revision: Union[str, Sequence[str], None] = "0013"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def _rewrite_prefix(old: str, new: str) -> None:
    conn = op.get_bind()
    plugins = sa.Table("plugins", sa.MetaData(), autoload_with=conn)
    for row in conn.execute(sa.select(plugins.c.id, plugins.c.args)).all():
        args = list(row.args or [])
        rewritten = [new + a[len(old):] if isinstance(a, str) and a.startswith(old) else a for a in args]
        if rewritten != args:
            conn.execute(plugins.update().where(plugins.c.id == row.id).values(args=rewritten))


def upgrade() -> None:
    _rewrite_prefix("spire_mcp.", "atlas_mcp.")


def downgrade() -> None:
    _rewrite_prefix("atlas_mcp.", "spire_mcp.")
