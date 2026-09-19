"""carry the ha_token credential into the Home Assistant plugin's config

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-19

WR-06 (06-REVIEW.md): plan 06-01 moved the Home Assistant token into
`plugin_config_values` (D-01) and left the `ha_token` credential slot
behind. `0008_plugin_tables.py` seeded `HA_TOKEN` purely from the
configuration file's own `mcp.servers.ha.env` block and never looked at
the `provider_credentials` row an operator may have written through the
settings screen or the setup wizard -- so an operator who had rotated the
token in the browser silently got the older, file-sourced value back, and
every later write through that screen went into a table nothing reads.

This migration carries that row's ciphertext across, in the one direction
Phase 3's own resolution order already states: a value an operator saved
in the database wins over one the configuration file's environment
expansion produced. Nothing is re-encrypted -- the same Fernet key
version is stored beside the same ciphertext, so this migration needs no
secret key of its own and cannot fail on a missing one.

The `provider_credentials` row is deleted afterwards, deliberately: one
token, one storage location. `routes/credentials.py` answers the
`ha_token` slot from the plugin's configuration value now, so leaving the
row behind would recreate exactly the ambiguity this fixes.

A deployment with no `ha_token` credential row, or no Home Assistant
plugin row, is the ordinary case: this migration does nothing at all.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0009"
down_revision: Union[str, Sequence[str], None] = "0008"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_SLOT = "ha_token"
_PLUGIN_SLUG = "ha"
_CONFIG_KEY = "HA_TOKEN"


def upgrade() -> None:
    conn = op.get_bind()
    metadata = sa.MetaData()
    credentials = sa.Table("provider_credentials", metadata, autoload_with=conn)
    plugins = sa.Table("plugins", metadata, autoload_with=conn)
    config_values = sa.Table("plugin_config_values", metadata, autoload_with=conn)

    credential = conn.execute(
        sa.select(credentials.c.ciphertext, credentials.c.key_version).where(
            credentials.c.slot == _SLOT
        )
    ).first()
    if credential is None:
        return

    plugin_id = conn.execute(
        sa.select(plugins.c.id).where(plugins.c.slug == _PLUGIN_SLUG)
    ).scalar_one_or_none()
    if plugin_id is None:
        # No Home Assistant plugin to carry it to. The credential row is
        # left exactly where it is rather than discarded: this migration
        # moves a value, it never destroys one it could not deliver.
        return

    now = datetime.now(timezone.utc)
    existing = conn.execute(
        sa.select(config_values.c.id).where(
            sa.and_(
                config_values.c.plugin_id == plugin_id,
                config_values.c.key == _CONFIG_KEY,
            )
        )
    ).scalar_one_or_none()

    if existing is None:
        conn.execute(
            config_values.insert().values(
                plugin_id=plugin_id,
                key=_CONFIG_KEY,
                secret=True,
                value=None,
                ciphertext=credential.ciphertext,
                key_version=credential.key_version,
                updated_at=now,
            )
        )
    else:
        conn.execute(
            config_values.update()
            .where(config_values.c.id == existing)
            .values(
                secret=True,
                value=None,
                ciphertext=credential.ciphertext,
                key_version=credential.key_version,
                updated_at=now,
            )
        )

    conn.execute(credentials.delete().where(credentials.c.slot == _SLOT))


def downgrade() -> None:
    # Deliberately not reversible: the value is a single secret that now
    # lives in one place, and writing it back into `provider_credentials`
    # would recreate the two-places-one-reader state WR-06 is about. A
    # downgrade past this point keeps the token where the plugin reads it.
    pass
