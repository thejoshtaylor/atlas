"""seed WEATHER_UNITS on the weather plugin's stored config

Revision ID: 0013
Revises: 0012
Create Date: 2026-09-23

260923-lmg-PLAN.md's planner finding 1: nothing at boot syncs the plugin
catalog into an existing install's stored config rows. `config/plugin-
catalog.json` is read only in two places -- `GET /api/plugins/catalog`
and the catalog install path (`_catalog_install_config_values` in
`routes/plugins.py`), and the install path writes one row per declared
key only at install time. The config editor renders the stored database
rows (`_to_plugin_response` -> `PluginRepository.get_config_values`), and
the child's environment is built only from stored rows
(`plugins/manager.py::_env_from_config_values`). So adding WEATHER_UNITS
to the catalog in this same plan never reaches an existing install: the
editor would not show it, and the weather child would never receive it.

This migration is what carries the key into an existing install's stored
rows. `0008_plugin_tables.py` always seeds the builtin `weather` row
(LAT/LON only), and this migration runs after it on a fresh install too
-- so a fresh install and an upgraded one both end at the same state:
one WEATHER_UNITS row holding the plain string "celsius".

An operator's own saved value is never overwritten (T-lmg-05): the
insert only happens when no (weather, WEATHER_UNITS) row exists yet.
`0010`'s unique constraint on `(plugin_id, key)` would refuse a
duplicate insert anyway, but checking first keeps this migration's
intent explicit rather than leaning on that constraint to fail loudly.

The downgrade is real, not a stub (the `0005`/`0008` precedent): it
deletes the (weather, WEATHER_UNITS) row, even if the operator changed
its value. The plugin falls back to its own "celsius" default at
startup once the row is gone.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
from alembic import op

# revision identifiers, used by Alembic.
revision: str = "0013"
down_revision: Union[str, Sequence[str], None] = "0012"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

_PLUGIN_SLUG = "weather"
_CONFIG_KEY = "WEATHER_UNITS"
_DEFAULT_VALUE = "celsius"


def upgrade() -> None:
    conn = op.get_bind()
    metadata = sa.MetaData()
    plugins = sa.Table("plugins", metadata, autoload_with=conn)
    config_values = sa.Table("plugin_config_values", metadata, autoload_with=conn)

    plugin_id = conn.execute(
        sa.select(plugins.c.id).where(plugins.c.slug == _PLUGIN_SLUG)
    ).scalar_one_or_none()
    if plugin_id is None:
        # No weather plugin to seed. This migration moves a value onto a
        # row that must already exist; it never creates the plugin row
        # itself, and it adds no house data either way.
        return

    existing = conn.execute(
        sa.select(config_values.c.id).where(
            sa.and_(
                config_values.c.plugin_id == plugin_id,
                config_values.c.key == _CONFIG_KEY,
            )
        )
    ).scalar_one_or_none()
    if existing is not None:
        # An operator's own row already exists -- saved through the
        # editor, or seeded by an earlier run of this same migration.
        # Leave it exactly as it is.
        return

    conn.execute(
        config_values.insert().values(
            plugin_id=plugin_id,
            key=_CONFIG_KEY,
            secret=False,
            value=_DEFAULT_VALUE,
            ciphertext=None,
            key_version=None,
            updated_at=datetime.now(timezone.utc),
        )
    )


def downgrade() -> None:
    conn = op.get_bind()
    metadata = sa.MetaData()
    plugins = sa.Table("plugins", metadata, autoload_with=conn)
    config_values = sa.Table("plugin_config_values", metadata, autoload_with=conn)

    plugin_id = conn.execute(
        sa.select(plugins.c.id).where(plugins.c.slug == _PLUGIN_SLUG)
    ).scalar_one_or_none()
    if plugin_id is None:
        return

    conn.execute(
        config_values.delete().where(
            sa.and_(
                config_values.c.plugin_id == plugin_id,
                config_values.c.key == _CONFIG_KEY,
            )
        )
    )
