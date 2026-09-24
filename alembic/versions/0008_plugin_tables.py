"""plugin tables

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-18

Creates `plugins` and `plugin_config_values` (D-01, D-04, PLUG-01, PLUG-09)
and turns the two hardcoded MCP children `app.py`'s `lifespan` used to spawn
-- Home Assistant and weather -- into ordinary rows in the first table,
seeded from the operator's own, already-deployed `mcp.servers.*` block
(D-01) -- the same seed-then-reject move `0001_policy_tables.py` and
`0005_macro_tables.py` each made for their own legacy key, generalized
rather than copied (D-16): `app.py`'s two-stage boot gains one more entry
in its `_LEGACY_CONFIG_KEYS` tuple, not a second copy of the sequence that
reads it.

This file holds no house name, no real Home Assistant URL, and no real
token of its own. It reads them, if any exist, at run time from a file that
is not part of this repository: the one `ATLAS_CONFIG` names, the same
environment variable and the same default (`config/config.example.yaml`)
`app.py` already reads.

The seed step does two things, in this order, and unconditionally in the
first case:

1. **The two builtin rows always exist.** `config.example.yaml` carries no
   `safety:` or `macros:` block any more once those were retired, so a
   plugins seed conditional on the config file would leave a fresh install
   with an empty plugins list and no Home Assistant at all -- contradicting
   both DEP-03 (a clean clone reaches a running assistant with no file
   editing) and 06-UI-SPEC.md's vocabulary lock that this list is never
   empty. Home Assistant (`slug="ha"`, `enforces_policy=True`) and weather
   (`slug="weather"`) are inserted regardless of what the file says. When
   the file names a matching `mcp.servers.<slug>` block, that block's `env`
   becomes the row's config values; when it does not, the row's own
   declared keys (`HA_URL`/`HA_TOKEN` for Home Assistant,
   `WEATHER_LATITUDE`/`WEATHER_LONGITUDE` for weather -- the exact env keys
   `config.example.yaml` has always declared for each) are seeded with
   empty values, leaving the plugin visibly unconfigured for an admin to
   fill in the browser rather than silently absent.

2. **Every other `mcp.servers.<name>` entry becomes a fresh, non-builtin
   row.** Parsed through `McpServerConfig.from_config` -- the same parser
   the runtime path has always used, one parser and not two.

Secret classification is an explicit name list (`_SECRET_CONFIG_KEYS`),
never a rule over how a key is spelled: `HA_TOKEN` is encrypted through
`crypto.credentials.encrypt_credential` before it is ever written to
`plugin_config_values.ciphertext`; every other seeded key (including an
empty-string declared key) is stored plain. This migration builds its own
`SecurityConfig` from the same raw config text it reads -- `lifespan`
validates the secret key's strength before migrations run, so the key is
known good by the time this executes.

When the file itself cannot be opened or read, this migration raises
rather than seeding nothing silently -- the same rule `0001`/`0005` already
apply to their own legacy blocks.

`downgrade()` is real, not a stub, matching `0005`'s own precedent for a
migration that carries an operator's real data (here, a real Home Assistant
token) forward.
"""

from __future__ import annotations

import logging
import os
from datetime import datetime, timezone
from typing import Sequence, Union

import sqlalchemy as sa
import yaml
from alembic import op

from atlas.config import McpServerConfig, SecurityConfig, expand_env
from atlas.crypto.credentials import encrypt_credential

logger = logging.getLogger("alembic.plugin_seed")

# revision identifiers, used by Alembic.
revision: str = "0008"
down_revision: Union[str, Sequence[str], None] = "0007"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None

# D-01, D-04: the two children Phase 1 and Phase 4 hardcoded in `lifespan`,
# seeded here unconditionally -- see the module docstring for why "always
# these two rows" is deliberate, not a fallback. `declared_keys` is what a
# builtin row's config values fall back to when the file carries no
# matching `mcp.servers.<slug>` block: the exact env keys
# `config.example.yaml` has always declared for each child, so an
# unconfigured builtin plugin is visibly incomplete rather than silently
# absent.
_BUILTIN_PLUGINS = (
    {
        "slug": "ha",
        "display_name": "Home Assistant",
        "args": ["-m", "atlas_mcp.ha"],
        "enforces_policy": True,
        "declared_keys": ("HA_URL", "HA_TOKEN"),
    },
    {
        "slug": "weather",
        "display_name": "Weather",
        "args": ["-m", "atlas_mcp.weather"],
        "enforces_policy": False,
        "declared_keys": ("WEATHER_LATITUDE", "WEATHER_LONGITUDE"),
    },
)

# Orchestrator Addendum Q2: an explicit name list, never a rule over how a
# key is spelled -- weather holds no credential at all, which is exactly
# why Phase 4 chose it as the second child (D-04).
_SECRET_CONFIG_KEYS = frozenset({"HA_TOKEN"})


def upgrade() -> None:
    _create_tables()
    _seed_plugins()


def _create_tables() -> None:
    op.create_table(
        "plugins",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("slug", sa.Text(), nullable=False),
        sa.Column("display_name", sa.Text(), nullable=False),
        sa.Column("transport", sa.Text(), nullable=False),
        sa.Column("args", sa.JSON(), nullable=False),
        sa.Column("url", sa.Text(), nullable=True),
        sa.Column("enabled", sa.Boolean(), nullable=False),
        sa.Column("builtin", sa.Boolean(), nullable=False),
        sa.Column("enforces_policy", sa.Boolean(), nullable=False),
        sa.Column("timeout_ms", sa.Integer(), nullable=False),
        sa.Column("created_at", sa.DateTime(), nullable=False),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.Column("created_by_user_id", sa.Integer(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint("slug", name="uq_plugins_slug"),
        sa.ForeignKeyConstraint(
            ["created_by_user_id"], ["users.id"], name="fk_plugins_created_by_user_id"
        ),
    )
    op.create_table(
        "plugin_config_values",
        sa.Column("id", sa.Integer(), nullable=False),
        sa.Column("plugin_id", sa.Integer(), nullable=False),
        sa.Column("key", sa.Text(), nullable=False),
        sa.Column("secret", sa.Boolean(), nullable=False),
        sa.Column("value", sa.Text(), nullable=True),
        sa.Column("ciphertext", sa.LargeBinary(), nullable=True),
        sa.Column("key_version", sa.Integer(), nullable=True),
        sa.Column("updated_at", sa.DateTime(), nullable=False),
        sa.PrimaryKeyConstraint("id"),
        sa.ForeignKeyConstraint(
            ["plugin_id"],
            ["plugins.id"],
            name="fk_plugin_config_values_plugin_id",
            ondelete="CASCADE",
        ),
    )


def _read_config_text(config_path: str) -> str:
    """Read `config_path` (the file `ATLAS_CONFIG` names), raising rather
    than seeding nothing silently when it cannot be opened or read --
    matching `0001_policy_tables.py::_read_safety_block`'s and
    `0005_macro_tables.py::_read_macros_block`'s identical rule.
    """
    try:
        with open(config_path, encoding="utf-8") as fh:
            return fh.read()
    except OSError as exc:
        raise RuntimeError(
            f"the plugin seed migration could not read {config_path!r} (named by "
            "ATLAS_CONFIG, or its default) to seed plugins -- refusing to seed "
            "nothing silently. Either make the file readable at that path, or "
            "point ATLAS_CONFIG at the file that holds the mcp.servers: block to "
            "carry forward."
        ) from exc


def _read_mcp_servers_block(config_path: str) -> dict | None:
    """The file's `mcp.servers:` mapping, or `None` when the file has no
    `mcp:` key, or an `mcp:` key with no `servers:` key -- a real, ordinary
    choice, not an error."""
    raw = yaml.safe_load(expand_env(_read_config_text(config_path))) or {}
    return (raw.get("mcp") or {}).get("servers")


def _security_config(config_path: str) -> SecurityConfig:
    """The `SecurityConfig` this migration encrypts `HA_TOKEN` under --
    built from the same raw config text `_read_mcp_servers_block` reads,
    per this file's own module docstring: `lifespan` validates the secret
    key's strength before migrations run, so the key is known good by the
    time this executes."""
    raw = yaml.safe_load(expand_env(_read_config_text(config_path))) or {}
    return SecurityConfig.from_config(raw.get("security"))


def _config_values_rows(
    plugin_id: int, env: dict, now: datetime, security: SecurityConfig
) -> list[dict]:
    """One `plugin_config_values` row per `env` entry -- a secret key
    (`_SECRET_CONFIG_KEYS`) encrypted through `encrypt_credential` before
    it is ever written, every other key stored plain (D-03)."""
    rows = []
    for key, value in env.items():
        if key in _SECRET_CONFIG_KEYS:
            ciphertext, key_version = encrypt_credential(value or "", security)
            rows.append(
                {
                    "plugin_id": plugin_id,
                    "key": key,
                    "secret": True,
                    "value": None,
                    "ciphertext": ciphertext,
                    "key_version": key_version,
                    "updated_at": now,
                }
            )
        else:
            rows.append(
                {
                    "plugin_id": plugin_id,
                    "key": key,
                    "secret": False,
                    "value": value or "",
                    "ciphertext": None,
                    "key_version": None,
                    "updated_at": now,
                }
            )
    return rows


def _seed_plugins() -> None:
    config_path = os.environ.get("ATLAS_CONFIG", "config/config.example.yaml")
    servers_raw = dict(_read_mcp_servers_block(config_path) or {})
    security = _security_config(config_path)

    now = datetime.now(timezone.utc)
    conn = op.get_bind()
    metadata = sa.MetaData()
    plugins_table = sa.Table("plugins", metadata, autoload_with=conn)
    config_values_table = sa.Table("plugin_config_values", metadata, autoload_with=conn)

    total_config_values = 0

    # First: the two builtin rows, unconditionally (module docstring).
    for builtin in _BUILTIN_PLUGINS:
        slug = builtin["slug"]
        server_block = servers_raw.pop(slug, None)
        if server_block is not None:
            env = dict(McpServerConfig.from_config(server_block).env)
        else:
            env = {key: "" for key in builtin["declared_keys"]}

        result = conn.execute(
            plugins_table.insert().values(
                slug=slug,
                display_name=builtin["display_name"],
                transport="stdio",
                args=list(builtin["args"]),
                url=None,
                enabled=True,
                builtin=True,
                enforces_policy=builtin["enforces_policy"],
                timeout_ms=5000,
                created_at=now,
                updated_at=now,
                created_by_user_id=None,
            )
        )
        plugin_id = result.inserted_primary_key[0]
        rows = _config_values_rows(plugin_id, env, now, security)
        if rows:
            conn.execute(config_values_table.insert(), rows)
        total_config_values += len(rows)

    # Second: every remaining `mcp.servers.<name>` entry -- one that names
    # neither builtin slug -- becomes a fresh, non-builtin row (D-01).
    for slug, server_block in servers_raw.items():
        server_config = McpServerConfig.from_config(server_block)
        result = conn.execute(
            plugins_table.insert().values(
                slug=slug,
                display_name=slug,
                transport="stdio",
                args=list(server_config.args),
                url=None,
                enabled=True,
                builtin=False,
                enforces_policy=False,
                timeout_ms=5000,
                created_at=now,
                updated_at=now,
                created_by_user_id=None,
            )
        )
        plugin_id = result.inserted_primary_key[0]
        rows = _config_values_rows(plugin_id, dict(server_config.env), now, security)
        if rows:
            conn.execute(config_values_table.insert(), rows)
        total_config_values += len(rows)

    logger.info(
        "plugin seed: %d builtin plugin(s), %d additional plugin(s) from mcp.servers.*, "
        "%d config value(s) seeded",
        len(_BUILTIN_PLUGINS),
        len(servers_raw),
        total_config_values,
    )


def downgrade() -> None:
    op.drop_table("plugin_config_values")
    op.drop_table("plugins")
