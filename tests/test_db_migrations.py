"""The migration path itself, and the seed that carries an operator's real
denylist across the config-to-database cut (DEP-04, SAFE-05, D-11).

DEP-04 promises "an upgrade keeps existing data" -- the one property this
project has never had to prove before, because Phase 3 is the first phase
with a database at all. A migration that is not actually idempotent (running
it twice against an already-migrated schema errors, or silently duplicates
rows) would fail that promise the first time an operator restarts after this
phase ships. D-11 promises something narrower and higher-stakes: the
operator's real, already-deployed `safety:` block -- eight entries on this
specific house, entered before this phase existed -- must be seeded into the
database by the first migration, not discarded. A migration that runs clean
but seeds an empty policy is a silent safety regression: the process boots,
the denylist editor loads, and nothing looks wrong until the entity that used
to be denied is not.

Both tests are marked `integration` and need a real Postgres. This
repository has no prior conditional-skip precedent, so the mechanics here
are new: `pytest.mark.skipif` reads `ATLAS_TEST_DATABASE_URL` from the
environment and skips, naming that variable, when it is absent -- matching
D-04's "the suite runs with no Postgres reachable at all." To make these two
tests run instead of skip: `eval "$(scripts/dev-postgres.sh)"` (starts a
throwaway local Postgres and exports both `DATABASE_URL` and
`ATLAS_TEST_DATABASE_URL`), then re-run the suite.

Neither test touches an operator's real configuration file or real database.
Both write their own, invented `safety:` block to a temporary file and point
`ATLAS_CONFIG` at that -- the same `switch.example_*`/`light.example_*`
naming `tests/conftest.py` and `tests/test_repo_hygiene.py` already enforce
project-wide.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

_TEST_DB_URL = os.environ.get("ATLAS_TEST_DATABASE_URL")

pytestmark = pytest.mark.integration

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "ATLAS_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these two tests instead of skipping them"
    ),
)


def _migration_url(async_url: str) -> str:
    """The synchronous `psycopg` connection string Alembic needs, derived
    the same way `atlas.config.DatabaseConfig.migration_url` derives
    it from the async runtime URL -- a driver swap, nothing else."""
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    """Drop every table (and Alembic's own bookkeeping table) either
    migration touches, so each test starts from a genuinely empty database
    regardless of what a previous test run left behind on the same
    throwaway Postgres.

    Plan 03-05 extended this list: `0002_accounts.py` adds `users`,
    `invites`, and `refresh_tokens`, and adds foreign keys from
    `safety_policy`/`policy_rules` onto `users` -- `CASCADE` handles the
    drop order regardless of which side a leftover constraint points from,
    but every table either migration creates must be named here or a
    second test run against the same throwaway Postgres finds the first
    run's tables still present. Plan 03-07 extends it again:
    `0003_credentials.py` adds `provider_credentials`. Plan 03-09 extends
    it once more: `0004_setup_state.py` adds `setup_state`, `setup_steps`,
    and `settings`.
    """
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        # Phase 6 (plan 06-01 follow-up): drop the whole schema, never a
        # hand-maintained table list. Migration 0008 added `plugins` and
        # `plugin_config_values`, and not one of the nine copies of this
        # helper knew about them -- so a reset dropped `alembic_version` but
        # left those two tables standing, and the very next `upgrade head`
        # died on `DuplicateTable: relation "plugins" already exists`. Every
        # Postgres-backed test after the first one failed that way. A list
        # that has to be edited in nine files each time a migration lands is
        # itself the defect; a schema drop cannot drift out of date.
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def _run_upgrade_head() -> None:
    """Run every migration up to `head` against `ATLAS_TEST_DATABASE_URL`,
    reading `ATLAS_CONFIG` (already set by the caller) the same way `app.py`
    will at real startup (Task 4)."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(_TEST_DB_URL))
    command.upgrade(cfg, "head")


async def _fetch_rows(async_url: str) -> tuple[list, list]:
    engine = create_async_engine(async_url)
    async with engine.connect() as conn:
        policy_rows = (
            await conn.execute(text("SELECT id, mode FROM safety_policy"))
        ).fetchall()
        rule_rows = (
            await conn.execute(text("SELECT kind, value FROM policy_rules ORDER BY id"))
        ).fetchall()
    await engine.dispose()
    return policy_rows, rule_rows


async def _fetch_macro_rows(async_url: str) -> list:
    """Every `macros` row -- id and phrase only, matching `_fetch_rows`'
    own "just enough to prove seeded vs. not" shape. Plan 04-05's sibling
    of `_fetch_rows` above, kept separate rather than folded in: the two
    migrations seed from two independent legacy keys, and a reader should
    be able to check one without reading the other's shape."""
    engine = create_async_engine(async_url)
    async with engine.connect() as conn:
        macro_rows = (
            await conn.execute(text("SELECT id, phrase FROM macros ORDER BY id"))
        ).fetchall()
    await engine.dispose()
    return macro_rows


async def _fetch_plugin_rows(async_url: str) -> list[tuple[str, bool, bool]]:
    """`(slug, builtin, enforces_policy)` for every seeded `plugins` row,
    in id order -- migration `0008`'s own sibling of `_fetch_macro_rows`
    above."""
    engine = create_async_engine(async_url)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT slug, builtin, enforces_policy FROM plugins ORDER BY id")
                )
            ).fetchall()
        return [(r.slug, r.builtin, r.enforces_policy) for r in rows]
    finally:
        await engine.dispose()


@skip_without_postgres
async def test_migrations_run_from_empty_and_are_idempotent(monkeypatch):
    """Running every migration twice against the same database, from empty,
    must succeed both times with no duplicated row and no error.

    `config/config.example.yaml` carries no `macros:` block (D-09: the key
    is retired), so migration `0005`'s own seed step seeds nothing against
    it -- this test's macro assertion proves the "absent key" case, the
    same way its policy assertion already proves `safety:`'s "absent key"
    case. Plan 06-01: the same file carries no `mcp:` block either (D-01) --
    migration `0008` still seeds exactly the two builtin plugin rows
    unconditionally (this file's own module docstring: "a fresh install
    reaches a running assistant with no file editing"), which is what the
    plugin assertion below proves for the "absent key" case."""
    await _reset_schema(_TEST_DB_URL)
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    # Phase 7 (D-15): server.bind_host / security.cookie_secure -- the two
    # new ${VAR} placeholders config.example.yaml expands.
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()
    first_policy, first_rules = await _fetch_rows(_TEST_DB_URL)
    assert first_policy == [(1, "allow_all_except_denylist")]
    assert first_rules == []
    assert await _fetch_macro_rows(_TEST_DB_URL) == []
    assert await _fetch_plugin_rows(_TEST_DB_URL) == [("ha", True, True), ("weather", True, False)]

    # Idempotent: Alembic's own applied-revision bookkeeping means a second
    # upgrade to the same head is a no-op, not a second seed.
    _run_upgrade_head()
    second_policy, second_rules = await _fetch_rows(_TEST_DB_URL)
    assert second_policy == first_policy
    assert second_rules == first_rules
    assert await _fetch_macro_rows(_TEST_DB_URL) == []
    assert await _fetch_plugin_rows(_TEST_DB_URL) == [("ha", True, True), ("weather", True, False)]


@skip_without_postgres
async def test_seeded_policy_matches_the_config_block_it_came_from(tmp_path: Path, monkeypatch):
    """The first migration's seed step must carry an equivalent-shaped
    `safety:` block into the database rows `Policy.from_db_rows` reads --
    row-for-row equality with the block it seeded from, not merely a
    non-zero row count."""
    await _reset_schema(_TEST_DB_URL)

    safety_block = {
        "mode": "allow_all_except_denylist",
        "deny_entities": [
            "switch.example_seed_test_socket",
            "switch.example_seed_test_heater",
        ],
        "deny_patterns": ["switch.example_seed_test_camera_*"],
        "allow_entities": ["light.example_seed_test_lamp"],
        "allow_patterns": ["light.example_seed_test_*"],
    }
    raw = {
        "server": {"transport": "websocket"},
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": "test-key"},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": "test-key",
            "models": [{"model": "fake-model"}],
        },
        "tts": {"url": "https://tts.invalid/v1/tts", "api_key": "test-key", "voice_id": "eve"},
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "atlas_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-key"},
                },
            },
        },
        "database": {"url": _TEST_DB_URL},
        "safety": safety_block,
    }
    config_path = tmp_path / "seed-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("ATLAS_CONFIG", str(config_path))
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()
    policy_rows, rule_rows = await _fetch_rows(_TEST_DB_URL)

    assert policy_rows == [(1, safety_block["mode"])]

    seeded_deny_entities = {v for k, v in rule_rows if k == "deny_entity"}
    seeded_deny_patterns = {v for k, v in rule_rows if k == "deny_pattern"}
    seeded_allow_entities = {v for k, v in rule_rows if k == "allow_entity"}
    seeded_allow_patterns = {v for k, v in rule_rows if k == "allow_pattern"}

    assert seeded_deny_entities == set(safety_block["deny_entities"])
    assert seeded_deny_patterns == set(safety_block["deny_patterns"])
    assert seeded_allow_entities == set(safety_block["allow_entities"])
    assert seeded_allow_patterns == set(safety_block["allow_patterns"])
    # Nothing added and nothing dropped: the total row count matches exactly
    # what the block named, not a superset or a subset of it.
    total_seeded = sum(1 for value in safety_block.values() if isinstance(value, list) for _ in value)
    assert len(rule_rows) == total_seeded


@skip_without_postgres
async def test_seeded_macros_match_the_config_block_they_came_from(tmp_path: Path, monkeypatch):
    """Migration `0005`'s seed step must carry an equivalent-shaped
    `macros:` block into the database -- row-for-row equality with the
    block it seeded from, not merely a non-zero row count, the same
    discipline `test_seeded_policy_matches_the_config_block_it_came_from`
    above already applies to `safety:` (D-09)."""
    await _reset_schema(_TEST_DB_URL)

    macros_block = [
        {
            "phrase": "example db migration macro",
            "aliases": ["example db migration alias"],
            "reply": "okay",
            "actions": [
                {
                    "tool": "ha_call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": "switch.example_migration_seed_fan",
                    },
                },
                {
                    "tool": "ha_call_service",
                    "arguments": {
                        "domain": "light",
                        "service": "turn_off",
                        "entity_id": "light.example_migration_seed_lamp",
                    },
                },
            ],
        },
    ]
    raw = {
        "server": {"transport": "websocket"},
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": "test-key"},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": "test-key",
            "models": [{"model": "fake-model"}],
        },
        "tts": {"url": "https://tts.invalid/v1/tts", "api_key": "test-key", "voice_id": "eve"},
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "atlas_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-key"},
                },
            },
        },
        "database": {"url": _TEST_DB_URL},
        "macros": macros_block,
    }
    config_path = tmp_path / "macro-seed-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("ATLAS_CONFIG", str(config_path))
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()

    async def _fetch_seeded_macro() -> tuple[tuple, list, list]:
        engine = create_async_engine(_TEST_DB_URL)
        try:
            async with engine.connect() as conn:
                macro_row = (
                    await conn.execute(text("SELECT id, phrase, reply FROM macros"))
                ).one()
                action_rows = (
                    await conn.execute(
                        text(
                            "SELECT tool, arguments FROM macro_actions "
                            "WHERE macro_id = :macro_id ORDER BY position"
                        ),
                        {"macro_id": macro_row.id},
                    )
                ).fetchall()
                alias_rows = (
                    await conn.execute(
                        text("SELECT alias FROM macro_aliases WHERE macro_id = :macro_id"),
                        {"macro_id": macro_row.id},
                    )
                ).fetchall()
                return macro_row, list(action_rows), list(alias_rows)
        finally:
            await engine.dispose()

    macro_row, action_rows, alias_rows = await _fetch_seeded_macro()

    expected = macros_block[0]
    assert macro_row.phrase == expected["phrase"]
    assert macro_row.reply == expected["reply"]
    assert {r.alias for r in alias_rows} == set(expected["aliases"])
    assert [
        {"tool": r.tool, "arguments": r.arguments} for r in action_rows
    ] == expected["actions"]


@skip_without_postgres
async def test_seeded_plugins_match_the_mcp_servers_block_they_came_from(
    tmp_path: Path, monkeypatch
):
    """Migration `0008`'s seed step must carry an equivalent-shaped
    `mcp.servers:` block into the database -- row-for-row equality with
    the block it seeded from, not merely a non-zero row count, the same
    discipline `test_seeded_macros_match_the_config_block_they_came_from`
    above already applies to `macros:` (D-01). Also proves D-03: `HA_TOKEN`
    is ciphertext at rest, never the plaintext this test wrote to the
    config file.
    """
    await _reset_schema(_TEST_DB_URL)

    ha_token_plaintext = "a-plainly-fictional-migration-seed-test-token"
    raw = {
        "server": {"transport": "websocket"},
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": "test-key"},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": "test-key",
            "models": [{"model": "fake-model"}],
        },
        "tts": {"url": "https://tts.invalid/v1/tts", "api_key": "test-key", "voice_id": "eve"},
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "atlas_mcp.ha"],
                    "env": {
                        "HA_URL": "http://ha.invalid:8123",
                        "HA_TOKEN": ha_token_plaintext,
                    },
                },
                "weather": {
                    "args": ["-m", "atlas_mcp.weather"],
                    "env": {"WEATHER_LATITUDE": "51.5", "WEATHER_LONGITUDE": "-0.1"},
                },
            },
        },
        "database": {"url": _TEST_DB_URL},
        "security": {},
    }
    config_path = tmp_path / "plugin-seed-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("ATLAS_CONFIG", str(config_path))
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()

    async def _fetch_seeded_plugins():
        engine = create_async_engine(_TEST_DB_URL)
        try:
            async with engine.connect() as conn:
                plugin_rows = (
                    await conn.execute(
                        text(
                            "SELECT id, slug, display_name, args, builtin, enforces_policy "
                            "FROM plugins ORDER BY id"
                        )
                    )
                ).fetchall()
                config_value_rows = {}
                for row in plugin_rows:
                    values = (
                        await conn.execute(
                            text(
                                "SELECT key, secret, value, ciphertext, key_version "
                                "FROM plugin_config_values WHERE plugin_id = :plugin_id"
                            ),
                            {"plugin_id": row.id},
                        )
                    ).fetchall()
                    config_value_rows[row.slug] = list(values)
                return plugin_rows, config_value_rows
        finally:
            await engine.dispose()

    plugin_rows, config_value_rows = await _fetch_seeded_plugins()

    by_slug = {row.slug: row for row in plugin_rows}
    assert set(by_slug) == {"ha", "weather"}

    assert by_slug["ha"].args == ["-m", "atlas_mcp.ha"]
    assert by_slug["ha"].builtin is True
    assert by_slug["ha"].enforces_policy is True
    assert by_slug["weather"].args == ["-m", "atlas_mcp.weather"]
    assert by_slug["weather"].builtin is True
    assert by_slug["weather"].enforces_policy is False

    ha_values = {v.key: v for v in config_value_rows["ha"]}
    assert ha_values["HA_URL"].secret is False
    assert ha_values["HA_URL"].value == "http://ha.invalid:8123"
    assert ha_values["HA_TOKEN"].secret is True
    assert ha_values["HA_TOKEN"].value is None
    assert ha_values["HA_TOKEN"].ciphertext is not None
    # D-03: the plaintext token must never appear verbatim in the stored
    # ciphertext -- proving this row is genuinely encrypted, not a
    # base64-shaped no-op.
    assert ha_token_plaintext.encode("utf-8") not in ha_values["HA_TOKEN"].ciphertext

    weather_values = {v.key: v for v in config_value_rows["weather"]}
    assert weather_values["WEATHER_LATITUDE"].secret is False
    assert weather_values["WEATHER_LATITUDE"].value == "51.5"
    assert weather_values["WEATHER_LONGITUDE"].value == "-0.1"
    # Migration 0013 seeds WEATHER_UNITS on every fresh install too --
    # 0008 always seeds the builtin weather row first, and 0013 runs
    # after it, so a fresh install and an upgraded one both end at the
    # same state (260923-lmg-PLAN.md's planner finding 2).
    assert weather_values["WEATHER_UNITS"].secret is False
    assert weather_values["WEATHER_UNITS"].value == "celsius"


async def _fetch_workflow_step_index_and_constraint_names(async_url: str) -> tuple[set, set]:
    """The index/constraint names `run()` (`inspect`) reports on
    `workflow_steps` -- run through `asyncio.to_thread` since
    `sqlalchemy.inspect` is a synchronous reflection API with no async
    counterpart, the same reason `_run_upgrade_head` above runs Alembic's
    own (synchronous) `command.upgrade` the same way."""
    from sqlalchemy import inspect

    engine = create_async_engine(async_url)

    def _inspect(sync_conn) -> tuple[set, set]:
        inspector = inspect(sync_conn)
        index_names = {ix["name"] for ix in inspector.get_indexes("workflow_steps")}
        constraint_names = {
            uc["name"] for uc in inspector.get_unique_constraints("workflow_steps")
        }
        return index_names, constraint_names

    async with engine.connect() as conn:
        index_names, constraint_names = await conn.run_sync(_inspect)
    await engine.dispose()
    return index_names, constraint_names


@skip_without_postgres
async def test_workflow_tables_upgrade_from_empty_with_index_and_constraint_and_are_idempotent(
    monkeypatch,
):
    """Migration 0006 (plan 05-01): `workflow_runs`/`workflow_steps` exist
    after an upgrade to head from empty, `ix_workflow_steps_status_due_at`
    and `uq_workflow_steps_run_id_position` both exist on `workflow_steps`,
    and running every migration twice is still a no-op the second time --
    the same idempotency guarantee `test_migrations_run_from_empty_and_are_
    idempotent` above already proves for 0001-0005, extended to 0006."""
    await _reset_schema(_TEST_DB_URL)
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    # Phase 7 (D-15): server.bind_host / security.cookie_secure -- the two
    # new ${VAR} placeholders config.example.yaml expands.
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()

    engine = create_async_engine(_TEST_DB_URL)
    async with engine.connect() as conn:
        run_count = (
            await conn.execute(text("SELECT count(*) FROM workflow_runs"))
        ).scalar_one()
        step_count = (
            await conn.execute(text("SELECT count(*) FROM workflow_steps"))
        ).scalar_one()
    await engine.dispose()
    assert run_count == 0
    assert step_count == 0

    index_names, constraint_names = await _fetch_workflow_step_index_and_constraint_names(
        _TEST_DB_URL
    )
    assert "ix_workflow_steps_status_due_at" in index_names
    assert "uq_workflow_steps_run_id_position" in constraint_names

    # Idempotent: a second upgrade to the same head must not error and
    # must not duplicate the index or the constraint.
    _run_upgrade_head()
    second_index_names, second_constraint_names = (
        await _fetch_workflow_step_index_and_constraint_names(_TEST_DB_URL)
    )
    assert second_index_names == index_names
    assert second_constraint_names == constraint_names


def _run_upgrade_to(revision: str) -> None:
    """`_run_upgrade_head` above, stopped at a named revision -- what a
    test needs to stand where a real deployment stood before the migration
    under test existed."""
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(_TEST_DB_URL))
    command.upgrade(cfg, revision)


@skip_without_postgres
async def test_a_stored_ha_token_credential_is_carried_into_the_plugin_that_reads_it(
    tmp_path: Path, monkeypatch
):
    """WR-06 (06-REVIEW.md), DEP-04 ("an upgrade keeps existing data"):
    migration `0008` seeded `HA_TOKEN` purely from the configuration
    file's own `mcp.servers.ha.env` block and never looked at the
    `ha_token` credential row an operator may have saved in the browser --
    so an operator who had rotated that token silently got the older,
    file-sourced value back, and every later write through the credentials
    screen went into a table nothing reads.

    Migration `0009` carries the credential row across, in the direction
    Phase 3's own resolution order already states (a value saved in the
    database wins over one the configuration file's environment expansion
    produced), and deletes it afterwards so one token has one home.
    """
    from atlas.config import SecurityConfig
    from atlas.crypto.credentials import decrypt_credential, encrypt_credential

    await _reset_schema(_TEST_DB_URL)

    file_seeded_value = "a-plainly-fictional-token-from-the-config-file"
    rotated_value = "a-plainly-fictional-token-the-operator-rotated-to"
    raw = {
        "server": {"transport": "websocket"},
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": "test-key"},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": "test-key",
            "models": [{"model": "fake-model"}],
        },
        "tts": {"url": "https://tts.invalid/v1/tts", "api_key": "test-key", "voice_id": "eve"},
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "atlas_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid:8123", "HA_TOKEN": file_seeded_value},
                },
            },
        },
        "database": {"url": _TEST_DB_URL},
        "security": {},
    }
    config_path = tmp_path / "ha-token-carry-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("ATLAS_CONFIG", str(config_path))
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    # Stand where a real deployment stood: plugins seeded from the file,
    # and a token the operator rotated through the credentials screen.
    _run_upgrade_to("0008")
    ciphertext, key_version = encrypt_credential(rotated_value, SecurityConfig())

    engine = create_async_engine(_TEST_DB_URL)
    try:
        async with engine.begin() as conn:
            await conn.execute(
                text(
                    "INSERT INTO provider_credentials (slot, ciphertext, key_version, updated_at) "
                    "VALUES ('ha_token', :ciphertext, :key_version, now())"
                ),
                {"ciphertext": ciphertext, "key_version": key_version},
            )
    finally:
        await engine.dispose()

    _run_upgrade_head()

    engine = create_async_engine(_TEST_DB_URL)
    try:
        async with engine.connect() as conn:
            stored = (
                await conn.execute(
                    text(
                        "SELECT v.secret, v.value, v.ciphertext, v.key_version "
                        "FROM plugin_config_values v JOIN plugins p ON p.id = v.plugin_id "
                        "WHERE p.slug = 'ha' AND v.key = 'HA_TOKEN'"
                    )
                )
            ).fetchall()
            remaining = (
                await conn.execute(
                    text("SELECT slot FROM provider_credentials WHERE slot = 'ha_token'")
                )
            ).fetchall()
    finally:
        await engine.dispose()

    assert len(stored) == 1, "the carry must update the seeded row, never add a second one"
    [row] = stored
    assert row.secret is True
    assert row.value is None
    assert decrypt_credential(row.ciphertext, row.key_version, SecurityConfig()) == rotated_value, (
        "the operator's rotated token must survive the upgrade -- the file-seeded "
        "value silently winning is the WR-06 data loss"
    )
    assert remaining == [], "one token, one home: the credential row is not left behind"

    # Idempotent, like every other migration in this file.
    _run_upgrade_head()


@skip_without_postgres
async def test_migration_0013_seeds_weather_units_without_overwriting_an_operators_value(
    tmp_path: Path, monkeypatch
):
    """260923-lmg-PLAN.md's planner finding 1: nothing at boot syncs the
    plugin catalog into an existing install's stored config rows -- the
    catalog is read only at install time and by the catalog route, and
    both the config editor and the child env are built from stored rows
    only. Adding WEATHER_UNITS to the catalog therefore never reaches an
    existing install without a migration to carry it. This migration must
    seed a fresh weather row with "celsius", never overwrite a value an
    operator already saved (T-lmg-05), and do nothing at all when there
    is no weather plugin to seed.
    """
    from alembic import command
    from alembic.config import Config as AlembicConfig

    await _reset_schema(_TEST_DB_URL)

    raw = {
        "server": {"transport": "websocket"},
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": "test-key"},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": "test-key",
            "models": [{"model": "fake-model"}],
        },
        "tts": {"url": "https://tts.invalid/v1/tts", "api_key": "test-key", "voice_id": "eve"},
        "mcp": {
            "servers": {
                "ha": {
                    "args": ["-m", "atlas_mcp.ha"],
                    "env": {
                        "HA_URL": "http://ha.invalid:8123",
                        "HA_TOKEN": "a-plainly-fictional-migration-test-token",
                    },
                },
            },
        },
        "database": {"url": _TEST_DB_URL},
        "security": {},
    }
    config_path = tmp_path / "weather-units-migration-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("ATLAS_CONFIG", str(config_path))
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    async def _weather_units_rows():
        engine = create_async_engine(_TEST_DB_URL)
        try:
            async with engine.connect() as conn:
                return (
                    await conn.execute(
                        text(
                            "SELECT v.secret, v.value, v.ciphertext "
                            "FROM plugin_config_values v JOIN plugins p ON p.id = v.plugin_id "
                            "WHERE p.slug = 'weather' AND v.key = 'WEATHER_UNITS'"
                        )
                    )
                ).fetchall()
        finally:
            await engine.dispose()

    def _downgrade_to_0012() -> None:
        cfg = AlembicConfig("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", _migration_url(_TEST_DB_URL))
        command.downgrade(cfg, "0012")

    # 1. Before 0013 runs: 0008 seeded the weather plugin, but no
    # WEATHER_UNITS row exists yet.
    _run_upgrade_to("0012")
    assert await _weather_units_rows() == []

    # 2. Upgrade to head: exactly one row, plain-stored "celsius".
    _run_upgrade_head()
    rows = await _weather_units_rows()
    assert len(rows) == 1
    assert rows[0].secret is False
    assert rows[0].value == "celsius"
    assert rows[0].ciphertext is None

    # 3. Downgrade: the seeded row is removed.
    _downgrade_to_0012()
    assert await _weather_units_rows() == []

    # 4. An operator's own saved value is never overwritten by a later
    # upgrade (T-lmg-05).
    engine = create_async_engine(_TEST_DB_URL)
    try:
        async with engine.begin() as conn:
            plugin_id = (
                await conn.execute(text("SELECT id FROM plugins WHERE slug = 'weather'"))
            ).scalar_one()
            await conn.execute(
                text(
                    "INSERT INTO plugin_config_values "
                    "(plugin_id, key, secret, value, ciphertext, key_version, updated_at) "
                    "VALUES (:plugin_id, 'WEATHER_UNITS', false, 'fahrenheit', NULL, NULL, now())"
                ),
                {"plugin_id": plugin_id},
            )
    finally:
        await engine.dispose()

    _run_upgrade_head()
    rows = await _weather_units_rows()
    assert len(rows) == 1, "the migration must keep the operator's row, never add a second one"
    assert rows[0].value == "fahrenheit", "an operator's saved value must never be overwritten"

    # 5. No weather plugin at all: the migration does nothing, and the
    # upgrade still succeeds.
    _downgrade_to_0012()
    engine = create_async_engine(_TEST_DB_URL)
    try:
        async with engine.begin() as conn:
            weather_plugin_id = (
                await conn.execute(text("SELECT id FROM plugins WHERE slug = 'weather'"))
            ).scalar_one()
            await conn.execute(
                text("DELETE FROM plugin_config_values WHERE plugin_id = :plugin_id"),
                {"plugin_id": weather_plugin_id},
            )
            await conn.execute(text("DELETE FROM plugins WHERE id = :plugin_id"), {"plugin_id": weather_plugin_id})
    finally:
        await engine.dispose()

    _run_upgrade_head()
    assert await _weather_units_rows() == []

    # 6. Idempotent, like every other migration in this file.
    _run_upgrade_head()
    assert await _weather_units_rows() == []


@skip_without_postgres
async def test_a_plugins_config_key_cannot_be_stored_twice(tmp_path, monkeypatch):
    """IN-03 (06-REVIEW.md): `set_config_values` is a read-then-upsert with
    no row lock, so without database-level uniqueness two concurrent saves
    for the same plugin could both miss the existing row and insert
    duplicates for one key -- after which the child silently received
    whichever row an unordered `SELECT` returned last. Migration `0010`
    makes that shape impossible to store, the same way `uq_plugins_slug`
    already does for the parent table.
    """
    from sqlalchemy.exc import IntegrityError

    await _reset_schema(_TEST_DB_URL)
    raw = {
        "server": {"transport": "websocket"},
        "stt": {"url": "wss://stt.invalid/v1/stt", "api_key": "test-key"},
        "brain": {
            "base_url": "https://brain.invalid/v1",
            "api_key": "test-key",
            "models": [{"model": "fake-model"}],
        },
        "tts": {"url": "https://tts.invalid/v1/tts", "api_key": "test-key", "voice_id": "eve"},
        "database": {"url": _TEST_DB_URL},
        "security": {},
    }
    config_path = tmp_path / "config-value-uniqueness-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")
    monkeypatch.setenv("ATLAS_CONFIG", str(config_path))
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)
    _run_upgrade_head()

    engine = create_async_engine(_TEST_DB_URL)
    try:
        async with engine.begin() as conn:
            plugin_id = (
                await conn.execute(text("SELECT id FROM plugins WHERE slug = 'ha'"))
            ).scalar_one()
            await conn.execute(
                text(
                    "INSERT INTO plugin_config_values "
                    "(plugin_id, key, secret, value, ciphertext, key_version, updated_at) "
                    "VALUES (:plugin_id, 'A_NEW_KEY', false, 'first', NULL, NULL, now())"
                ),
                {"plugin_id": plugin_id},
            )

        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO plugin_config_values "
                        "(plugin_id, key, secret, value, ciphertext, key_version, updated_at) "
                        "VALUES (:plugin_id, 'A_NEW_KEY', false, 'second', NULL, NULL, now())"
                    ),
                    {"plugin_id": plugin_id},
                )
    finally:
        await engine.dispose()


async def _fetch_provider_selection_rows(async_url: str) -> list[tuple[str, str]]:
    """`(slot, provider_name)` for every seeded `provider_selections` row,
    in id order -- migration `0011`'s own sibling of `_fetch_macro_rows`
    above."""
    engine = create_async_engine(async_url)
    try:
        async with engine.connect() as conn:
            rows = (
                await conn.execute(
                    text("SELECT slot, provider_name FROM provider_selections ORDER BY id")
                )
            ).fetchall()
        return [(r.slot, r.provider_name) for r in rows]
    finally:
        await engine.dispose()


@skip_without_postgres
async def test_provider_selections_seed_all_three_slots_with_xai_and_are_idempotent(
    monkeypatch,
):
    """Migration 0011 (D-01, D-03, PROV-01): `provider_selections` exists
    after an upgrade to head from empty, all three slots (`stt`, `tts`,
    `brain`) are seeded pointing at `"xai"` -- the one entry every registry
    holds this plan and the only implementation `config.example.yaml` has
    ever configured -- and running every migration twice is still a no-op
    the second time, the same idempotency guarantee this file's other
    migration tests already prove."""
    await _reset_schema(_TEST_DB_URL)
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    # Phase 7 (D-15): server.bind_host / security.cookie_secure -- the two
    # new ${VAR} placeholders config.example.yaml expands.
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()

    rows = await _fetch_provider_selection_rows(_TEST_DB_URL)
    assert rows == [("stt", "xai"), ("tts", "xai"), ("brain", "xai")]

    # Idempotent: a second upgrade to the same head must not error and
    # must not duplicate a single row.
    _run_upgrade_head()
    assert await _fetch_provider_selection_rows(_TEST_DB_URL) == rows


@skip_without_postgres
async def test_provider_selections_slot_is_unique(monkeypatch):
    """`uq_provider_selections_slot` (migration 0011): a second row for a
    slot that already has one must be refused at the database level, the
    same defense-in-depth `uq_plugins_slug` already gives `plugins.slug`."""
    from sqlalchemy.exc import IntegrityError

    await _reset_schema(_TEST_DB_URL)
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    # Phase 7 (D-15): server.bind_host / security.cookie_secure -- the two
    # new ${VAR} placeholders config.example.yaml expands.
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)
    _run_upgrade_head()

    engine = create_async_engine(_TEST_DB_URL)
    try:
        with pytest.raises(IntegrityError):
            async with engine.begin() as conn:
                await conn.execute(
                    text(
                        "INSERT INTO provider_selections "
                        "(slot, provider_name, options, updated_at) "
                        "VALUES ('stt', 'xai', '{}', now())"
                    )
                )
    finally:
        await engine.dispose()


@skip_without_postgres
async def test_upgrade_over_real_data_keeps_every_row_and_the_credential_still_decrypts(
    monkeypatch,
):
    """DEP-04, T-07-33, T-07-35: the second half of DEP-04 this project has
    never proven -- that an upgrade over a database an operator already
    wrote real data into keeps that data. Starts from `0010`, the revision
    that preceded this phase, writes one row through each of the real
    repositories an operator's data actually arrives through (a safety
    policy rule, a stored credential, an account, a plugin configuration
    value), upgrades to head, and reads every one of them back through the
    same repositories -- byte-identical, including the timezone handling
    the repository boundary normalizes. The encrypted credential is
    checked the concrete way that actually protects an operator: it must
    still decrypt, with the same key, after the upgrade. `provider_selections`
    (migration `0011`) must exist and be seeded. A second upgrade to the
    same head, and a downgrade of `0011` followed by a re-upgrade, must
    both leave every one of these rows exactly as they were.
    """
    from alembic import command
    from alembic.config import Config as AlembicConfig
    from sqlalchemy.ext.asyncio import async_sessionmaker

    from atlas.config import SecurityConfig
    from atlas.crypto.credentials import decrypt_credential, encrypt_credential
    from atlas.db.engine import get_current_revision, run_migrations
    from atlas.db.postgres import (
        PostgresAccountRepository,
        PostgresCredentialRepository,
        PostgresPluginRepository,
        PostgresPolicyRepository,
    )
    from atlas.db.repository import PluginConfigValue

    await _reset_schema(_TEST_DB_URL)
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    # Phase 7 (D-15): server.bind_host / security.cookie_secure -- the two
    # new ${VAR} placeholders config.example.yaml expands.
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    test_secret_key = "test-secret-key-not-a-real-generated-value"
    monkeypatch.setenv("ATLAS_SECRET_KEY", test_secret_key)
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    migration_url = _migration_url(_TEST_DB_URL)

    # Stand where a real deployment stood immediately before this phase's
    # own migration existed.
    _run_upgrade_to("0010")
    assert get_current_revision(migration_url) == "0010"

    engine = create_async_engine(_TEST_DB_URL)
    try:
        sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
        security = SecurityConfig()

        policy_repo = PostgresPolicyRepository(sessionmaker)
        credential_repo = PostgresCredentialRepository(sessionmaker)
        account_repo = PostgresAccountRepository(sessionmaker)
        plugin_repo = PostgresPluginRepository(sessionmaker)

        # A safety policy rule, written the way the denylist editor writes one.
        written_rule = await policy_repo.add_rule(
            kind="deny_entity",
            value="switch.example_pre_upgrade_socket",
            note="written before the 07-07 upgrade",
            created_by_user_id=None,
        )

        # A stored, encrypted credential -- the row this task's own
        # behaviour singles out: surviving structurally but no longer
        # decrypting is the same loss as not surviving at all.
        credential_plaintext = "a-plainly-fictional-pre-upgrade-credential"
        ciphertext, key_version = encrypt_credential(credential_plaintext, security)
        written_credential = await credential_repo.upsert_credential(
            "stt_api_key", ciphertext=ciphertext, key_version=key_version, updated_by_user_id=None
        )

        # An account.
        written_user = await account_repo.create_user(
            email="pre-upgrade@example.invalid",
            display_name="Pre Upgrade Operator",
            password_hash="not-a-real-hash",
            role="admin",
        )

        # A plugin configuration value, on the builtin `ha` row migration
        # 0008 always seeds.
        [ha_plugin] = [p for p in await plugin_repo.list_plugins() if p.slug == "ha"]
        await plugin_repo.set_config_values(
            ha_plugin.id,
            [
                PluginConfigValue(
                    key="EXAMPLE_PRE_UPGRADE_KEY",
                    secret=False,
                    value="pre-upgrade-value",
                    ciphertext=None,
                    key_version=None,
                )
            ],
        )
        written_config_values = {v.key: v for v in await plugin_repo.get_config_values(ha_plugin.id)}

        # The real migration runner `lifespan` calls -- not a fake, not a
        # second reimplementation of it (the plan's own key link).
        run_migrations(migration_url)
        assert get_current_revision(migration_url) == "0013"

        async def _assert_pre_upgrade_rows_intact() -> list[tuple[str, str]]:
            reread_rules = {r.id: r for r in await policy_repo.list_rules()}
            assert reread_rules[written_rule.id] == written_rule

            reread_credential = await credential_repo.get_credential("stt_api_key")
            assert reread_credential == written_credential
            assert (
                decrypt_credential(
                    reread_credential.ciphertext, reread_credential.key_version, security
                )
                == credential_plaintext
            ), "the credential survived the schema change but no longer decrypts -- the WR-06-shaped loss this task exists to catch"

            reread_user = await account_repo.get_user_by_id(written_user.id)
            assert reread_user == written_user

            reread_config_values = {
                v.key: v for v in await plugin_repo.get_config_values(ha_plugin.id)
            }
            assert reread_config_values == written_config_values

            return await _fetch_provider_selection_rows(_TEST_DB_URL)

        provider_rows = await _assert_pre_upgrade_rows_intact()
        assert provider_rows == [("stt", "xai"), ("tts", "xai"), ("brain", "xai")]

        # A second run at head: no-op. The stamped revision is unchanged
        # and nothing is added, removed, or rewritten -- old data or new.
        run_migrations(migration_url)
        assert get_current_revision(migration_url) == "0013"
        assert await _assert_pre_upgrade_rows_intact() == provider_rows

        # A downgrade of this phase's own migration, and a re-upgrade,
        # return to the same state -- for the table the migration owns,
        # and for every row a prior migration's real repository wrote,
        # which this downgrade has no relationship to at all.
        cfg = AlembicConfig("alembic.ini")
        cfg.set_main_option("sqlalchemy.url", migration_url)
        command.downgrade(cfg, "0010")
        assert get_current_revision(migration_url) == "0010"

        run_migrations(migration_url)
        assert get_current_revision(migration_url) == "0013"
        assert await _assert_pre_upgrade_rows_intact() == provider_rows
    finally:
        await engine.dispose()
