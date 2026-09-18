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
are new: `pytest.mark.skipif` reads `SPIRE_TEST_DATABASE_URL` from the
environment and skips, naming that variable, when it is absent -- matching
D-04's "the suite runs with no Postgres reachable at all." To make these two
tests run instead of skip: `eval "$(scripts/dev-postgres.sh)"` (starts a
throwaway local Postgres and exports both `DATABASE_URL` and
`SPIRE_TEST_DATABASE_URL`), then re-run the suite.

Neither test touches an operator's real configuration file or real database.
Both write their own, invented `safety:` block to a temporary file and point
`SPIRE_CONFIG` at that -- the same `switch.example_*`/`light.example_*`
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

_TEST_DB_URL = os.environ.get("SPIRE_TEST_DATABASE_URL")

pytestmark = pytest.mark.integration

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "SPIRE_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these two tests instead of skipping them"
    ),
)


def _migration_url(async_url: str) -> str:
    """The synchronous `psycopg` connection string Alembic needs, derived
    the same way `spire_voice.config.DatabaseConfig.migration_url` derives
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
        for table in (
            # Plan 05-01: 0006 adds these two -- workflow_steps first, it
            # holds the foreign key onto workflow_runs.
            "workflow_steps",
            "workflow_runs",
            "macro_actions",
            "macro_aliases",
            "macros",
            "settings",
            "setup_steps",
            "setup_state",
            "provider_credentials",
            "refresh_tokens",
            "invites",
            "users",
            "policy_rules",
            "safety_policy",
            "audit_log",
            "alembic_version",
        ):
            await conn.execute(text(f"DROP TABLE IF EXISTS {table} CASCADE"))
    await engine.dispose()


def _run_upgrade_head() -> None:
    """Run every migration up to `head` against `SPIRE_TEST_DATABASE_URL`,
    reading `SPIRE_CONFIG` (already set by the caller) the same way `app.py`
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


@skip_without_postgres
async def test_migrations_run_from_empty_and_are_idempotent(monkeypatch):
    """Running every migration twice against the same database, from empty,
    must succeed both times with no duplicated row and no error.

    `config/config.example.yaml` carries no `macros:` block (D-09: the key
    is retired), so migration `0005`'s own seed step seeds nothing against
    it -- this test's macro assertion proves the "absent key" case, the
    same way its policy assertion already proves `safety:`'s "absent key"
    case."""
    await _reset_schema(_TEST_DB_URL)
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    # Plan 04-03: config.example.yaml's mcp.servers.weather block adds two
    # more ${...} placeholders this real-file load must expand too.
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()
    first_policy, first_rules = await _fetch_rows(_TEST_DB_URL)
    assert first_policy == [(1, "allow_all_except_denylist")]
    assert first_rules == []
    assert await _fetch_macro_rows(_TEST_DB_URL) == []

    # Idempotent: Alembic's own applied-revision bookkeeping means a second
    # upgrade to the same head is a no-op, not a second seed.
    _run_upgrade_head()
    second_policy, second_rules = await _fetch_rows(_TEST_DB_URL)
    assert second_policy == first_policy
    assert second_rules == first_rules
    assert await _fetch_macro_rows(_TEST_DB_URL) == []


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
                    "args": ["-m", "spire_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-key"},
                },
            },
        },
        "database": {"url": _TEST_DB_URL},
        "safety": safety_block,
    }
    config_path = tmp_path / "seed-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("SPIRE_CONFIG", str(config_path))
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
                    "args": ["-m", "spire_mcp.ha"],
                    "env": {"HA_URL": "http://ha.invalid", "HA_TOKEN": "test-key"},
                },
            },
        },
        "database": {"url": _TEST_DB_URL},
        "macros": macros_block,
    }
    config_path = tmp_path / "macro-seed-test-config.yaml"
    config_path.write_text(yaml.safe_dump(raw), encoding="utf-8")

    monkeypatch.setenv("SPIRE_CONFIG", str(config_path))
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
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
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
