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
    `0003_credentials.py` adds `provider_credentials`.
    """
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        for table in (
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


@skip_without_postgres
async def test_migrations_run_from_empty_and_are_idempotent(monkeypatch):
    """Running every migration twice against the same database, from empty,
    must succeed both times with no duplicated row and no error."""
    await _reset_schema(_TEST_DB_URL)
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    _run_upgrade_head()
    first_policy, first_rules = await _fetch_rows(_TEST_DB_URL)
    assert first_policy == [(1, "allow_all_except_denylist")]
    assert first_rules == []

    # Idempotent: Alembic's own applied-revision bookkeeping means a second
    # upgrade to the same head is a no-op, not a second seed.
    _run_upgrade_head()
    second_policy, second_rules = await _fetch_rows(_TEST_DB_URL)
    assert second_policy == first_policy
    assert second_rules == first_rules


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
