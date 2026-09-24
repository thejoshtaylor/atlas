"""A defect discovered incidentally while fixing WR-03 (code review), not
named by the review itself: every `Mapped[datetime]` column
`atlas.db.models` declares maps, with no `timezone=True`, to
Postgres' `TIMESTAMP WITHOUT TIME ZONE` -- and every datetime this
project's runtime code has ever written is `datetime.now(timezone.utc)`,
timezone-*aware*. Nothing in this suite exercised a real write through the
asyncpg driver before WR-03's own concurrency test needed one: every prior
test either drove `FakeAccountRepository` (plain Python, no column type at
all) or exercised a write through Alembic's *synchronous* `psycopg` driver
(`tests/test_db_migrations.py`), which tolerates the aware-into-naive
mismatch silently. asyncpg does not -- it raises `DataError: can't
subtract offset-naive and offset-aware datetimes` on the very first
insert, which is exactly what surfaced while writing WR-03's own real-
Postgres test.

`atlas/db/postgres.py` now converts every datetime at its own
database boundary (`_to_naive_utc` on write, `_to_aware_utc` on read) --
this file proves that conversion holds for every repository class in that
module, not only `PostgresAccountRepository` (WR-03's own concurrency test
already covers that one directly). Each test below is a real write
followed by a real read-back against a reachable Postgres, asserting no
exception and that the round-tripped value is both timezone-aware and
equal to the microsecond.

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_db_migrations.py`'s own precedent.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.db.postgres import (
    PostgresCredentialRepository,
    PostgresPolicyRepository,
    PostgresSettingsRepository,
    PostgresSetupRepository,
)

_TEST_DB_URL = os.environ.get("ATLAS_TEST_DATABASE_URL")

pytestmark = pytest.mark.integration

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "ATLAS_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these tests instead of skipping them"
    ),
)


def _migration_url(async_url: str) -> str:
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
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


def _run_upgrade_head(async_url: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, "head")


@pytest.fixture
async def sessionmaker(monkeypatch):
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
    # Plan 04-03: config.example.yaml's mcp.servers.weather block adds two
    # more ${...} placeholders this real-file load must expand too.
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_head(_TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@skip_without_postgres
async def test_policy_rule_datetimes_round_trip_aware(sessionmaker):
    repo = PostgresPolicyRepository(sessionmaker)
    before = datetime.now(timezone.utc)

    rule = await repo.add_rule(
        kind="deny_entity", value="switch.example_datetime_roundtrip", note=None, created_by_user_id=None
    )

    assert rule.created_at.tzinfo is not None, "add_rule's return value must be timezone-aware"
    assert rule.created_at >= before

    [listed] = [r for r in await repo.list_rules() if r.id == rule.id]
    assert listed.created_at.tzinfo is not None, "list_rules' return value must be timezone-aware"
    assert listed.created_at == rule.created_at


@skip_without_postgres
async def test_setup_step_datetimes_round_trip_aware(sessionmaker):
    repo = PostgresSetupRepository(sessionmaker)
    now = datetime.now(timezone.utc)

    step = await repo.complete_step("hub", detail={"entity_count": 3}, completed_at=now)

    assert step.completed_at.tzinfo is not None, "complete_step's return value must be timezone-aware"
    assert step.completed_at == now

    reread = await repo.get_step("hub")
    assert reread is not None
    assert reread.completed_at.tzinfo is not None, "get_step's return value must be timezone-aware"
    assert reread.completed_at == now


@skip_without_postgres
async def test_setting_datetimes_round_trip_aware(sessionmaker):
    repo = PostgresSettingsRepository(sessionmaker)
    now = datetime.now(timezone.utc)

    setting = await repo.set_setting("audio_source", "camera", updated_by_user_id=None, updated_at=now)

    assert setting.updated_at.tzinfo is not None, "set_setting's return value must be timezone-aware"
    assert setting.updated_at == now

    reread = await repo.get_setting("audio_source")
    assert reread is not None
    assert reread.updated_at.tzinfo is not None, "get_setting's return value must be timezone-aware"
    assert reread.updated_at == now


@skip_without_postgres
async def test_credential_datetimes_round_trip_aware(sessionmaker):
    repo = PostgresCredentialRepository(sessionmaker)
    before = datetime.now(timezone.utc)

    credential = await repo.upsert_credential(
        "stt_api_key", ciphertext=b"not-real-ciphertext", key_version=1, updated_by_user_id=None
    )

    assert credential.updated_at.tzinfo is not None, "upsert_credential's return value must be timezone-aware"
    assert credential.updated_at >= before

    reread = await repo.get_credential("stt_api_key")
    assert reread is not None
    assert reread.updated_at.tzinfo is not None, "get_credential's return value must be timezone-aware"
    assert reread.updated_at == credential.updated_at
