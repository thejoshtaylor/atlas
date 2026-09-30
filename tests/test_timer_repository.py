"""`PostgresTimerRepository` against a real Postgres (quick task 260930-06x).

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_edge_device_repository.py`.
"""

from __future__ import annotations

import os
from dataclasses import replace
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.db.timer_postgres import PostgresTimerRepository

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

NOW = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)


def _migration_url(async_url: str) -> str:
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
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
async def repo(monkeypatch):
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)
    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_head(_TEST_DB_URL)
    engine = create_async_engine(_TEST_DB_URL)
    yield PostgresTimerRepository(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()


async def _timer(repo, *, label="pasta", due_in=60):
    return await repo.create_timer(
        kind="timer",
        label=label,
        due_at=NOW + timedelta(seconds=due_in),
        remaining_s=None,
        duration_s=due_in,
        time_of_day=None,
        repeat_days=0,
        enabled=True,
        created_at=NOW,
    )


async def _alarm(repo, *, label="tea", repeat_days=0, due_in=60):
    return await repo.create_timer(
        kind="alarm",
        label=label,
        due_at=NOW + timedelta(seconds=due_in),
        remaining_s=None,
        duration_s=None,
        time_of_day="07:00",
        repeat_days=repeat_days,
        enabled=True,
        created_at=NOW,
    )


@skip_without_postgres
async def test_create_get_list_save_delete_round_trip(repo):
    timer = await _timer(repo)
    assert timer.due_at == NOW + timedelta(seconds=60)
    assert timer.created_at == NOW
    assert await repo.get_timer(timer.id) == timer
    assert await repo.get_timer(999999) is None

    paused = replace(timer, due_at=None, remaining_s=42, label="tea")
    assert await repo.save_timer(paused) == paused
    assert await repo.list_timers() == [paused]

    assert await repo.delete_timer(timer.id) is True
    assert await repo.delete_timer(timer.id) is False
    assert await repo.save_timer(paused) is None
    assert await repo.list_timers() == []


@skip_without_postgres
async def test_advance_fired_deletes_a_due_timer_once(repo):
    timer = await _timer(repo)
    due = NOW + timedelta(seconds=60)

    assert await repo.advance_fired(timer.id, now=due, next_due_at=None) is True
    assert await repo.get_timer(timer.id) is None
    assert await repo.advance_fired(timer.id, now=due, next_due_at=None) is False


@skip_without_postgres
async def test_advance_fired_turns_a_one_time_alarm_off(repo):
    alarm = await _alarm(repo)

    assert await repo.advance_fired(alarm.id, now=NOW + timedelta(seconds=60), next_due_at=None) is True

    after = await repo.get_timer(alarm.id)
    assert after is not None
    assert after.enabled is False
    assert after.due_at is None


@skip_without_postgres
async def test_advance_fired_moves_a_repeating_alarm(repo):
    alarm = await _alarm(repo, repeat_days=0b0011111)
    next_due = NOW + timedelta(days=1)

    assert await repo.advance_fired(alarm.id, now=NOW + timedelta(seconds=60), next_due_at=next_due) is True

    after = await repo.get_timer(alarm.id)
    assert after is not None
    assert after.enabled is True
    assert after.due_at == next_due


@skip_without_postgres
async def test_advance_fired_refuses_not_due_disabled_and_missing(repo):
    timer = await _timer(repo)
    assert await repo.advance_fired(timer.id, now=NOW, next_due_at=None) is False

    alarm = await _alarm(repo)
    await repo.save_timer(replace(alarm, enabled=False, due_at=None))
    assert await repo.advance_fired(alarm.id, now=NOW + timedelta(days=1), next_due_at=None) is False

    assert await repo.advance_fired(999999, now=NOW, next_due_at=None) is False
