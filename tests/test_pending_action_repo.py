"""`PostgresPendingActionRepository` against a real Postgres (T-09-25).

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_wake_events.py`'s own precedent -- the direct sibling this file
follows for its `sessionmaker` fixture (reset schema, migrate to head, hand
back a fresh `async_sessionmaker`).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.db.pending_action_repository import PostgresPendingActionRepository

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
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def _run_upgrade_to(async_url: str, revision: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, revision)


@pytest.fixture
async def sessionmaker(monkeypatch):
    """`tests/test_wake_events.py`'s own `sessionmaker` fixture, unchanged
    -- reset schema, migrate to head, hand back a fresh `async_sessionmaker`."""
    monkeypatch.setenv("ATLAS_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    monkeypatch.setenv("CALIBRATION_ROUTE_ENABLED", "false")
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_to(_TEST_DB_URL, "head")

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _now() -> datetime:
    return datetime(2026, 9, 24, 12, 0, 0, tzinfo=timezone.utc)


@skip_without_postgres
async def test_create_supersedes_the_same_sources_earlier_awaiting_row(sessionmaker):
    repo = PostgresPendingActionRepository(sessionmaker)
    now = _now()

    first = await repo.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={"title": "Dentist"},
        readback="add Dentist...",
        created_at=now,
        expires_at=now + timedelta(seconds=60),
    )
    second = await repo.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={"title": "Vet"},
        readback="add Vet...",
        created_at=now + timedelta(seconds=5),
        expires_at=now + timedelta(seconds=65),
    )

    reread_first = await repo.get(first.id)
    assert reread_first is not None
    assert reread_first.status == "superseded"
    assert reread_first.resolved_at is not None

    reread_second = await repo.get(second.id)
    assert reread_second is not None
    assert reread_second.status == "awaiting"


@skip_without_postgres
async def test_create_leaves_another_sources_row_alone(sessionmaker):
    repo = PostgresPendingActionRepository(sessionmaker)
    now = _now()

    camera_row = await repo.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={},
        readback="camera readback",
        created_at=now,
        expires_at=now + timedelta(seconds=60),
    )
    await repo.create(
        source="browser",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={},
        readback="browser readback",
        created_at=now,
        expires_at=now + timedelta(seconds=60),
    )

    reread_camera = await repo.get(camera_row.id)
    assert reread_camera is not None
    assert reread_camera.status == "awaiting"


@skip_without_postgres
async def test_claim_for_confirmation_succeeds_exactly_once(sessionmaker):
    repo = PostgresPendingActionRepository(sessionmaker)
    now = _now()

    row = await repo.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={},
        readback="readback",
        created_at=now,
        expires_at=now + timedelta(seconds=60),
    )

    first_claim = await repo.claim_for_confirmation(row.id, now)
    assert first_claim is not None
    assert first_claim.status == "confirmed"

    second_claim = await repo.claim_for_confirmation(row.id, now)
    assert second_claim is None


@skip_without_postgres
async def test_claim_for_confirmation_returns_none_after_expiry(sessionmaker):
    repo = PostgresPendingActionRepository(sessionmaker)
    now = _now()

    row = await repo.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={},
        readback="readback",
        created_at=now,
        expires_at=now + timedelta(seconds=1),
    )

    claim = await repo.claim_for_confirmation(row.id, now + timedelta(seconds=5))
    assert claim is None


@skip_without_postgres
async def test_resolve_records_status_detail_and_time(sessionmaker):
    repo = PostgresPendingActionRepository(sessionmaker)
    now = _now()

    row = await repo.create(
        source="camera",
        action="calendar_create",
        tool_name="calendar_insert_event",
        arguments={},
        readback="readback",
        created_at=now,
        expires_at=now + timedelta(seconds=60),
    )

    resolved_at = now + timedelta(seconds=2)
    await repo.resolve(row.id, "executed", "added it", resolved_at)

    reread = await repo.get(row.id)
    assert reread is not None
    assert reread.status == "executed"
    assert reread.result_detail == "added it"
    assert reread.resolved_at == resolved_at
