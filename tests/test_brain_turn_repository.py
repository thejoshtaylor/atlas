"""`PostgresBrainTurnRepository` against a real Postgres (quick task 261001-mp8).

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_edge_device_repository.py`.
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.db.brain_turn_postgres import PostgresBrainTurnRepository
from atlas.db.brain_turn_repository import BrainTurnEntry

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
    yield PostgresBrainTurnRepository(async_sessionmaker(engine, expire_on_commit=False))
    await engine.dispose()


def _entry(turn_id: str, created_at: datetime, **overrides) -> BrainTurnEntry:
    values = dict(
        turn_id=turn_id,
        created_at=created_at,
        transcript="Hey Atlas, turn on the example lamp.",
        normalized_transcript="turn on the example lamp",
        transcript_fingerprint="abc123",
        continuation=False,
        tool_calls=(
            {"name": "ha_get_state", "arguments": {"entity_id": "light.example_lamp"}, "kind": "read", "ok": True, "error": None},
            {
                "name": "ha_call_service",
                "arguments": {"service": "turn_on", "entity_id": "light.example_lamp"},
                "kind": "write",
                "ok": False,
                "error": "boom",
            },
        ),
        reply_text="Done.",
        tier_index=0,
        tier_model="example-model",
        brain_latency_ms=1250,
        outcome="completed",
    )
    values.update(overrides)
    return BrainTurnEntry(**values)


@skip_without_postgres
async def test_a_brain_turn_round_trips_with_ordered_tool_calls(repo):
    entry = _entry("t1", NOW)

    await repo.record_brain_turn(entry)
    stored = await repo.list_brain_turns()

    assert len(stored) == 1
    assert stored[0].id > 0
    assert stored[0].turn_id == "t1"
    assert stored[0].created_at == NOW
    assert stored[0].tool_calls == entry.tool_calls
    assert [c["name"] for c in stored[0].tool_calls] == ["ha_get_state", "ha_call_service"]
    assert stored[0].reply_text == "Done."
    assert stored[0].tier_index == 0
    assert stored[0].tier_model == "example-model"
    assert stored[0].brain_latency_ms == 1250
    assert stored[0].outcome == "completed"
    assert stored[0].continuation is False


@skip_without_postgres
async def test_list_brain_turns_returns_newest_first_and_honors_a_limit(repo):
    await repo.record_brain_turn(_entry("old", NOW - timedelta(hours=2)))
    await repo.record_brain_turn(_entry("new", NOW))
    await repo.record_brain_turn(_entry("mid", NOW - timedelta(hours=1)))

    assert [t.turn_id for t in await repo.list_brain_turns()] == ["new", "mid", "old"]
    assert [t.turn_id for t in await repo.list_brain_turns(limit=2)] == ["new", "mid"]


@skip_without_postgres
async def test_delete_brain_turns_before_removes_only_older_rows(repo):
    await repo.record_brain_turn(_entry("old", NOW - timedelta(days=10)))
    await repo.record_brain_turn(_entry("older", NOW - timedelta(days=20)))
    await repo.record_brain_turn(_entry("fresh", NOW - timedelta(days=1)))

    removed = await repo.delete_brain_turns_before(NOW - timedelta(days=7))

    assert removed == 2
    assert [t.turn_id for t in await repo.list_brain_turns()] == ["fresh"]
    assert await repo.delete_brain_turns_before(NOW - timedelta(days=7)) == 0
