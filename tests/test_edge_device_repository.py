"""`PostgresEdgeDeviceRepository` against a real Postgres (D-03, Phase 10,
plan 10-04) -- the `<behavior>` cases from `10-04-PLAN.md`'s Task 1,
proven against real Postgres rather than the in-memory
`FakeEdgeDeviceRepository` (`tests/edge_fakes.py`) `tests/test_edge_tracer.py`
and `tests/test_edge_source.py` already drive.

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_db_migrations.py`'s own precedent. Every hash literal here is
built by calling `hash_edge_token(...)`, never a hard-coded string, so
`tests/test_repo_hygiene.py`'s credential scan has nothing to match.
"""

from __future__ import annotations

import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.auth.edge_tokens import hash_edge_token
from atlas.db.edge_postgres import PostgresEdgeDeviceRepository

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


def _run_upgrade_head(async_url: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, "head")


async def _create_admin_user(sessionmaker) -> int:
    """One `users` row to satisfy `edge_devices.created_by_user_id`'s
    foreign key -- a plain insert, not `PostgresAccountRepository`, since
    this file's only interest in the row is its id."""
    async with sessionmaker() as session:
        result = await session.execute(
            text(
                "INSERT INTO users (email, display_name, password_hash, role, created_at) "
                "VALUES (:email, 'Test Admin', 'not-a-real-hash', 'admin', :created_at) "
                "RETURNING id"
            ),
            # `users.created_at` is naive-UTC (`db/postgres.py`'s own
            # module-wide convention) -- strip tzinfo, matching
            # `_to_naive_utc`, since this raw insert bypasses that helper.
            {
                "email": "edge-repo-test-admin@example.invalid",
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            },
        )
        user_id = result.scalar_one()
        await session.commit()
        return int(user_id)


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
    monkeypatch.setenv("BIND_HOST", "127.0.0.1")
    monkeypatch.setenv("COOKIE_SECURE", "false")
    monkeypatch.setenv("HA_URL", "test-value")
    monkeypatch.setenv("HA_TOKEN", "test-value")
    monkeypatch.setenv("WEATHER_LATITUDE", "0.0")
    monkeypatch.setenv("WEATHER_LONGITUDE", "0.0")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_head(_TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@skip_without_postgres
async def test_create_device_stores_the_hash_it_is_given_with_no_revocation(sessionmaker):
    repo = PostgresEdgeDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    now = datetime.now(timezone.utc)

    device = await repo.create_device(
        name="kitchen-pi", token_hash=hash_edge_token("t-1"), created_by_user_id=admin_id, created_at=now
    )

    assert device.revoked_at is None
    assert device.last_connected_at is None
    assert device.created_by_user_id == admin_id
    assert device.created_at == now


@skip_without_postgres
async def test_get_active_by_token_hash_finds_the_device_and_misses_an_unknown_hash(sessionmaker):
    repo = PostgresEdgeDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)

    device = await repo.create_device(
        name="kitchen-pi",
        token_hash=hash_edge_token("t-1"),
        created_by_user_id=admin_id,
        created_at=datetime.now(timezone.utc),
    )

    found = await repo.get_active_by_token_hash(hash_edge_token("t-1"))
    assert found is not None
    assert found.id == device.id

    assert await repo.get_active_by_token_hash(hash_edge_token("t-unknown")) is None


@skip_without_postgres
async def test_get_active_by_token_hash_misses_a_revoked_device(sessionmaker):
    repo = PostgresEdgeDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)

    device = await repo.create_device(
        name="kitchen-pi",
        token_hash=hash_edge_token("t-1"),
        created_by_user_id=admin_id,
        created_at=datetime.now(timezone.utc),
    )
    assert await repo.revoke_device(device.id, revoked_at=datetime.now(timezone.utc)) is True

    assert await repo.get_active_by_token_hash(hash_edge_token("t-1")) is None


@skip_without_postgres
async def test_revoke_device_sets_revoked_at_once_and_a_second_call_returns_false(sessionmaker):
    repo = PostgresEdgeDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)

    device = await repo.create_device(
        name="kitchen-pi",
        token_hash=hash_edge_token("t-1"),
        created_by_user_id=admin_id,
        created_at=datetime.now(timezone.utc),
    )
    revoked_at = datetime.now(timezone.utc)

    assert await repo.revoke_device(device.id, revoked_at=revoked_at) is True
    assert await repo.revoke_device(device.id, revoked_at=datetime.now(timezone.utc)) is False

    [listed] = await repo.list_devices()
    assert listed.revoked_at is not None
    assert listed.revoked_at == revoked_at


@skip_without_postgres
async def test_revoke_device_on_an_unknown_id_returns_false(sessionmaker):
    repo = PostgresEdgeDeviceRepository(sessionmaker)
    assert await repo.revoke_device(999999, revoked_at=datetime.now(timezone.utc)) is False


@skip_without_postgres
async def test_mark_connected_sets_last_connected_at(sessionmaker):
    repo = PostgresEdgeDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)

    device = await repo.create_device(
        name="kitchen-pi",
        token_hash=hash_edge_token("t-1"),
        created_by_user_id=admin_id,
        created_at=datetime.now(timezone.utc),
    )
    at = datetime.now(timezone.utc)
    await repo.mark_connected(device.id, at=at)

    [listed] = await repo.list_devices()
    assert listed.last_connected_at == at


@skip_without_postgres
async def test_list_devices_returns_every_device_revoked_included_oldest_first(sessionmaker):
    repo = PostgresEdgeDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)

    first = await repo.create_device(
        name="first-pi",
        token_hash=hash_edge_token("t-1"),
        created_by_user_id=admin_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    second = await repo.create_device(
        name="second-pi",
        token_hash=hash_edge_token("t-2"),
        created_by_user_id=admin_id,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    await repo.revoke_device(first.id, revoked_at=datetime.now(timezone.utc))

    listed = await repo.list_devices()
    assert [d.id for d in listed] == [first.id, second.id]
    assert listed[0].revoked_at is not None


@skip_without_postgres
async def test_a_second_create_device_with_the_same_hash_raises_unique_violation(sessionmaker):
    from sqlalchemy.exc import IntegrityError

    repo = PostgresEdgeDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    token_hash = hash_edge_token("t-1")

    await repo.create_device(
        name="kitchen-pi", token_hash=token_hash, created_by_user_id=admin_id, created_at=datetime.now(timezone.utc)
    )

    with pytest.raises(IntegrityError):
        await repo.create_device(
            name="another-pi",
            token_hash=token_hash,
            created_by_user_id=admin_id,
            created_at=datetime.now(timezone.utc),
        )
