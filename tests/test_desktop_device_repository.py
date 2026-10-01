"""`PostgresDesktopDeviceRepository` against a real Postgres (Phase 14, plan
14-04; D-01, D-06, D-14, D-28) -- the rules the in-memory fake in
`tests/desktop_fakes.py` already proves, proven again against the real
store, plus the concurrent "set default" race the database itself must win.

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_edge_device_repository.py`. Every hash is built by calling
`hash_desktop_token(...)`, never a hard-coded string, so
`tests/test_repo_hygiene.py`'s credential scan has nothing to match.
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from atlas.auth.desktop_tokens import hash_desktop_token
from atlas.db.desktop_postgres import PostgresDesktopDeviceRepository
from atlas.db.desktop_repository import (
    DesktopDefaultConflict,
    DesktopDeviceChanges,
    DesktopDeviceNameTaken,
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
    """One `users` row to satisfy `desktop_devices.created_by_user_id`'s
    foreign key -- a plain insert, since this file only needs its id."""
    async with sessionmaker() as session:
        result = await session.execute(
            text(
                "INSERT INTO users (email, display_name, password_hash, role, created_at) "
                "VALUES (:email, 'Test Admin', 'not-a-real-hash', 'admin', :created_at) "
                "RETURNING id"
            ),
            {
                "email": "desktop-repo-test-admin@example.invalid",
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
            },
        )
        user_id = result.scalar_one()
        await session.commit()
        return int(user_id)


async def _create_edge_device(sessionmaker, admin_id: int, name: str = "kitchen-pi") -> int:
    async with sessionmaker() as session:
        result = await session.execute(
            text(
                "INSERT INTO edge_devices (name, token_hash, created_at, created_by_user_id) "
                "VALUES (:name, :token_hash, :created_at, :admin_id) RETURNING id"
            ),
            {
                "name": name,
                "token_hash": f"edge-hash-for-{name}",
                "created_at": datetime.now(timezone.utc).replace(tzinfo=None),
                "admin_id": admin_id,
            },
        )
        edge_id = result.scalar_one()
        await session.commit()
        return int(edge_id)


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
    monkeypatch.setenv("ATLAS_SECRET_KEY", "test-secret-key-not-a-real-generated-value")
    monkeypatch.setenv("DATABASE_URL", _TEST_DB_URL)

    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_head(_TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


async def _make(repo, admin_id: int, name: str, token: str):
    return await repo.create_device(
        name=name,
        token_hash=hash_desktop_token(token),
        created_by_user_id=admin_id,
        created_at=datetime.now(timezone.utc),
    )


async def _default_count(sessionmaker) -> int:
    async with sessionmaker() as session:
        return int(
            (await session.execute(text("SELECT count(*) FROM desktop_devices WHERE is_default"))).scalar_one()
        )


@skip_without_postgres
async def test_create_then_read_back_with_clean_defaults(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    now = datetime.now(timezone.utc)

    device = await repo.create_device(
        name="desk-mac", token_hash=hash_desktop_token("t-1"), created_by_user_id=admin_id, created_at=now
    )

    assert device.is_default is False
    assert device.last_seen_at is None
    assert device.edge_device_id is None
    assert device.revoked_at is None
    assert device.created_by_user_id == admin_id
    assert device.created_at == now
    assert device.created_at.tzinfo is not None

    assert await repo.get_device(device.id) == device
    assert await repo.list_devices() == [device]
    found = await repo.get_active_by_token_hash(hash_desktop_token("t-1"))
    assert found is not None and found.id == device.id
    assert await repo.get_active_by_token_hash(hash_desktop_token("t-unknown")) is None
    assert await repo.get_device(999999) is None


@skip_without_postgres
async def test_active_names_are_unique_ignoring_case_and_a_revoked_mac_frees_its_name(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)

    first = await _make(repo, admin_id, "Desk-Mac", "t-1")
    with pytest.raises(DesktopDeviceNameTaken):
        await _make(repo, admin_id, "desk-mac", "t-2")

    assert await repo.revoke_device(first.id, revoked_at=datetime.now(timezone.utc)) is True
    second = await _make(repo, admin_id, "desk-mac", "t-2")
    assert second.name == "desk-mac"


@skip_without_postgres
async def test_a_duplicate_token_hash_still_raises_integrity_error(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)

    await _make(repo, admin_id, "desk-mac", "t-1")
    with pytest.raises(IntegrityError):
        await _make(repo, admin_id, "other-mac", "t-1")


@skip_without_postgres
async def test_update_renames_and_a_clash_raises_name_taken(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    now = datetime.now(timezone.utc)
    a = await _make(repo, admin_id, "desk-mac", "t-1")
    await _make(repo, admin_id, "laptop-mac", "t-2")

    renamed = await repo.update_device(a.id, DesktopDeviceChanges(name="study-mac"), at=now)
    assert renamed is not None and renamed.name == "study-mac"

    with pytest.raises(DesktopDeviceNameTaken):
        await repo.update_device(a.id, DesktopDeviceChanges(name="LAPTOP-MAC"), at=now)
    unchanged = await repo.get_device(a.id)
    assert unchanged is not None and unchanged.name == "study-mac"


@skip_without_postgres
async def test_update_maps_clears_and_leaves_the_edge_device(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    edge_id = await _create_edge_device(sessionmaker, admin_id)
    now = datetime.now(timezone.utc)
    a = await _make(repo, admin_id, "desk-mac", "t-1")

    mapped = await repo.update_device(
        a.id, DesktopDeviceChanges(set_edge_device=True, edge_device_id=edge_id), at=now
    )
    assert mapped is not None and mapped.edge_device_id == edge_id

    # set_edge_device False leaves the mapping alone, even with an id given.
    kept = await repo.update_device(a.id, DesktopDeviceChanges(name="study-mac", edge_device_id=None), at=now)
    assert kept is not None and kept.edge_device_id == edge_id

    cleared = await repo.update_device(
        a.id, DesktopDeviceChanges(set_edge_device=True, edge_device_id=None), at=now
    )
    assert cleared is not None and cleared.edge_device_id is None


@skip_without_postgres
async def test_setting_a_default_moves_it_and_false_leaves_none(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    now = datetime.now(timezone.utc)
    a = await _make(repo, admin_id, "desk-mac", "t-1")
    b = await _make(repo, admin_id, "laptop-mac", "t-2")

    first = await repo.update_device(a.id, DesktopDeviceChanges(is_default=True), at=now)
    assert first is not None and first.is_default is True

    moved = await repo.update_device(b.id, DesktopDeviceChanges(is_default=True), at=now)
    assert moved is not None and moved.is_default is True
    a_after = await repo.get_device(a.id)
    assert a_after is not None and a_after.is_default is False
    assert await _default_count(sessionmaker) == 1

    unset = await repo.update_device(b.id, DesktopDeviceChanges(is_default=False), at=now)
    assert unset is not None and unset.is_default is False
    assert await _default_count(sessionmaker) == 0


@skip_without_postgres
async def test_concurrent_set_default_leaves_exactly_one_default(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    now = datetime.now(timezone.utc)
    a = await _make(repo, admin_id, "desk-mac", "t-1")
    b = await _make(repo, admin_id, "laptop-mac", "t-2")

    for _ in range(5):
        results = await asyncio.gather(
            repo.update_device(a.id, DesktopDeviceChanges(is_default=True), at=now),
            repo.update_device(b.id, DesktopDeviceChanges(is_default=True), at=now),
            return_exceptions=True,
        )
        for result in results:
            if isinstance(result, BaseException):
                assert isinstance(result, DesktopDefaultConflict), repr(result)
        assert await _default_count(sessionmaker) == 1


@skip_without_postgres
async def test_the_database_refuses_a_second_default_row_without_the_application(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    a = await _make(repo, admin_id, "desk-mac", "t-1")
    b = await _make(repo, admin_id, "laptop-mac", "t-2")

    async with sessionmaker() as session:
        await session.execute(text("UPDATE desktop_devices SET is_default = true WHERE id = :id"), {"id": a.id})
        await session.commit()
    with pytest.raises(IntegrityError):
        async with sessionmaker() as session:
            await session.execute(
                text("UPDATE desktop_devices SET is_default = true WHERE id = :id"), {"id": b.id}
            )
            await session.commit()


@skip_without_postgres
async def test_revoke_is_once_clears_default_and_blocks_updates_and_token_lookup(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    now = datetime.now(timezone.utc)
    a = await _make(repo, admin_id, "desk-mac", "t-1")
    await repo.update_device(a.id, DesktopDeviceChanges(is_default=True), at=now)
    revoked_at = datetime.now(timezone.utc)

    assert await repo.revoke_device(a.id, revoked_at=revoked_at) is True
    assert await repo.revoke_device(a.id, revoked_at=datetime.now(timezone.utc)) is False
    assert await repo.revoke_device(999999, revoked_at=revoked_at) is False

    row = await repo.get_device(a.id)
    assert row is not None
    assert row.revoked_at == revoked_at
    assert row.is_default is False
    assert await repo.get_active_by_token_hash(hash_desktop_token("t-1")) is None
    assert await repo.update_device(a.id, DesktopDeviceChanges(name="new-name"), at=now) is None
    assert await repo.update_device(999999, DesktopDeviceChanges(name="new-name"), at=now) is None


@skip_without_postgres
async def test_mark_seen_sets_last_seen_and_ignores_a_missing_row(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    a = await _make(repo, admin_id, "desk-mac", "t-1")
    at = datetime.now(timezone.utc)

    await repo.mark_seen(a.id, at=at)
    await repo.mark_seen(999999, at=at)

    [listed] = await repo.list_devices()
    assert listed.last_seen_at == at
    assert listed.last_seen_at.tzinfo is not None


@skip_without_postgres
async def test_list_devices_is_oldest_first_revoked_included(sessionmaker):
    repo = PostgresDesktopDeviceRepository(sessionmaker)
    admin_id = await _create_admin_user(sessionmaker)
    first = await repo.create_device(
        name="first-mac",
        token_hash=hash_desktop_token("t-1"),
        created_by_user_id=admin_id,
        created_at=datetime(2026, 1, 1, tzinfo=timezone.utc),
    )
    second = await repo.create_device(
        name="second-mac",
        token_hash=hash_desktop_token("t-2"),
        created_by_user_id=admin_id,
        created_at=datetime(2026, 1, 2, tzinfo=timezone.utc),
    )
    await repo.revoke_device(first.id, revoked_at=datetime.now(timezone.utc))

    listed = await repo.list_devices()
    assert [d.id for d in listed] == [first.id, second.id]
    assert listed[0].revoked_at is not None
