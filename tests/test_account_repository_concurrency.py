"""The concurrent path WR-03 (code review) named as the gap: two routes
this phase's own design states answer "exactly once" -- create-admin and
invite acceptance -- had no atomicity between their check and their write.
`tests/test_auth_setup.py` and `tests/test_invites.py` already prove the
*happy* path (one request, in isolation) works; neither ever launches two
requests at the same instant, so neither could have caught this. This file
fires real concurrent calls at the real `PostgresAccountRepository` -- not
`FakeAccountRepository`, which has no interleaving to race in the first
place (`tests/conftest.py`'s own comment on both fixes explains why) -- and
asserts exactly one of N racing callers wins.

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_db_migrations.py`'s own precedent exactly (same skip reason,
same `eval "$(scripts/dev-postgres.sh)"` recovery instruction).
"""

from __future__ import annotations

import asyncio
import os
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from spire_voice.db.postgres import PostgresAccountRepository

_TEST_DB_URL = os.environ.get("SPIRE_TEST_DATABASE_URL")

pytestmark = pytest.mark.integration

skip_without_postgres = pytest.mark.skipif(
    _TEST_DB_URL is None,
    reason=(
        "SPIRE_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite, to exercise these tests instead of skipping them"
    ),
)


def _migration_url(async_url: str) -> str:
    """The synchronous `psycopg` connection string Alembic needs -- the
    same driver-swap `tests/test_db_migrations.py`'s own `_migration_url`
    performs, kept local rather than imported so this file has no
    cross-test-module coupling."""
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    """Drop every table either policy or accounts migration touches, plus
    Alembic's own bookkeeping table -- the same list `tests/
    test_db_migrations.py`'s own `_reset_schema` drops. Both tests below
    need a genuinely empty `users`/`invites` table, not whatever a
    previous run of this same throwaway Postgres left behind."""
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        for table in (
            # Plan 05-01: migration 0006 adds these two -- workflow_steps
            # first, it holds the foreign key onto workflow_runs.
            "workflow_steps",
            "workflow_runs",
            # Plan 04-05: migration 0005 adds these three.
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


def _run_upgrade_head(async_url: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, "head")


def _set_migration_env(monkeypatch) -> None:
    """`config/config.example.yaml` (Alembic's own default `SPIRE_CONFIG`
    target, per `0001_policy_tables.py`'s own docstring) expands
    `${NAME}` placeholders for every provider -- the same env vars
    `tests/test_db_migrations.py`'s own tests set, invented values only."""
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


@skip_without_postgres
async def test_concurrent_create_admin_calls_produce_exactly_one_user(monkeypatch):
    """WR-03 fix: ten concurrent `create_user_if_no_user_exists` calls,
    fired at the same instant against a genuinely empty `users` table,
    must produce exactly one `User` (the rest `None`) and exactly one row
    in the database -- not merely "usually one" under this test's own
    machine's scheduling, but guaranteed by the `pg_advisory_xact_lock`
    the fix serializes every caller through.
    """
    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repo = PostgresAccountRepository(sessionmaker)

    async def _attempt(n: int):
        return await repo.create_user_if_no_user_exists(
            email=f"racer-{n}@example.invalid",
            display_name=f"Racer {n}",
            password_hash="not-a-real-hash-never-checked",
            role="admin",
        )

    results = await asyncio.gather(*(_attempt(n) for n in range(10)))
    await engine.dispose()

    winners = [r for r in results if r is not None]
    assert len(winners) == 1, (
        f"expected exactly one winner among 10 concurrent create-admin calls, got "
        f"{len(winners)}: {winners!r}"
    )

    check_engine = create_async_engine(_TEST_DB_URL)
    async with check_engine.connect() as conn:
        row_count = (await conn.execute(text("SELECT count(*) FROM users"))).scalar_one()
    await check_engine.dispose()
    assert row_count == 1, f"expected exactly one users row after the race, found {row_count}"


@skip_without_postgres
async def test_concurrent_invite_accept_calls_claim_exactly_once(monkeypatch):
    """WR-03 fix: ten concurrent `claim_invite` calls against the same
    still-valid invite must produce exactly one `True` -- the rest `False`
    -- so the route layer above (`routes/accounts.py::accept_invite`)
    creates a user for exactly one of the racing callers, never two.
    """
    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repo = PostgresAccountRepository(sessionmaker)

    # An invite needs an existing user as its created_by_user_id (a real
    # foreign key, per 0002_accounts.py) -- an ordinary, uncontested
    # create_user, not the method under test above.
    admin = await repo.create_user(
        email="inviting-admin@example.invalid",
        display_name="Inviting Admin",
        password_hash="not-a-real-hash-never-checked",
        role="admin",
    )
    now = datetime.now(timezone.utc)
    invite = await repo.create_invite(
        token_hash="not-a-real-token-hash-never-checked",
        role="operator",
        email=None,
        expires_at=now + timedelta(days=7),
        created_by_user_id=admin.id,
    )

    async def _attempt():
        return await repo.claim_invite(invite.id, now=datetime.now(timezone.utc))

    results = await asyncio.gather(*(_attempt() for _ in range(10)))
    await engine.dispose()

    successes = [r for r in results if r is True]
    assert len(successes) == 1, (
        f"expected exactly one successful claim among 10 concurrent accepts, got "
        f"{len(successes)}: {results!r}"
    )

    check_engine = create_async_engine(_TEST_DB_URL)
    async with check_engine.connect() as conn:
        accepted_at = (
            await conn.execute(text("SELECT accepted_at FROM invites WHERE id = :id"), {"id": invite.id})
        ).scalar_one()
    await check_engine.dispose()
    assert accepted_at is not None, "the winning claim must have set accepted_at"
