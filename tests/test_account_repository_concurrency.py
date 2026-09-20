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


# --- WR-11 (code review): concurrent provider-selection saves -----------


@skip_without_postgres
async def test_concurrent_set_selection_calls_for_an_unseeded_slot_do_not_500(monkeypatch):
    """WR-11 fix. `set_selection` was a SELECT followed by an
    INSERT-or-UPDATE, with no row lock and no `ON CONFLICT`. Two
    concurrent `PUT /api/providers` calls for a slot with no row yet both
    saw `None`, both inserted, and `uq_provider_selections_slot` rejected
    the second with an `IntegrityError` that reached the route uncaught.

    The row is deleted first on purpose: migration 0011 seeds all three
    slots, so the no-row case is rare on a real deployment -- but
    `repository.py`'s own protocol documents `get_selection` returning
    `None` as a real state, and a rare race is still a 500 on a save that
    was perfectly valid.

    Twenty concurrent callers, all of which must succeed, leaving exactly
    one row whose provider name is one of the twenty that were offered.

    The pool is pre-warmed on purpose. Without that, every caller's first
    `await` is a TCP connect plus authentication, and the resulting
    scheduling hides the race entirely -- this test passed against the
    defective code until the warm-up was added. With it, 19 of 20 callers
    raised `IntegrityError` on the pre-fix implementation.
    """
    from sqlalchemy import delete, select

    from spire_voice.db.models import ProviderSelectionRow
    from spire_voice.db.postgres import PostgresProviderSelectionRepository

    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)

    racers = 20
    engine = create_async_engine(_TEST_DB_URL, pool_size=racers, max_overflow=0)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)

    async with sessionmaker() as session:
        await session.execute(
            delete(ProviderSelectionRow).where(ProviderSelectionRow.slot == "stt")
        )
        await session.commit()

    repo = PostgresProviderSelectionRepository(sessionmaker)
    now = datetime.now(timezone.utc)

    # Every connection established before the racers start, so the first
    # `await` inside `set_selection` is its own query and not a TCP
    # connect -- see this test's docstring.
    await asyncio.gather(*(repo.get_selection("stt") for _ in range(racers)))

    async def _save(n: int):
        return await repo.set_selection(
            "stt", f"racer-{n}", {"n": n}, updated_by_user_id=None, updated_at=now
        )

    results = await asyncio.gather(
        *(_save(n) for n in range(racers)), return_exceptions=True
    )

    failures = [r for r in results if isinstance(r, BaseException)]
    assert not failures, (
        f"every concurrent save must succeed; {len(failures)} raised: {failures!r}"
    )

    async with sessionmaker() as session:
        rows = (
            (
                await session.execute(
                    select(ProviderSelectionRow).where(ProviderSelectionRow.slot == "stt")
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()

    assert len(rows) == 1, f"expected exactly one row for the slot, found {len(rows)}"
    assert rows[0].provider_name in {f"racer-{n}" for n in range(racers)}


@skip_without_postgres
async def test_set_selection_updates_the_existing_row_rather_than_adding_a_second(monkeypatch):
    """The upsert's other half: a slot that already has a row is updated
    in place, and `set_selection` still returns the stored state."""
    from sqlalchemy import select

    from spire_voice.db.models import ProviderSelectionRow
    from spire_voice.db.postgres import PostgresProviderSelectionRepository

    _set_migration_env(monkeypatch)
    await _reset_schema(_TEST_DB_URL)
    await asyncio.to_thread(_run_upgrade_head, _TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    sessionmaker = async_sessionmaker(engine, expire_on_commit=False)
    repo = PostgresProviderSelectionRepository(sessionmaker)
    now = datetime.now(timezone.utc)

    first = await repo.set_selection(
        "tts", "piper", {"voice": "lessac"}, updated_by_user_id=None, updated_at=now
    )
    second = await repo.set_selection(
        "tts", "xai", {}, updated_by_user_id=None, updated_at=now
    )

    async with sessionmaker() as session:
        rows = (
            (
                await session.execute(
                    select(ProviderSelectionRow).where(ProviderSelectionRow.slot == "tts")
                )
            )
            .scalars()
            .all()
        )
    await engine.dispose()

    assert first.provider_name == "piper"
    assert first.options == {"voice": "lessac"}
    assert second.provider_name == "xai"
    assert second.options == {}
    assert len(rows) == 1
    assert rows[0].provider_name == "xai"
    # The upsert updates the row rather than replacing it: the identity a
    # future audit row or foreign key would point at survives a save.
    assert rows[0].id == first.id
