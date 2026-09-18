"""`PostgresMacroRepository` against a real Postgres (D-09, MACRO-03).

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_db_postgres_datetime_roundtrip.py`'s own precedent -- the
direct sibling this file follows for its `sessionmaker` fixture (reset
schema, migrate to head, hand back a fresh `async_sessionmaker`).

`config/config.example.yaml` no longer carries a `macros:` block (D-09:
the key is retired), so migration `0005`'s own seed step seeds nothing
when this fixture runs it -- every macro these tests exercise is created
directly through the repository under test, proving the repository's own
CRUD contract rather than the migration's seed step (`tests/
test_db_migrations.py` covers that separately).
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from spire_voice.config import MacroConfig
from spire_voice.db.postgres import PostgresMacroRepository

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
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        for table in (
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


@pytest.fixture
async def sessionmaker(monkeypatch):
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

    await _reset_schema(_TEST_DB_URL)
    _run_upgrade_head(_TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


@skip_without_postgres
async def test_a_created_macro_round_trips_with_actions_in_written_order(sessionmaker):
    repo = PostgresMacroRepository(sessionmaker)

    created = await repo.create_macro(
        phrase="example goodnight macro",
        aliases=["example goodnight", "example night night macro"],
        reply="okay, goodnight",
        actions=[
            ("ha_call_service", {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_repo_fan"}),
            ("ha_call_service", {"domain": "light", "service": "turn_off", "entity_id": "light.example_repo_lamp"}),
        ],
        created_by_user_id=None,
    )

    assert created.phrase == "example goodnight macro"
    assert created.reply == "okay, goodnight"
    assert set(created.aliases) == {"example goodnight", "example night night macro"}
    assert [a.tool for a in created.actions] == ["ha_call_service", "ha_call_service"]
    assert [a.arguments["entity_id"] for a in created.actions] == [
        "switch.example_repo_fan",
        "light.example_repo_lamp",
    ]
    assert [a.position for a in created.actions] == [0, 1]

    reread = await repo.get_macro(created.id)
    assert reread is not None
    assert reread.phrase == created.phrase
    assert [a.arguments["entity_id"] for a in reread.actions] == [
        "switch.example_repo_fan",
        "light.example_repo_lamp",
    ]

    [listed] = [m for m in await repo.list_macros() if m.id == created.id]
    assert listed.phrase == created.phrase


@skip_without_postgres
async def test_an_update_that_reorders_actions_persists_the_new_order(sessionmaker):
    repo = PostgresMacroRepository(sessionmaker)

    created = await repo.create_macro(
        phrase="example reorder macro",
        aliases=[],
        reply="done",
        actions=[
            ("ha_call_service", {"entity_id": "light.example_reorder_first"}),
            ("ha_call_service", {"entity_id": "light.example_reorder_second"}),
        ],
        created_by_user_id=None,
    )
    assert [a.arguments["entity_id"] for a in created.actions] == [
        "light.example_reorder_first",
        "light.example_reorder_second",
    ]

    updated = await repo.update_macro(
        created.id,
        phrase=created.phrase,
        aliases=created.aliases,
        reply=created.reply,
        # Reversed order, plus a third action -- an update replaces the
        # action list wholesale (the Protocol's own documented contract),
        # not a diff against the old one.
        actions=[
            ("ha_call_service", {"entity_id": "light.example_reorder_second"}),
            ("ha_call_service", {"entity_id": "light.example_reorder_first"}),
            ("ha_call_service", {"entity_id": "light.example_reorder_third"}),
        ],
    )

    assert [a.arguments["entity_id"] for a in updated.actions] == [
        "light.example_reorder_second",
        "light.example_reorder_first",
        "light.example_reorder_third",
    ]
    assert [a.position for a in updated.actions] == [0, 1, 2]

    reread = await repo.get_macro(created.id)
    assert reread is not None
    assert [a.arguments["entity_id"] for a in reread.actions] == [
        "light.example_reorder_second",
        "light.example_reorder_first",
        "light.example_reorder_third",
    ]


@skip_without_postgres
async def test_a_delete_removes_the_macro_and_its_actions_and_aliases_together(sessionmaker):
    repo = PostgresMacroRepository(sessionmaker)

    created = await repo.create_macro(
        phrase="example delete macro",
        aliases=["example delete alias"],
        reply="gone soon",
        actions=[("ha_call_service", {"entity_id": "light.example_delete_lamp"})],
        created_by_user_id=None,
    )

    await repo.delete_macro(created.id)

    assert await repo.get_macro(created.id) is None
    assert created.id not in {m.id for m in await repo.list_macros()}

    async with sessionmaker() as session:
        action_count = (
            await session.execute(
                text("SELECT COUNT(*) FROM macro_actions WHERE macro_id = :macro_id"),
                {"macro_id": created.id},
            )
        ).scalar_one()
        alias_count = (
            await session.execute(
                text("SELECT COUNT(*) FROM macro_aliases WHERE macro_id = :macro_id"),
                {"macro_id": created.id},
            )
        ).scalar_one()
    assert action_count == 0
    assert alias_count == 0

    # A second delete of the same, already-gone id is not an error --
    # `MacroRepository.delete_macro`'s own documented contract.
    await repo.delete_macro(created.id)


@skip_without_postgres
async def test_a_database_loaded_macro_matches_normalized_keys_with_a_file_loaded_one(
    sessionmaker,
):
    """The proof this plan's own must-have names explicitly: a macro
    loaded back from the database produces the same normalized keys as
    the same macro read from a file -- matching behaviour is unchanged by
    the move (D-09)."""
    repo = PostgresMacroRepository(sessionmaker)

    raw = {
        "phrase": "Example Normalized-Keys Macro",
        "aliases": ["example   normalized keys macro!", "EXAMPLE NORMALIZED KEYS MACRO"],
        "reply": "ok",
        "actions": [{"tool": "ha_call_service", "arguments": {"entity_id": "light.example_norm"}}],
    }
    file_loaded = MacroConfig.from_config(raw)

    db_loaded = await repo.create_macro(
        phrase=raw["phrase"],
        aliases=raw["aliases"],
        reply=raw["reply"],
        actions=[(a["tool"], a["arguments"]) for a in raw["actions"]],
        created_by_user_id=None,
    )

    assert db_loaded.normalized_keys == file_loaded.normalized_keys
