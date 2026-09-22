"""`PostgresPluginRepository`'s write half against a real Postgres (D-01,
D-03, D-04, plan 06-06 Task 2): install, enable/disable, configuration
edits, and delete.

Marked `integration` and skipped without a reachable Postgres, matching
`tests/test_macro_repo.py`'s own precedent -- the direct sibling this file
follows for its `sessionmaker` fixture (reset schema, migrate to head, hand
back a fresh `async_sessionmaker`). Migration `0008`'s own seed step always
creates the two builtin rows (Home Assistant, weather) regardless of what
`config.example.yaml` says -- every test below either reads those rows
directly or creates its own additional plugin through the repository under
test.
"""

from __future__ import annotations

import os

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

from spire_voice.config import SecurityConfig
from spire_voice.crypto.credentials import encrypt_credential
from spire_voice.db.postgres import PostgresPluginRepository
from spire_voice.db.repository import PluginAlreadyExistsError, PluginConfigValue

_TEST_DB_URL = os.environ.get("SPIRE_TEST_DATABASE_URL")
_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"

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
    monkeypatch.setenv("SPIRE_CONFIG", "config/config.example.yaml")
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    monkeypatch.setenv("XAI_API_KEY", "test-value")
    monkeypatch.setenv("TAPO_USER", "test-value")
    monkeypatch.setenv("TAPO_PASSWORD", "test-value")
    monkeypatch.setenv("SPEAKER_ENSURE_URL", "test-value")
    monkeypatch.setenv("CAMERA_RTSP_URL", "rtsp://test.invalid:554/stream1")
    monkeypatch.setenv("SPEAKER_BACKEND", "go2rtc")
    # Phase 7 (D-15): server.bind_host / security.cookie_secure -- the two
    # new ${VAR} placeholders config.example.yaml expands.
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
async def test_a_created_plugin_round_trips_with_its_config_values(sessionmaker):
    repo = PostgresPluginRepository(sessionmaker)
    security = SecurityConfig()
    ciphertext, key_version = encrypt_credential("a-fictional-token", security)

    created = await repo.create_plugin(
        slug="example-repo-plugin",
        display_name="Example Repo Plugin",
        transport="stdio",
        args=["-m", "example_repo_module"],
        url=None,
        timeout_ms=5000,
        config_values=[
            PluginConfigValue(key="EXAMPLE_URL", secret=False, value="http://example.invalid", ciphertext=None, key_version=None),
            PluginConfigValue(key="EXAMPLE_TOKEN", secret=True, value=None, ciphertext=ciphertext, key_version=key_version),
        ],
        created_by_user_id=None,
    )

    assert created.slug == "example-repo-plugin"
    assert created.enabled is True
    assert created.builtin is False
    assert created.enforces_policy is False

    reread = await repo.get_plugin(created.id)
    assert reread is not None
    assert reread.slug == created.slug

    values = {v.key: v for v in await repo.get_config_values(created.id)}
    assert values["EXAMPLE_URL"].value == "http://example.invalid"
    assert values["EXAMPLE_TOKEN"].secret is True
    assert values["EXAMPLE_TOKEN"].value is None
    assert values["EXAMPLE_TOKEN"].ciphertext == ciphertext

    [listed] = [p for p in await repo.list_plugins() if p.id == created.id]
    assert listed.slug == created.slug


@skip_without_postgres
async def test_a_duplicate_slug_is_refused_by_name(sessionmaker):
    repo = PostgresPluginRepository(sessionmaker)
    await repo.create_plugin(
        slug="dup-example",
        display_name="Dup Example",
        transport="stdio",
        args=["-m", "example_repo_module"],
        url=None,
        timeout_ms=5000,
        config_values=[],
        created_by_user_id=None,
    )
    with pytest.raises(PluginAlreadyExistsError, match="dup-example"):
        await repo.create_plugin(
            slug="dup-example",
            display_name="Dup Example Again",
            transport="stdio",
            args=["-m", "example_repo_module_two"],
            url=None,
            timeout_ms=5000,
            config_values=[],
            created_by_user_id=None,
        )


@skip_without_postgres
async def test_set_enabled_flips_the_row_and_changes_nothing_else(sessionmaker):
    repo = PostgresPluginRepository(sessionmaker)
    created = await repo.create_plugin(
        slug="example-toggle-plugin",
        display_name="Example Toggle Plugin",
        transport="stdio",
        args=["-m", "example_repo_module"],
        url=None,
        timeout_ms=5000,
        config_values=[],
        created_by_user_id=None,
    )
    assert created.enabled is True

    disabled = await repo.set_enabled(created.id, False)
    assert disabled.enabled is False
    assert disabled.slug == created.slug
    assert disabled.display_name == created.display_name

    enabled_again = await repo.set_enabled(created.id, True)
    assert enabled_again.enabled is True


@skip_without_postgres
async def test_set_config_values_upserts_by_key_and_leaves_unmentioned_keys_alone(sessionmaker):
    repo = PostgresPluginRepository(sessionmaker)
    created = await repo.create_plugin(
        slug="example-config-plugin",
        display_name="Example Config Plugin",
        transport="stdio",
        args=["-m", "example_repo_module"],
        url=None,
        timeout_ms=5000,
        config_values=[
            PluginConfigValue(key="KEEP_ME", secret=False, value="original", ciphertext=None, key_version=None),
            PluginConfigValue(key="OVERWRITE_ME", secret=False, value="old", ciphertext=None, key_version=None),
        ],
        created_by_user_id=None,
    )

    updated = await repo.set_config_values(
        created.id,
        [PluginConfigValue(key="OVERWRITE_ME", secret=False, value="new", ciphertext=None, key_version=None)],
    )
    by_key = {v.key: v for v in updated}
    assert by_key["KEEP_ME"].value == "original"
    assert by_key["OVERWRITE_ME"].value == "new"

    # A brand-new key not present before is inserted, not refused.
    updated_again = await repo.set_config_values(
        created.id,
        [PluginConfigValue(key="NEW_KEY", secret=False, value="fresh", ciphertext=None, key_version=None)],
    )
    by_key_again = {v.key: v for v in updated_again}
    assert by_key_again["NEW_KEY"].value == "fresh"
    assert by_key_again["KEEP_ME"].value == "original"


@skip_without_postgres
async def test_delete_plugin_removes_it_and_its_config_values_together(sessionmaker):
    repo = PostgresPluginRepository(sessionmaker)
    created = await repo.create_plugin(
        slug="example-delete-plugin",
        display_name="Example Delete Plugin",
        transport="stdio",
        args=["-m", "example_repo_module"],
        url=None,
        timeout_ms=5000,
        config_values=[
            PluginConfigValue(key="SOME_KEY", secret=False, value="value", ciphertext=None, key_version=None),
        ],
        created_by_user_id=None,
    )

    await repo.delete_plugin(created.id)

    assert await repo.get_plugin(created.id) is None
    assert await repo.get_config_values(created.id) == []


@skip_without_postgres
async def test_deleting_an_already_gone_plugin_is_a_no_op(sessionmaker):
    repo = PostgresPluginRepository(sessionmaker)
    await repo.delete_plugin(999999)  # never raises
