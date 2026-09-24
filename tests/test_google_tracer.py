"""The Phase 9 tracer (09-01-PLAN.md, Task 1): one linked Google account's
calendar, read and spoken, through every layer this plan adds -- a
Postgres row with an encrypted refresh token, `GoogleTokenService` turning
it into a short-lived access token, `GoogleEnvBuilder`'s one-key
environment, a real `atlas_mcp.google` child spawned through
`PluginManager`, and an ordinary turn.

Three parts, in the order 09-01-PLAN.md's own `<action>` names them:

1. Postgres: store an OAuth client and one account with one enabled and
   one off calendar, build the env through the real `GoogleTokenService`
   over `FakeGoogle`, and assert the JSON has an access token, no refresh
   token or client secret anywhere in the serialized string, and only the
   enabled calendar.
2. Spawn the real child through `PluginManager`, call `calendar_list_events`
   with an unknown account, and assert the error names the linked label --
   proving the env reached the child, with no outbound network call.
3. Turn level: drive `run_turn` with a fake brain that calls
   `calendar_list_events` then answers, through a fake tool host that
   invokes `handle_calendar_list_events` against `FakeGoogle`, and assert
   the spoken text and that the tool result the brain received names the
   account and the event title.

Marked `integration` and requires a reachable Postgres (`ATLAS_TEST_DATABASE_URL`)
-- unlike `tests/test_db_migrations.py`'s other tests, this file does not
skip without one: the plan's own `<verify>` treats "skipped" in the
summary line as a failure, since a skip here means the tracer never
actually ran.
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import text
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine

import conftest
from google_fakes import FakeGoogle

from atlas_mcp.google import handle_calendar_list_events
from atlas_mcp.google_boundary import parse_accounts_env
from atlas_mcp.google_tools import GOOGLE_ACCOUNTS_ENV, GOOGLE_PLUGIN_MODULE, REQUIRED_SCOPES
from atlas_mcp.safety import Denied

from atlas.config import SecurityConfig
from atlas.crypto.credentials import encrypt_credential
from atlas.db.google_postgres import PostgresGoogleAccountRepository
from atlas.db.repository import Plugin
from atlas.google.env import GoogleEnvBuilder
from atlas.google.token_service import GoogleTokenService
from atlas.plugins.manager import PluginManager
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn

_TEST_DB_URL = os.environ.get("ATLAS_TEST_DATABASE_URL")
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_ROOT = os.path.join(_REPO_ROOT, "mcp")

pytestmark = pytest.mark.integration


def _migration_url(async_url: str) -> str:
    return "postgresql+psycopg://" + async_url[len("postgresql+asyncpg://") :]


async def _reset_schema(async_url: str) -> None:
    engine = create_async_engine(async_url)
    async with engine.begin() as conn:
        await conn.execute(text("DROP SCHEMA public CASCADE"))
        await conn.execute(text("CREATE SCHEMA public"))
    await engine.dispose()


def _run_upgrade_to_head(async_url: str) -> None:
    from alembic import command
    from alembic.config import Config as AlembicConfig

    cfg = AlembicConfig("alembic.ini")
    cfg.set_main_option("sqlalchemy.url", _migration_url(async_url))
    command.upgrade(cfg, "head")


@pytest.fixture
async def google_sessionmaker(monkeypatch):
    """`tests/test_wake_events.py`'s own `sessionmaker` fixture, unchanged
    -- reset schema, migrate to head, hand back a fresh
    `async_sessionmaker`."""
    assert _TEST_DB_URL, (
        "ATLAS_TEST_DATABASE_URL is not set -- run "
        "`eval \"$(scripts/dev-postgres.sh)\"` for a throwaway local Postgres, "
        "then re-run the suite; the Phase 9 tracer must run against a real "
        "Postgres, never skip"
    )
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
    _run_upgrade_to_head(_TEST_DB_URL)

    engine = create_async_engine(_TEST_DB_URL)
    yield async_sessionmaker(engine, expire_on_commit=False)
    await engine.dispose()


def _google_plugin(plugin_id: int = 1) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug="google",
        display_name="Google",
        transport="stdio",
        args=("-m", "atlas_mcp.google"),
        url=None,
        enabled=True,
        builtin=True,
        enforces_policy=False,
        timeout_ms=15000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )


async def test_env_carries_an_access_token_never_a_refresh_token_or_client_secret_and_only_the_enabled_calendar(
    google_sessionmaker,
):
    """Part 1: Postgres, the real `GoogleTokenService`, and `GoogleEnvBuilder`."""
    security = SecurityConfig()
    repo = PostgresGoogleAccountRepository(google_sessionmaker)

    client_secret_ciphertext, client_key_version = encrypt_credential("cs-1", security)
    await repo.set_oauth_client(
        client_id="client-1",
        client_secret_ciphertext=client_secret_ciphertext,
        key_version=client_key_version,
        updated_by_user_id=None,
        updated_at=datetime.now(timezone.utc),
    )

    refresh_ciphertext, refresh_key_version = encrypt_credential("rt-1", security)
    account = await repo.insert_account(
        label="work",
        email="work@example.com",
        refresh_token_ciphertext=refresh_ciphertext,
        key_version=refresh_key_version,
        granted_scopes=" ".join(REQUIRED_SCOPES),
        refresh_token_expires_at=None,
        linked_by_user_id=None,
        linked_at=datetime.now(timezone.utc),
    )
    await repo.add_calendars(
        account.id,
        [
            ("cal-on", "Team Offsite", True, True),
            ("cal-off", "Personal", False, False),
        ],
        discovered_at=datetime.now(timezone.utc),
    )
    [stored_account] = await repo.list_accounts()
    enabled_calendar = next(c for c in stored_account.calendars if c.google_calendar_id == "cal-on")
    await repo.set_calendar_access(
        enabled_calendar.id, "read_only", updated_at=datetime.now(timezone.utc)
    )

    fake_google = FakeGoogle()
    fake_google.add_refresh_token("rt-1", "at-1")

    token_service = GoogleTokenService(repo, security, fake_google.client)
    builder = GoogleEnvBuilder(repo, token_service)

    env = await builder(_google_plugin())
    raw = env[GOOGLE_ACCOUNTS_ENV]

    assert "rt-1" not in raw, "the refresh token must never reach the child's environment"
    assert "cs-1" not in raw, "the OAuth client secret must never reach the child's environment"
    assert "at-1" in raw, "the access token is exactly what the child needs"

    parsed = json.loads(raw)
    [account_json] = parsed["accounts"]
    assert account_json["label"] == "work"
    assert account_json["access_token"] == "at-1"
    calendar_ids = {c["calendar_id"] for c in account_json["calendars"]}
    assert calendar_ids == {"cal-on"}, "an off calendar must never reach the child (D-05)"


async def test_a_real_child_names_the_linked_label_for_an_unknown_account_with_no_network_call():
    """Part 2: a real `atlas_mcp.google` child, spawned through
    `PluginManager`, answers an unknown-account request by naming the
    linked label -- proving the env reached the child -- and never makes
    an outbound call (the refusal happens before any `httpx` call)."""
    fake_plugin_repo = conftest.FakePluginRepository(plugins=[_google_plugin()])

    async def _builder(_plugin) -> dict[str, str]:
        payload = {
            "accounts": [
                {
                    "label": "work",
                    "email": "work@example.com",
                    "is_default": True,
                    "access_token": "at-1",
                    "unreachable_reason": None,
                    "calendars": [
                        {
                            "calendar_id": "cal-on",
                            "name": "Team Offsite",
                            "primary": True,
                            "access": "read_only",
                        }
                    ],
                }
            ]
        }
        return {GOOGLE_ACCOUNTS_ENV: json.dumps(payload)}

    async def _no_policy():
        return None

    manager = PluginManager(
        fake_plugin_repo,
        mcp_root=_MCP_ROOT,
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
        custom_env_builders={GOOGLE_PLUGIN_MODULE: _builder},
    )
    try:
        await manager.start_all()
        host = manager.tool_host_for_module(GOOGLE_PLUGIN_MODULE)
        assert host is not None, "the google plugin must be running"

        result = await host.call_tool(
            "calendar_list_events",
            {"start": "2026-10-02", "end": "2026-10-03", "account": "nobody"},
        )
    finally:
        await manager.stop_all()

    is_error = bool(getattr(result, "isError", getattr(result, "is_error", False)))
    assert is_error, "an unknown account must be a tool error, not a silent empty result"
    text_out = result.content[0].text
    assert "work" in text_out


class _FakeGoogleToolHost:
    """Calls straight into `atlas_mcp.google`'s own handler function
    against `FakeGoogle`, the direct sibling of
    `tests/test_turn_controller.py::_FakeToolHost`."""

    def __init__(self, google: FakeGoogle, accounts, zone: ZoneInfo) -> None:
        self._google = google
        self._accounts = accounts
        self._zone = zone

    async def call_tool(self, name: str, arguments: dict):
        try:
            if name == "calendar_list_events":
                result = await handle_calendar_list_events(
                    self._accounts, self._google.client, self._zone, **arguments
                )
            else:
                raise AssertionError(f"unknown tool: {name}")
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


class _RecordingBrain:
    """A `_BrainProvider`-shaped double recording every `messages` list it
    was called with -- `tests/test_turn_controller.py::_RecordingBrain`'s
    own shape, so this test can assert the tool result the brain actually
    received names both the account and the event title."""

    def __init__(self, replies: list[BrainReply]) -> None:
        self._replies = list(replies)
        self.received_messages: list[list[dict]] = []
        self.call_count = 0

    async def chat(self, messages, tools=None) -> BrainReply:
        self.received_messages.append([dict(m) for m in messages])
        if not self._replies:
            raise AssertionError("_RecordingBrain.chat called more times than scripted")
        self.call_count += 1
        return self._replies.pop(0)


async def test_a_turn_speaks_the_brains_answer_built_from_the_tools_labeled_event(
    fake_audio_source, fake_stt, fake_tts
):
    """Part 3: a turn drives `calendar_list_events` through a fake tool
    host wired to `handle_calendar_list_events`/`FakeGoogle`, and speaks
    the brain's answer built from that tool's own, account-labeled
    result."""
    fake_google = FakeGoogle()
    fake_google.add_refresh_token("rt-1", "at-1")
    fake_google.add_events(
        "at-1",
        "cal-on",
        [
            {
                "id": "evt-1",
                "summary": "Team Offsite",
                "start": {"dateTime": "2026-10-02T09:00:00-04:00"},
                "end": {"dateTime": "2026-10-02T10:00:00-04:00"},
            }
        ],
    )

    accounts_env = json.dumps(
        {
            "accounts": [
                {
                    "label": "work",
                    "email": "work@example.com",
                    "is_default": True,
                    "access_token": "at-1",
                    "unreachable_reason": None,
                    "calendars": [
                        {
                            "calendar_id": "cal-on",
                            "name": "Team Offsite",
                            "primary": True,
                            "access": "read_only",
                        }
                    ],
                }
            ]
        }
    )
    accounts = parse_accounts_env(accounts_env)
    zone = ZoneInfo("UTC")
    tool_host = _FakeGoogleToolHost(fake_google, accounts, zone)

    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="what's on my calendar")])
    brain = _RecordingBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="calendar_list_events",
                        arguments={"start": "2026-10-02", "end": "2026-10-03"},
                    )
                ]
            ),
            BrainReply(text="work: team offsite at 9"),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you check calendars",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == ["work: team offsite at 9"]

    second_round_messages = brain.received_messages[1]
    tool_message = second_round_messages[-1]
    assert "work" in tool_message["content"]
    assert "Team Offsite" in tool_message["content"]
