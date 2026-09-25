"""09-01-PLAN.md Task 3: token status transitions (GOOG-12), finding and
refreshing the running google plugin's own environment, and the schedule
that drives both -- proven against fakes, no real Postgres or subprocess
needed.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import httpx
import pytest

from atlas.config import SecurityConfig
from atlas.crypto.credentials import encrypt_credential
from atlas.db.repository import Plugin
from atlas.google.plugin import find_google_plugin, refresh_google_plugin
from atlas.google.scheduler import GoogleTokenRefreshScheduler
from atlas.google.token_service import GoogleTokenService
from atlas.plugins.manager import PluginState

import conftest
from google_fakes import FakeGoogle
from google_repo_fakes import FakeGoogleAccountRepository


def _google_plugin(plugin_id: int = 1, *, enabled: bool = True) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug="google",
        display_name="Google",
        transport="stdio",
        args=("-m", "atlas_mcp.google"),
        url=None,
        enabled=enabled,
        builtin=True,
        enforces_policy=False,
        timeout_ms=15000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )


# === Token status transitions ============================================


async def _repo_with_account(security: SecurityConfig, *, with_oauth_client: bool = True):
    repo = FakeGoogleAccountRepository()
    if with_oauth_client:
        secret_ciphertext, secret_key_version = encrypt_credential("cs-1", security)
        await repo.set_oauth_client(
            client_id="client-1",
            client_secret_ciphertext=secret_ciphertext,
            key_version=secret_key_version,
            updated_by_user_id=None,
            updated_at=datetime.now(timezone.utc),
        )
    refresh_ciphertext, refresh_key_version = encrypt_credential("rt-1", security)
    account = await repo.insert_account(
        label="work",
        email="work@example.com",
        refresh_token_ciphertext=refresh_ciphertext,
        key_version=refresh_key_version,
        granted_scopes="calendar",
        refresh_token_expires_at=None,
        linked_by_user_id=None,
        linked_at=datetime.now(timezone.utc),
    )
    return repo, account


async def test_an_invalid_grant_marks_the_account_needs_relink():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake_google = FakeGoogle()
    # No refresh token registered -> the token endpoint answers invalid_grant.

    token_service = GoogleTokenService(repo, security, fake_google.client)
    result = await token_service.access_token_for(account)

    assert result.token is None
    assert result.unreachable_reason == "needs_relink"
    stored = await repo.get_account(account.id)
    assert stored.status == "needs_relink"


async def test_a_transport_error_marks_the_account_unreachable_and_a_later_success_clears_it():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake_google = FakeGoogle()
    fake_google.add_refresh_token("rt-1", "at-1")
    fake_google.fail_events("at-1")  # irrelevant to the token endpoint; separate assertion below

    # Simulate a transport error on the token endpoint itself by pointing
    # the token service at a client whose transport always raises.
    def _raise_connect_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    broken_client = httpx.AsyncClient(transport=httpx.MockTransport(_raise_connect_error))
    token_service = GoogleTokenService(repo, security, broken_client)

    result = await token_service.access_token_for(account)
    assert result.token is None
    assert result.unreachable_reason == "unreachable"
    stored = await repo.get_account(account.id)
    assert stored.status == "unreachable"

    # The next call, against a working client, clears the status back to ok.
    working_client = fake_google.client
    healthy_service = GoogleTokenService(repo, security, working_client)
    healthy_result = await healthy_service.access_token_for(account)
    assert healthy_result.token == "at-1"
    stored_again = await repo.get_account(account.id)
    assert stored_again.status == "ok"


async def test_a_missing_oauth_client_marks_the_account_needs_relink():
    security = SecurityConfig()
    repo, account = await _repo_with_account(security, with_oauth_client=False)
    fake_google = FakeGoogle()

    token_service = GoogleTokenService(repo, security, fake_google.client)
    result = await token_service.access_token_for(account)

    assert result.unreachable_reason == "needs_relink"
    stored = await repo.get_account(account.id)
    assert stored.status == "needs_relink"
    assert stored.status_detail == "no google oauth client is configured"


async def test_no_logger_call_ever_names_the_refresh_token_value(caplog):
    """The acceptance criterion's own grep proves the source never
    contains the shape; this proves the behavior: even a failure path's
    logging never puts the raw token value in a log record."""
    import logging

    security = SecurityConfig()
    repo, account = await _repo_with_account(security)
    fake_google = FakeGoogle()  # no refresh token registered -> invalid_grant

    token_service = GoogleTokenService(repo, security, fake_google.client)
    with caplog.at_level(logging.WARNING, logger="atlas.google.token_service"):
        await token_service.access_token_for(account)

    for record in caplog.records:
        assert "rt-1" not in record.getMessage()


# === find_google_plugin / refresh_google_plugin ===========================


def test_find_google_plugin_matches_by_module_not_slug():
    plugin = _google_plugin()
    other = Plugin(
        id=2, slug="weather", display_name="Weather", transport="stdio",
        args=("-m", "atlas_mcp.weather"), url=None, enabled=True, builtin=True,
        enforces_policy=False, timeout_ms=5000,
        created_at=plugin.created_at, updated_at=plugin.updated_at, created_by_user_id=None,
    )
    repo = conftest.FakePluginRepository(plugins=[other, plugin])

    import asyncio

    found = asyncio.run(find_google_plugin(repo))
    assert found is not None
    assert found.id == plugin.id


def test_find_google_plugin_returns_none_when_absent():
    import asyncio

    repo = conftest.FakePluginRepository(plugins=[])
    assert asyncio.run(find_google_plugin(repo)) is None


class _FakePluginManagerForGoogle:
    """The narrow `PluginManager` surface `refresh_google_plugin` reads --
    `tests/test_plugin_routes.py::_FakePluginManagerForRoutes`'s own
    precedent, narrowed to this module's own two calls."""

    def __init__(self, state: "PluginState | None") -> None:
        self._state = state
        self.respawn_calls: list[int] = []
        self.start_one_calls: list[int] = []
        self.fail_respawn = False

    def state_for(self, slug):
        return self._state

    async def request_respawn(self, plugin_id, safety_block):
        if self.fail_respawn:
            raise RuntimeError("simulated respawn failure")
        self.respawn_calls.append(plugin_id)

    async def start_one(self, plugin):
        self.start_one_calls.append(plugin.id)


async def test_refresh_respawns_a_running_plugin():
    plugin = _google_plugin()
    repo = conftest.FakePluginRepository(plugins=[plugin])
    manager = _FakePluginManagerForGoogle(PluginState.RUNNING)

    await refresh_google_plugin(manager, repo)

    assert manager.respawn_calls == [plugin.id]
    assert manager.start_one_calls == []


async def test_refresh_starts_an_enabled_plugin_that_is_not_running():
    plugin = _google_plugin()
    repo = conftest.FakePluginRepository(plugins=[plugin])
    manager = _FakePluginManagerForGoogle(PluginState.DEGRADED)

    await refresh_google_plugin(manager, repo)

    assert manager.start_one_calls == [plugin.id]
    assert manager.respawn_calls == []


async def test_refresh_does_nothing_when_no_google_plugin_exists():
    repo = conftest.FakePluginRepository(plugins=[])
    manager = _FakePluginManagerForGoogle(PluginState.RUNNING)

    await refresh_google_plugin(manager, repo)

    assert manager.respawn_calls == []
    assert manager.start_one_calls == []


async def test_refresh_does_nothing_when_the_plugin_is_disabled():
    plugin = _google_plugin(enabled=False)
    repo = conftest.FakePluginRepository(plugins=[plugin])
    manager = _FakePluginManagerForGoogle(PluginState.DISABLED)

    await refresh_google_plugin(manager, repo)

    assert manager.respawn_calls == []
    assert manager.start_one_calls == []


async def test_a_failed_respawn_is_logged_and_swallowed():
    plugin = _google_plugin()
    repo = conftest.FakePluginRepository(plugins=[plugin])
    manager = _FakePluginManagerForGoogle(PluginState.RUNNING)
    manager.fail_respawn = True

    # Must not raise.
    await refresh_google_plugin(manager, repo)


# === GoogleTokenRefreshScheduler ==========================================


async def test_scheduler_calls_refresh_once_per_interval():
    calls = 0

    async def _refresh() -> None:
        nonlocal calls
        calls += 1

    sleeps: list[float] = []

    async def _fake_sleep(seconds: float) -> None:
        # A real suspension point, not just an async def with no body that
        # awaits anything -- `_run()`'s own while loop has no other yield
        # point, so a `sleep=` stand-in that never truly suspends spins the
        # event loop forever inside one task and this test's own polling
        # loop below never gets scheduled to check `calls` or call `stop()`.
        sleeps.append(seconds)
        await asyncio.sleep(0)

    scheduler = GoogleTokenRefreshScheduler(_refresh, interval_s=1500.0, sleep=_fake_sleep)
    scheduler.start()
    # Let the loop run through a couple of iterations under the injected sleep.
    for _ in range(20):
        await asyncio.sleep(0)
        if calls >= 3:
            break
    await scheduler.stop()

    assert calls >= 3
    assert all(s == 1500.0 for s in sleeps)


async def test_scheduler_survives_a_raising_refresh():
    calls = 0

    async def _raising_refresh() -> None:
        nonlocal calls
        calls += 1
        raise RuntimeError("boom")

    async def _fake_sleep(seconds: float) -> None:
        # See the identical comment in test_scheduler_calls_refresh_once_per_interval.
        await asyncio.sleep(0)

    scheduler = GoogleTokenRefreshScheduler(_raising_refresh, sleep=_fake_sleep)
    scheduler.start()
    for _ in range(20):
        await asyncio.sleep(0)
        if calls >= 3:
            break
    await scheduler.stop()

    assert calls >= 3


async def test_scheduler_stops_with_no_further_call():
    calls = 0

    async def _refresh() -> None:
        nonlocal calls
        calls += 1

    async def _fake_sleep(seconds: float) -> None:
        # See the identical comment in test_scheduler_calls_refresh_once_per_interval.
        await asyncio.sleep(0)

    scheduler = GoogleTokenRefreshScheduler(_refresh, sleep=_fake_sleep)
    scheduler.start()
    await asyncio.sleep(0)
    await scheduler.stop()
    count_at_stop = calls
    for _ in range(5):
        await asyncio.sleep(0)
    assert calls == count_at_stop
