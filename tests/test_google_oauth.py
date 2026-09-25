"""Linking one Google account through a real web callback (09-03-PLAN.md,
Task 1) -- `PUT /api/google/client`, `POST /api/google/oauth/start`, and
`GET /api/google/oauth/callback`, proven against `FakeGoogleAccountRepository`
and `FakeGoogle`, the same "primitives in isolation" shape
`tests/test_plugin_routes.py` already uses.

Every token, secret, and code literal in this file is shorter than eight
characters (`"at-1"`, `"rt-1"`, `"cs-1"`, `"code-1"`) -- `tests/
test_repo_hygiene.py::_CREDENTIAL_RE` scans every tracked file and all of
git history for `token|secret|password|api_key` followed by an
eight-plus character quoted value, and a mistake here is permanent.
"""

from __future__ import annotations

import urllib.parse
from dataclasses import replace
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient

from google_fakes import FakeGoogle
from google_repo_fakes import FakeGoogleAccountRepository

from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig
from atlas.crypto.credentials import decrypt_credential
from atlas.google.token_service import GoogleTokenService
from atlas.plugins.manager import PluginState
from atlas.routes.google_accounts import router as google_accounts_router

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"
_HTTPS_ORIGIN = "https://atlas.example.com"


class _FakePluginManagerForGoogle:
    """`_FakePluginManagerForRoutes`'s own sibling in
    `tests/test_plugin_routes.py`, extended with `request_respawn` --
    `reconcile_google_plugin` calls that instead of `start_one` once the
    plugin is already `RUNNING`."""

    def __init__(self) -> None:
        self.states: dict[str, PluginState] = {}
        self.start_calls: list[int] = []
        self.stop_calls: list[int] = []
        self.respawn_calls: list[int] = []
        self.fail_start = False
        self.fail_respawn = False

    def state_for(self, slug):
        return self.states.get(slug)

    async def start_one(self, plugin):
        if self.fail_start:
            raise RuntimeError("simulated start failure")
        self.start_calls.append(plugin.id)
        self.states[plugin.slug] = PluginState.RUNNING

    async def stop_one(self, plugin):
        self.stop_calls.append(plugin.id)
        self.states[plugin.slug] = PluginState.DISABLED

    async def request_respawn(self, plugin_id, safety_block):
        if self.fail_respawn:
            raise RuntimeError("simulated respawn failure")
        self.respawn_calls.append(plugin_id)


def _build_app(security, account_repo, google_repo, plugin_repo, plugin_manager, fake_google: FakeGoogle):
    app = FastAPI()
    app.state.config = SimpleNamespace(security=security)
    app.state.account_repo = account_repo
    app.state.google_account_repo = google_repo
    app.state.google_http_client = fake_google.client
    app.state.google_token_service = GoogleTokenService(google_repo, security, fake_google.client)
    app.state.plugin_repo = plugin_repo
    app.state.plugin_manager = plugin_manager
    app.include_router(google_accounts_router)
    return app


def _issue_user(account_repo, *, role: str):
    import asyncio

    return asyncio.run(
        account_repo.create_user(
            email=f"{role}@example.invalid",
            display_name=f"A {role.title()}",
            password_hash="not-checked-by-this-test",
            role=role,
        )
    )


def _client_as(app, security, account_repo, *, role: str) -> TestClient:
    user = _issue_user(account_repo, role=role)
    token = issue_access_token(user_id=user.id, role=role, security=security)
    return TestClient(app, cookies={security.cookie_name: token}, follow_redirects=False)


def _state_from_authorization_url(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    return urllib.parse.parse_qs(parsed.query)["state"][0]


def test_link_account_stores_encrypted_refresh_token(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    put_response = client.put("/api/google/client", json={"client_id": "cid-1", "client_secret": "cs-1"})
    assert put_response.status_code == 200, put_response.text
    assert "cs-1" not in put_response.text

    start_response = client.post(
        "/api/google/oauth/start", json={"label": "work"}, headers={"Origin": _HTTPS_ORIGIN}
    )
    assert start_response.status_code == 200, start_response.text
    authorization_url = start_response.json()["authorization_url"]
    assert "cs-1" not in start_response.text
    raw_state = _state_from_authorization_url(authorization_url)

    fake_google.add_code("code-1", access_token="at-1", refresh_token="rt-1")
    fake_google.add_profile("at-1", "work@example.com")
    fake_google.add_calendar_list(
        "at-1",
        [
            {"id": "cal-1", "summary": "Cal One", "primary": True, "accessRole": "owner"},
            {"id": "cal-2", "summary": "Cal Two", "accessRole": "reader"},
        ],
    )

    callback_response = client.get(
        "/api/google/oauth/callback", params={"code": "code-1", "state": raw_state}
    )
    assert callback_response.status_code == 303, callback_response.text
    location = callback_response.headers["location"]
    assert location.startswith("/google/accounts/")
    assert location.endswith("?linked=1")
    assert "at-1" not in callback_response.text
    assert "rt-1" not in callback_response.text
    assert "cs-1" not in callback_response.text

    [stored] = google_repo._accounts.values()
    plaintext = decrypt_credential(stored.refresh_token_ciphertext, stored.key_version, security)
    assert plaintext == "rt-1"
    assert stored.refresh_token_ciphertext != b"rt-1"

    list_response = client.get("/api/google/accounts")
    assert list_response.status_code == 200, list_response.text
    [account] = list_response.json()
    assert account["email"] == "work@example.com"
    assert account["label"] == "work"
    assert {c["access"] for c in account["calendars"]} == {"off"}
    assert "at-1" not in list_response.text
    assert "rt-1" not in list_response.text
    assert manager.start_calls, "the first successful link must start the google plugin"


def test_start_refuses_a_plain_http_origin(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")
    client.put("/api/google/client", json={"client_id": "cid-1", "client_secret": "cs-1"})

    response = client.post(
        "/api/google/oauth/start", json={"label": "work"}, headers={"Origin": "http://atlas.example.com"}
    )
    assert response.status_code == 400
    assert "https" in response.json()["detail"]
    assert google_repo._oauth_states == {}


def test_start_refuses_with_no_origin_header(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")
    client.put("/api/google/client", json={"client_id": "cid-1", "client_secret": "cs-1"})

    response = client.post("/api/google/oauth/start", json={"label": "work"})
    assert response.status_code == 400
    assert google_repo._oauth_states == {}


def test_start_refuses_with_no_client_configured(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.post(
        "/api/google/oauth/start", json={"label": "work"}, headers={"Origin": _HTTPS_ORIGIN}
    )
    assert response.status_code == 409


def _linked_scenario(monkeypatch, fake_account_repository, fake_plugin_repository, *, label="work"):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")
    client.put("/api/google/client", json={"client_id": "cid-1", "client_secret": "cs-1"})
    return security, account_repo, google_repo, plugin_repo, manager, fake_google, app, client


def _start_link(client, fake_google, *, label="work", relink_account_id=None) -> str:
    payload = {"label": label}
    if relink_account_id is not None:
        payload["relink_account_id"] = relink_account_id
    response = client.post("/api/google/oauth/start", json=payload, headers={"Origin": _HTTPS_ORIGIN})
    assert response.status_code == 200, response.text
    return _state_from_authorization_url(response.json()["authorization_url"])


def test_callback_with_unknown_state_redirects_state_invalid(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    _sec, _acct, _repo, _plugins, _mgr, _google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )

    response = client.get("/api/google/oauth/callback", params={"code": "code-1", "state": "bogus"})
    assert response.status_code == 303
    assert response.headers["location"] == "/google?link_error=state_invalid"


def test_callback_with_a_reused_state_redirects_state_invalid(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    _sec, _acct, google_repo, _plugins, _mgr, fake_google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )
    raw_state = _start_link(client, fake_google)
    fake_google.add_code("code-1", access_token="at-1", refresh_token="rt-1")
    fake_google.add_profile("at-1", "work@example.com")
    fake_google.add_calendar_list("at-1", [])

    first = client.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
    assert first.status_code == 303
    assert first.headers["location"].startswith("/google/accounts/")

    second = client.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
    assert second.status_code == 303
    assert second.headers["location"] == "/google?link_error=state_invalid"


def test_callback_completed_by_a_different_admin_than_started_it_redirects_state_invalid(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """B2-WR-02 regression: `oauth_callback`'s own CSRF defense for a
    multi-admin household -- the state row must also have been created by
    the same admin completing the callback, not merely exist and be
    unused. A regression that dropped or inverted
    `consumed.created_by_user_id != admin.id` (it sits on the same line
    as the `is None` check) would not be caught without this test."""
    _sec, account_repo, _repo, _plugins, _mgr, fake_google, app, client_a = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )
    raw_state = _start_link(client_a, fake_google)
    fake_google.add_code("code-1", access_token="at-1", refresh_token="rt-1")

    client_b = _client_as(app, _sec, account_repo, role="admin")
    response = client_b.get(
        "/api/google/oauth/callback", params={"code": "code-1", "state": raw_state}
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/google?link_error=state_invalid"


def test_callback_with_access_denied_redirects_denied(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    _sec, _acct, _repo, _plugins, _mgr, fake_google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )
    raw_state = _start_link(client, fake_google)

    response = client.get(
        "/api/google/oauth/callback", params={"error": "access_denied", "state": raw_state}
    )
    assert response.status_code == 303
    assert response.headers["location"] == "/google?link_error=denied"


def test_callback_with_missing_scope_revokes_and_redirects_scopes_missing(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    _sec, _acct, google_repo, _plugins, _mgr, fake_google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )
    raw_state = _start_link(client, fake_google)
    fake_google.add_code(
        "code-1",
        access_token="at-1",
        refresh_token="rt-1",
        scope="https://www.googleapis.com/auth/calendar",
    )

    response = client.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
    assert response.status_code == 303
    assert response.headers["location"] == "/google?link_error=scopes_missing"
    assert fake_google.revoked == ["rt-1"]
    assert google_repo._accounts == {}


def test_callback_with_no_refresh_token_redirects_no_refresh_token(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    _sec, _acct, google_repo, _plugins, _mgr, fake_google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )
    raw_state = _start_link(client, fake_google)
    fake_google.add_code("code-1", access_token="at-1", refresh_token=None)

    response = client.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
    assert response.status_code == 303
    assert response.headers["location"] == "/google?link_error=no_refresh_token"
    assert google_repo._accounts == {}


def test_second_link_for_the_same_address_updates_the_existing_row(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    _sec, _acct, google_repo, _plugins, manager, fake_google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )

    raw_state = _start_link(client, fake_google, label="work")
    fake_google.add_code("code-1", access_token="at-1", refresh_token="rt-1")
    fake_google.add_profile("at-1", "work@example.com")
    fake_google.add_calendar_list("at-1", [])
    first = client.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
    assert first.status_code == 303
    [account] = google_repo._accounts.values()
    account_id = account.id

    raw_state2 = _start_link(client, fake_google, label="work", relink_account_id=account_id)
    fake_google.add_code("code-2", access_token="at-2", refresh_token="rt-2")
    fake_google.add_profile("at-2", "work@example.com")
    fake_google.add_calendar_list("at-2", [])
    second = client.get("/api/google/oauth/callback", params={"code": "code-2", "state": raw_state2})
    assert second.status_code == 303
    assert second.headers["location"] == f"/google/accounts/{account_id}?linked=1"

    assert len(google_repo._accounts) == 1
    [updated] = google_repo._accounts.values()
    assert updated.id == account_id
    assert updated.label == "work"
    assert updated.status == "ok"
    plaintext = decrypt_credential(updated.refresh_token_ciphertext, updated.key_version, _sec)
    assert plaintext == "rt-2"
    assert manager.respawn_calls or manager.start_calls.count(account_id) == 0


def test_refresh_token_expires_in_is_stored(monkeypatch, fake_account_repository, fake_plugin_repository):
    _sec, _acct, google_repo, _plugins, _mgr, fake_google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )
    raw_state = _start_link(client, fake_google)
    fake_google.add_code(
        "code-1", access_token="at-1", refresh_token="rt-1", refresh_token_expires_in=604800
    )
    fake_google.add_profile("at-1", "work@example.com")
    fake_google.add_calendar_list("at-1", [])

    response = client.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
    assert response.status_code == 303
    [account] = google_repo._accounts.values()
    assert account.refresh_token_expires_at is not None


def test_a_first_successful_link_starts_the_plugin_a_later_link_respawns_it(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    _sec, _acct, google_repo, _plugins, manager, fake_google, _app, client = _linked_scenario(
        monkeypatch, fake_account_repository, fake_plugin_repository
    )

    raw_state = _start_link(client, fake_google, label="work")
    fake_google.add_code("code-1", access_token="at-1", refresh_token="rt-1")
    fake_google.add_profile("at-1", "work@example.com")
    fake_google.add_calendar_list("at-1", [])
    client.get("/api/google/oauth/callback", params={"code": "code-1", "state": raw_state})
    assert len(manager.start_calls) == 1
    assert manager.respawn_calls == []

    [account] = google_repo._accounts.values()
    raw_state2 = _start_link(client, fake_google, label="work", relink_account_id=account.id)
    fake_google.add_code("code-2", access_token="at-2", refresh_token="rt-2")
    fake_google.add_profile("at-2", "work@example.com")
    fake_google.add_calendar_list("at-2", [])
    client.get("/api/google/oauth/callback", params={"code": "code-2", "state": raw_state2})
    assert manager.respawn_calls == [account.id]


def test_viewer_and_operator_get_403_from_every_route(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)

    for role in ("viewer", "operator"):
        client = _client_as(app, security, account_repo, role=role)
        assert client.get("/api/google/client").status_code == 403
        assert client.put(
            "/api/google/client", json={"client_id": "cid-1", "client_secret": "cs-1"}
        ).status_code == 403
        assert client.post(
            "/api/google/oauth/start", json={"label": "work"}, headers={"Origin": _HTTPS_ORIGIN}
        ).status_code == 403
        assert client.get(
            "/api/google/oauth/callback", params={"code": "code-1", "state": "bogus"}
        ).status_code == 403
        assert client.get("/api/google/accounts").status_code == 403
