"""Labels, the default account, each calendar's own access, refreshing the
calendar list, and unlinking (09-03-PLAN.md, Task 2) -- every route in
`src/atlas/routes/google_accounts.py` beyond linking itself, proven against
`FakeGoogleAccountRepository`, `FakeGoogle`, and a stand-in plugin manager
that records `start_one`/`stop_one`/`request_respawn` and can be told to
fail, the same "primitives in isolation" shape `tests/test_google_oauth.py`
already uses.

Every token, secret, and code literal in this file is shorter than eight
characters, for the identical `tests/test_repo_hygiene.py` reason
`tests/test_google_oauth.py`'s own module docstring states.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from google_fakes import FakeGoogle
from google_repo_fakes import FakeGoogleAccountRepository
from test_google_tracer import google_sessionmaker  # noqa: F401 -- reused Postgres fixture

from atlas_mcp.google_tools import REQUIRED_SCOPES

from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig
from atlas.crypto.credentials import encrypt_credential
from atlas.db.google_postgres import PostgresGoogleAccountRepository
from atlas.google.token_service import GoogleTokenService
from atlas.plugins.manager import PluginState
from atlas.routes.google_accounts import router as google_accounts_router

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


class _FakePluginManagerForGoogle:
    """`tests/test_google_oauth.py::_FakePluginManagerForGoogle`'s own
    sibling here -- duplicated rather than imported, matching this
    codebase's own per-test-file fake convention (`tests/test_plugin_routes.py`'s
    `_FakePluginManagerForRoutes` is never imported elsewhere either)."""

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


def _run(coro):
    return asyncio.run(coro)


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


def _client_as(app, security, account_repo, *, role: str) -> TestClient:
    user = _run(
        account_repo.create_user(
            email=f"{role}@example.invalid",
            display_name=f"A {role.title()}",
            password_hash="not-checked-by-this-test",
            role=role,
        )
    )
    token = issue_access_token(user_id=user.id, role=role, security=security)
    return TestClient(app, cookies={security.cookie_name: token}, follow_redirects=False)


def _seed_client(google_repo, security) -> None:
    ciphertext, key_version = encrypt_credential("cs-1", security)
    _run(
        google_repo.set_oauth_client(
            client_id="cid-1",
            client_secret_ciphertext=ciphertext,
            key_version=key_version,
            updated_by_user_id=None,
            updated_at=datetime.now(timezone.utc),
        )
    )


def _seed_account(google_repo, security, *, label="work", email="work@example.com", refresh_token="rt-1"):
    ciphertext, key_version = encrypt_credential(refresh_token, security)
    return _run(
        google_repo.insert_account(
            label=label,
            email=email,
            refresh_token_ciphertext=ciphertext,
            key_version=key_version,
            granted_scopes=" ".join(REQUIRED_SCOPES),
            refresh_token_expires_at=None,
            linked_by_user_id=None,
            linked_at=datetime.now(timezone.utc),
        )
    )


def _seed_calendar(google_repo, account_id, *, calendar_id="cal-1", name="Cal One", can_write=True):
    _run(
        google_repo.add_calendars(
            account_id,
            [(calendar_id, name, True, can_write)],
            discovered_at=datetime.now(timezone.utc),
        )
    )
    [account] = [a for a in _run(google_repo.list_accounts()) if a.id == account_id]
    return next(c for c in account.calendars if c.google_calendar_id == calendar_id)


def test_get_unknown_account_is_a_named_404(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.get("/api/google/accounts/999")
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_get_account_returns_its_calendars(monkeypatch, fake_account_repository, fake_plugin_repository):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security)
    _seed_calendar(google_repo, account.id)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.get(f"/api/google/accounts/{account.id}")
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["id"] == account.id
    [calendar] = body["calendars"]
    assert calendar["calendar_id"] == "cal-1"


def test_patch_label_refuses_a_label_another_account_uses(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account1 = _seed_account(google_repo, security, label="work", email="work@example.com")
    account2 = _seed_account(google_repo, security, label="home", email="home@example.com")

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.patch(f"/api/google/accounts/{account2.id}", json={"label": "work"})
    assert response.status_code == 409

    renamed = client.patch(f"/api/google/accounts/{account2.id}", json={"label": "personal"})
    assert renamed.status_code == 200, renamed.text
    assert renamed.json()["label"] == "personal"


def test_patch_is_default_true_makes_this_account_the_only_default(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account1 = _seed_account(google_repo, security, label="work", email="work@example.com")
    account2 = _seed_account(google_repo, security, label="home", email="home@example.com")

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    first = client.patch(f"/api/google/accounts/{account1.id}", json={"is_default": True})
    assert first.status_code == 200, first.text
    assert first.json()["is_default"] is True

    second = client.patch(f"/api/google/accounts/{account2.id}", json={"is_default": True})
    assert second.status_code == 200, second.text
    assert second.json()["is_default"] is True

    refreshed_first = client.get(f"/api/google/accounts/{account1.id}")
    assert refreshed_first.json()["is_default"] is False, "only one account may be default at a time"

    cleared = client.patch(f"/api/google/accounts/{account2.id}", json={"is_default": False})
    assert cleared.status_code == 200, cleared.text
    assert cleared.json()["is_default"] is False


def test_put_calendar_access_stores_it_and_respawns(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security)
    calendar = _seed_calendar(google_repo, account.id, can_write=True)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.put(
        f"/api/google/accounts/{account.id}/calendars/{calendar.id}", json={"access": "read_only"}
    )
    assert response.status_code == 200, response.text
    [returned_calendar] = response.json()["calendars"]
    assert returned_calendar["access"] == "read_only"
    assert manager.start_calls, "the first write must start the google plugin"


def test_put_calendar_access_refuses_read_write_when_google_marks_it_read_only(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security)
    calendar = _seed_calendar(google_repo, account.id, can_write=False)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.put(
        f"/api/google/accounts/{account.id}/calendars/{calendar.id}", json={"access": "read_write"}
    )
    assert response.status_code == 400
    assert "read_write" in response.json()["detail"] or "read-only" in response.json()["detail"]


def test_put_calendar_access_404s_for_a_calendar_of_another_account(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account1 = _seed_account(google_repo, security, label="work", email="work@example.com")
    account2 = _seed_account(google_repo, security, label="home", email="home@example.com")
    calendar = _seed_calendar(google_repo, account1.id)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.put(
        f"/api/google/accounts/{account2.id}/calendars/{calendar.id}", json={"access": "read_only"}
    )
    assert response.status_code == 404


def test_a_failed_respawn_after_narrowing_stops_the_child_and_answers_503(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security)
    calendar = _seed_calendar(google_repo, account.id, can_write=True)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    # Widen first so the plugin is RUNNING -- a narrowing reconcile then
    # goes through `request_respawn`, not `start_one`.
    widen = client.put(
        f"/api/google/accounts/{account.id}/calendars/{calendar.id}", json={"access": "read_write"}
    )
    assert widen.status_code == 200, widen.text
    assert manager.states["google"] is PluginState.RUNNING

    manager.fail_respawn = True
    response = client.put(
        f"/api/google/accounts/{account.id}/calendars/{calendar.id}", json={"access": "off"}
    )
    assert response.status_code == 503
    assert "stopped" in response.json()["detail"]
    assert manager.stop_calls, "a failed reconcile after narrowing must stop the child"
    assert manager.states["google"] is PluginState.DISABLED


def test_a_failed_respawn_after_widening_answers_503_without_stopping_anything(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security)
    calendar = _seed_calendar(google_repo, account.id, can_write=True)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    # Any successful write starts the plugin -- a label rename is enough
    # and touches no calendar access at all.
    started = client.patch(f"/api/google/accounts/{account.id}", json={"label": "renamed"})
    assert started.status_code == 200, started.text
    assert manager.states["google"] is PluginState.RUNNING

    manager.fail_respawn = True
    response = client.put(
        f"/api/google/accounts/{account.id}/calendars/{calendar.id}", json={"access": "read_only"}
    )
    assert response.status_code == 503
    assert "stopped" not in response.json()["detail"]
    assert manager.stop_calls == [], "a failed reconcile after widening must not stop the child"
    assert manager.states["google"] is PluginState.RUNNING


def test_refresh_calendars_adds_new_ones_off_and_leaves_access_untouched(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security, refresh_token="rt-1")
    calendar = _seed_calendar(google_repo, account.id, calendar_id="cal-1", name="Old Name")
    _run(google_repo.set_calendar_access(calendar.id, "read_only", updated_at=datetime.now(timezone.utc)))
    fake_google.add_refresh_token("rt-1", "at-1")
    fake_google.add_calendar_list(
        "at-1",
        [
            {"id": "cal-1", "summary": "New Name", "primary": True, "accessRole": "owner"},
            {"id": "cal-2", "summary": "Second Cal", "accessRole": "reader"},
        ],
    )

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.post(f"/api/google/accounts/{account.id}/calendars/refresh")
    assert response.status_code == 200, response.text
    by_id = {c["calendar_id"]: c for c in response.json()["calendars"]}
    assert by_id["cal-1"]["name"] == "New Name"
    assert by_id["cal-1"]["access"] == "read_only", "refresh must never touch a stored access value"
    assert by_id["cal-2"]["access"] == "off", "a newly discovered calendar always starts off"


def test_delete_account_revokes_deletes_and_respawns(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security, refresh_token="rt-1")
    _seed_calendar(google_repo, account.id)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.delete(f"/api/google/accounts/{account.id}")
    assert response.status_code == 204
    assert fake_google.revoked == ["rt-1"]
    assert google_repo._accounts == {}
    assert manager.start_calls, "the unlink's own reconcile must still reach the child"

    listing = client.get("/api/google/accounts")
    assert listing.json() == []


def test_delete_account_unlinks_even_when_revoke_fails(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """A revoke failure must never block the unlink -- `revoke_token`
    itself never raises, so this proves the route never gates the delete
    on the revoke response at all."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security, refresh_token="rt-1")

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.delete(f"/api/google/accounts/{account.id}")
    assert response.status_code == 204
    assert google_repo._accounts == {}


def test_delete_account_unlinks_even_when_the_stored_ciphertext_cannot_be_decrypted(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """B1-WR-03 regression: a corrupted refresh-token ciphertext (or an
    incompatible/rotated secret key) must never leave the row permanently
    stuck -- `decrypt_credential`'s own `InvalidToken` is caught the same
    best-effort way a revoke failure already is, and the row is still
    removed. `revoke_token` is never reached: there is no refresh token to
    revoke with."""
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    # Deliberately not `encrypt_credential`'s own output -- an arbitrary
    # byte string a real Fernet token can never decrypt, standing in for a
    # corrupted ciphertext or a key rotated out from under it.
    account = _run(
        google_repo.insert_account(
            label="work",
            email="work@example.com",
            refresh_token_ciphertext=b"not-a-real-fernet-token",
            key_version=1,
            granted_scopes=" ".join(REQUIRED_SCOPES),
            refresh_token_expires_at=None,
            linked_by_user_id=None,
            linked_at=datetime.now(timezone.utc),
        )
    )

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)
    client = _client_as(app, security, account_repo, role="admin")

    response = client.delete(f"/api/google/accounts/{account.id}")
    assert response.status_code == 204
    assert google_repo._accounts == {}
    assert fake_google.revoked == []


def test_every_task_2_route_is_403_for_viewer_and_operator(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    google_repo = FakeGoogleAccountRepository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForGoogle()
    fake_google = FakeGoogle()
    _seed_client(google_repo, security)
    account = _seed_account(google_repo, security)
    calendar = _seed_calendar(google_repo, account.id)

    app = _build_app(security, account_repo, google_repo, plugin_repo, manager, fake_google)

    for role in ("viewer", "operator"):
        client = _client_as(app, security, account_repo, role=role)
        assert client.get(f"/api/google/accounts/{account.id}").status_code == 403
        assert client.patch(f"/api/google/accounts/{account.id}", json={"label": "x"}).status_code == 403
        assert (
            client.put(
                f"/api/google/accounts/{account.id}/calendars/{calendar.id}", json={"access": "off"}
            ).status_code
            == 403
        )
        assert client.post(f"/api/google/accounts/{account.id}/calendars/refresh").status_code == 403
        assert client.delete(f"/api/google/accounts/{account.id}").status_code == 403


# --- Postgres-backed proof -------------------------------------------


async def test_postgres_update_account_is_default_clears_every_other_default(google_sessionmaker):
    """The acceptance criterion this plan names explicitly: a real
    Postgres `update_account(..., is_default=True)` clears every other
    account's own default flag in the same transaction (D-04) -- proven
    against `PostgresGoogleAccountRepository` directly, not the fake."""
    security = SecurityConfig()
    repo = PostgresGoogleAccountRepository(google_sessionmaker)

    async def _insert(label: str, email: str):
        ciphertext, key_version = encrypt_credential("rt-1", security)
        return await repo.insert_account(
            label=label,
            email=email,
            refresh_token_ciphertext=ciphertext,
            key_version=key_version,
            granted_scopes=" ".join(REQUIRED_SCOPES),
            refresh_token_expires_at=None,
            linked_by_user_id=None,
            linked_at=datetime.now(timezone.utc),
        )

    account1 = await _insert("work", "work@example.com")
    account2 = await _insert("home", "home@example.com")

    now = datetime.now(timezone.utc)
    await repo.update_account(account1.id, is_default=True, at=now)
    [refreshed1] = [a for a in await repo.list_accounts() if a.id == account1.id]
    assert refreshed1.is_default is True

    await repo.update_account(account2.id, is_default=True, at=now)
    accounts_by_id = {a.id: a for a in await repo.list_accounts()}
    assert accounts_by_id[account2.id].is_default is True
    assert accounts_by_id[account1.id].is_default is False, (
        "setting a new default must clear every other account's own default flag "
        "in the same transaction"
    )
