"""Plugin authoring over HTTP (D-01 .. D-16, PLUG-03, PLUG-04, PLUG-08).

Every test here builds a small, throwaway `FastAPI()` app carrying only
`spire_voice.routes.plugins`'s own router -- the same "primitives in
isolation" shape `tests/test_policy_routes.py` already uses, since this
file's whole point is the plugin routes themselves, not the rest of the
application.

Plan 06-06, Task 2's own scope: the repository write half and every named
refusal, proven here against `FakePluginRepository`. Live-effect proof
(D-15: install/enable/disable/config-save reaching the running assistant
before the route returns) is proven against `_FakePluginManagerForRoutes`
below for this task, and against the real `PluginManager` in Task 3's own
additional tests in this same file.
"""

from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager, contextmanager
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.testclient import TestClient
from mcp.types import Tool

from spire_voice.auth.tokens import issue_access_token
from spire_voice.config import SecurityConfig
from spire_voice.plugins import manager as manager_module
from spire_voice.plugins.manager import PluginManager, PluginState
from spire_voice.routes.plugins import router as plugins_router

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_ROOT = os.path.join(_REPO_ROOT, "mcp")


async def _no_policy():
    return None

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


class _FakePluginManagerForRoutes:
    """Stands in for `PluginManager` at exactly the surface
    `routes/plugins.py` reads/calls -- read methods return whatever a test
    seeded, `start_one`/`stop_one` record their own calls and optionally
    raise, matching `tests/test_policy_routes.py`'s own
    `_ManagerStandingInForRespawn` precedent for the identical reason: the
    real manager's persistent-lifecycle-task machinery is out of scope for
    a route-layer test, which only needs to prove the route calls the
    right method and awaits it."""

    def __init__(self) -> None:
        self.states: dict[str, PluginState] = {}
        self.reasons: dict[str, str] = {}
        self.hosts: dict[str, object] = {}
        self.owners: dict[str, tuple[str, ...]] = {}
        self.start_calls: list[int] = []
        self.stop_calls: list[int] = []
        # WR-03 (code review): deleting a plugin must drop the manager's
        # own bookkeeping for it, not merely stop it -- recorded
        # separately from `stop_calls` so a test can tell the two writes
        # apart.
        self.forget_calls: list[int] = []
        self.fail_reconcile = False

    def state_for(self, slug):
        return self.states.get(slug)

    def reason_for(self, slug):
        return self.reasons.get(slug)

    def tool_host_for(self, slug):
        return self.hosts.get(slug)

    def owners_of_bare_name(self, name):
        return self.owners.get(name, ())

    async def start_one(self, plugin):
        if self.fail_reconcile:
            raise RuntimeError("simulated reconcile failure")
        self.start_calls.append(plugin.id)
        self.states[plugin.slug] = PluginState.RUNNING

    async def stop_one(self, plugin):
        if self.fail_reconcile:
            raise RuntimeError("simulated reconcile failure")
        self.stop_calls.append(plugin.id)
        self.states[plugin.slug] = PluginState.DISABLED

    async def forget(self, plugin_id):
        if self.fail_reconcile:
            raise RuntimeError("simulated reconcile failure")
        self.forget_calls.append(plugin_id)


def _build_plugins_app(security, account_repo, plugin_repo, plugin_manager) -> FastAPI:
    app = FastAPI()
    app.state.config = SimpleNamespace(security=security)
    app.state.account_repo = account_repo
    app.state.plugin_repo = plugin_repo
    app.state.plugin_manager = plugin_manager
    app.include_router(plugins_router)
    return app


def _issue_cookie(security, account_repo, *, role: str):
    user = asyncio.run(
        account_repo.create_user(
            email=f"{role}@example.invalid",
            display_name=f"A {role.title()}",
            password_hash="not-checked-by-this-test",
            role=role,
        )
    )
    return user


def _admin_client(app, security, account_repo):
    admin = _issue_cookie(security, account_repo, role="admin")
    token = issue_access_token(user_id=admin.id, role="admin", security=security)
    return TestClient(app, cookies={security.cookie_name: token})


def test_listing_plugins_never_returns_a_secret_value(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.crypto.credentials import encrypt_credential
    from spire_voice.db.repository import Plugin
    from datetime import datetime, timezone

    ciphertext, key_version = encrypt_credential("a-real-secret-token", security)
    now = datetime.now(timezone.utc)
    ha = Plugin(
        id=1, slug="ha", display_name="Home Assistant", transport="stdio", args=("-m", "spire_mcp.ha"),
        url=None, enabled=True, builtin=True, enforces_policy=True, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(
        plugins=[ha],
        config_values={
            1: [
                __import__("spire_voice.db.repository", fromlist=["PluginConfigValue"]).PluginConfigValue(
                    key="HA_TOKEN", secret=True, value=None, ciphertext=ciphertext, key_version=key_version
                ),
            ]
        },
    )
    manager = _FakePluginManagerForRoutes()
    manager.states["ha"] = PluginState.RUNNING

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/plugins")
    assert response.status_code == 200
    [entry] = response.json()
    [config_entry] = entry["config_values"]
    assert config_entry["key"] == "HA_TOKEN"
    assert config_entry["is_set"] is True
    assert config_entry["value"] is None
    assert "a-real-secret-token" not in response.text
    assert str(ciphertext) not in response.text


def test_get_unknown_plugin_is_a_named_404(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.get("/api/plugins/999")
    assert response.status_code == 404
    assert "999" in response.json()["detail"]


def test_installing_from_the_catalog_creates_a_row_and_reconciles_live(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={
            "catalog_entry": "Weather",
            "config_values": {"WEATHER_LATITUDE": {"value": "51.5"}},
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["transport"] == "stdio"
    assert body["args"] == ["-m", "spire_mcp.weather"]
    assert body["display_name"] == "Weather"
    assert body["slug"] == "weather"
    assert body["builtin"] is False

    by_key = {v["key"]: v for v in body["config_values"]}
    assert by_key["WEATHER_LATITUDE"]["value"] == "51.5"
    assert by_key["WEATHER_LONGITUDE"]["value"] == ""
    assert manager.start_calls == [body["id"]]


def test_installing_from_the_catalog_with_a_secret_left_blank_is_not_set(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post("/api/plugins", json={"catalog_entry": "Home Assistant"})
    assert response.status_code == 201, response.text
    body = response.json()
    by_key = {v["key"]: v for v in body["config_values"]}
    assert by_key["HA_TOKEN"]["secret"] is True
    assert by_key["HA_TOKEN"]["is_set"] is False
    assert by_key["HA_TOKEN"]["value"] is None


def test_installing_from_the_catalog_refuses_an_undeclared_config_key(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={"catalog_entry": "Weather", "config_values": {"NOT_A_REAL_KEY": {"value": "x"}}},
    )
    assert response.status_code == 400
    assert not plugin_repo.plugins


def test_installing_an_unknown_catalog_entry_is_refused_by_name(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post("/api/plugins", json={"catalog_entry": "Does Not Exist"})
    assert response.status_code == 400
    assert "Does Not Exist" in response.json()["detail"]


def test_installing_by_a_hand_entered_command_creates_a_stdio_row(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={
            "display_name": "Example Custom Plugin",
            "transport": "command",
            "command": "-m example_custom_module",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["transport"] == "stdio"
    assert body["args"] == ["-m", "example_custom_module"]


def test_a_command_naming_an_interpreter_is_refused(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={
            "display_name": "Bad Plugin",
            "transport": "command",
            "command": "python3 -m example_custom_module",
        },
    )
    assert response.status_code == 400
    assert not plugin_repo.plugins


def test_installing_by_a_hand_entered_url_creates_a_remote_row(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={
            "display_name": "Example Remote Plugin",
            "transport": "url",
            "url": "https://example.invalid/mcp",
        },
    )
    assert response.status_code == 201, response.text
    body = response.json()
    assert body["transport"] == "remote"
    assert body["url"] == "https://example.invalid/mcp"
    assert body["args"] == []


def test_a_plain_http_url_on_a_non_loopback_host_is_refused(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={
            "display_name": "Example Remote Plugin",
            "transport": "url",
            "url": "http://example.invalid/mcp",
        },
    )
    assert response.status_code == 400
    assert not plugin_repo.plugins


def test_installing_with_neither_a_catalog_entry_nor_a_custom_source_is_refused(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post("/api/plugins", json={"display_name": "Nothing Given"})
    assert response.status_code == 400


def test_installing_with_both_a_catalog_entry_and_a_command_is_refused(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={"catalog_entry": "Weather", "transport": "command", "command": "-m x"},
    )
    assert response.status_code == 400
    assert not plugin_repo.plugins


def test_a_request_naming_enforces_policy_is_rejected_with_422(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """T-06-28: no route may set the policy-enforcing flag -- enforced
    structurally, since no request model carries the field at all."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={
            "display_name": "Sneaky Plugin",
            "transport": "command",
            "command": "-m example_custom_module",
            "enforces_policy": True,
        },
    )
    assert response.status_code == 422
    assert not plugin_repo.plugins


def test_enabling_and_disabling_flip_the_row_and_reconcile_live(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1, slug="example", display_name="Example", transport="stdio", args=("-m", "example_module"),
        url=None, enabled=True, builtin=False, enforces_policy=False, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(plugins=[plugin])
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.put("/api/plugins/1/enabled", json={"enabled": False})
    assert response.status_code == 200
    assert response.json()["enabled"] is False
    assert manager.stop_calls == [1]
    assert plugin_repo.plugins[1].enabled is False
    # Nothing else about the row changed.
    assert plugin_repo.plugins[1].display_name == "Example"

    response = client.put("/api/plugins/1/enabled", json={"enabled": True})
    assert response.status_code == 200
    assert response.json()["enabled"] is True
    assert manager.start_calls == [1]


def test_saving_configuration_writes_plain_and_encrypts_secret(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1, slug="example", display_name="Example", transport="stdio", args=("-m", "example_module"),
        url=None, enabled=True, builtin=False, enforces_policy=False, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(plugins=[plugin])
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.put(
        "/api/plugins/1/config",
        json={"values": {"PLAIN_KEY": {"value": "plain-value"}, "SECRET_KEY": {"value": "hunter2", "secret": True}}},
    )
    assert response.status_code == 200
    values = plugin_repo.config_values[1]
    by_key = {v.key: v for v in values}
    assert by_key["PLAIN_KEY"].value == "plain-value"
    assert by_key["SECRET_KEY"].secret is True
    assert by_key["SECRET_KEY"].value is None
    assert by_key["SECRET_KEY"].ciphertext is not None
    assert "hunter2" not in response.text
    assert manager.start_calls == [1]


def test_saving_a_blank_secret_leaves_an_already_set_value_alone(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.crypto.credentials import encrypt_credential
    from spire_voice.db.repository import Plugin, PluginConfigValue
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1, slug="example", display_name="Example", transport="stdio", args=("-m", "example_module"),
        url=None, enabled=True, builtin=False, enforces_policy=False, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    ciphertext, key_version = encrypt_credential("already-set-secret", security)
    plugin_repo = fake_plugin_repository(
        plugins=[plugin],
        config_values={1: [PluginConfigValue(key="SECRET_KEY", secret=True, value=None, ciphertext=ciphertext, key_version=key_version)]},
    )
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.put("/api/plugins/1/config", json={"values": {"SECRET_KEY": {"value": "", "secret": True}}})
    assert response.status_code == 200
    [value] = plugin_repo.config_values[1]
    assert value.ciphertext == ciphertext, "a blank secret submission must leave the existing value alone"


def test_saving_configuration_on_a_disabled_plugin_does_not_reconcile(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1, slug="example", display_name="Example", transport="stdio", args=("-m", "example_module"),
        url=None, enabled=False, builtin=False, enforces_policy=False, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(plugins=[plugin])
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.put("/api/plugins/1/config", json={"values": {"KEY": {"value": "v"}}})
    assert response.status_code == 200
    assert manager.start_calls == []
    assert manager.stop_calls == []


def test_deleting_a_non_builtin_plugin_removes_it_and_reconciles_live(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1, slug="example", display_name="Example", transport="stdio", args=("-m", "example_module"),
        url=None, enabled=True, builtin=False, enforces_policy=False, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(plugins=[plugin])
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.delete("/api/plugins/1")
    assert response.status_code == 204
    assert 1 not in plugin_repo.plugins
    # WR-03 (code review): a delete forgets the row rather than recording
    # it disabled -- a `DISABLED` entry for a dead id shadows a plugin
    # later reinstalled under the same display name (and so the same slug).
    assert manager.forget_calls == [1]
    assert manager.stop_calls == []


def test_deleting_a_builtin_plugin_is_refused_by_name_and_it_still_exists(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ha = Plugin(
        id=1, slug="ha", display_name="Home Assistant", transport="stdio", args=("-m", "spire_mcp.ha"),
        url=None, enabled=True, builtin=True, enforces_policy=True, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(plugins=[ha])
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.delete("/api/plugins/1")
    assert response.status_code == 400
    assert "Home Assistant" in response.json()["detail"]
    assert 1 in plugin_repo.plugins
    assert manager.stop_calls == []
    assert manager.forget_calls == []


def test_a_reconcile_failure_is_reported_as_a_failed_write(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()
    manager.fail_reconcile = True

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post("/api/plugins", json={"catalog_entry": "Weather"})
    assert response.status_code == 502


def test_every_plugin_route_refuses_an_operator(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    operator = _issue_cookie(security, account_repo, role="operator")
    token = issue_access_token(user_id=operator.id, role="operator", security=security)
    client = TestClient(app, cookies={security.cookie_name: token})

    assert client.get("/api/plugins").status_code == 403
    assert client.get("/api/plugins/1").status_code == 403
    assert client.post("/api/plugins", json={"catalog_entry": "Weather"}).status_code == 403
    assert client.put("/api/plugins/1/enabled", json={"enabled": False}).status_code == 403
    assert client.put("/api/plugins/1/config", json={"values": {}}).status_code == 403
    assert client.delete("/api/plugins/1").status_code == 403


def test_every_plugin_route_refuses_a_viewer(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    viewer = _issue_cookie(security, account_repo, role="viewer")
    token = issue_access_token(user_id=viewer.id, role="viewer", security=security)
    client = TestClient(app, cookies={security.cookie_name: token})

    assert client.get("/api/plugins").status_code == 403
    assert client.post("/api/plugins", json={"catalog_entry": "Weather"}).status_code == 403


# --- Task 3: live effect against the real PluginManager --------------------
#
# Every test below spawns the real `spire_mcp.ha`/`spire_mcp.weather` child
# already shipped in this repository's own `mcp/` directory, the same
# real-subprocess discipline `tests/test_plugin_manager.py` already uses --
# proving install/enable/disable/config-save actually reach a real running
# child through the HTTP layer, not merely that the route calls a method a
# fake recorded.
#
# `TestClient` used as a context manager runs the app's own `lifespan`
# (FastAPI's own testing docs) -- here, only its shutdown half matters:
# `manager.stop_all()`, run on the exact same portal loop every request in
# this `with` block ran on, so every plugin's own lifecycle task (and the
# real subprocess it owns) is cancelled and awaited cleanly rather than
# leaked past the end of the test.


@contextmanager
def _real_plugins_client(security, account_repo, plugin_repo):
    manager = PluginManager(
        plugin_repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_no_policy
    )

    @asynccontextmanager
    async def _lifespan(_app):
        yield
        await manager.stop_all()

    app = FastAPI(lifespan=_lifespan)
    app.state.config = SimpleNamespace(security=security)
    app.state.account_repo = account_repo
    app.state.plugin_repo = plugin_repo
    app.state.plugin_manager = manager
    app.include_router(plugins_router)

    admin = asyncio.run(
        account_repo.create_user(
            email="admin@example.invalid", display_name="An Admin",
            password_hash="not-checked-by-this-test", role="admin",
        )
    )
    token = issue_access_token(user_id=admin.id, role="admin", security=security)
    with TestClient(app, cookies={security.cookie_name: token}) as client:
        yield client, manager


def test_installing_a_plugin_starts_it_and_its_tools_reach_the_schema_before_returning(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()

    with _real_plugins_client(security, account_repo, plugin_repo) as (client, manager):
        response = client.post(
            "/api/plugins",
            json={
                "catalog_entry": "Weather",
                "config_values": {"WEATHER_LATITUDE": {"value": "51.5"}, "WEATHER_LONGITUDE": {"value": "-0.1"}},
            },
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["state"] == "running"
        assert {tool["name"] for tool in body["tools"]}

        tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert tool_names, "the newly installed plugin's tools must already be in the schema"


def test_disabling_a_plugin_stops_it_and_withdraws_its_tools_before_returning(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()

    with _real_plugins_client(security, account_repo, plugin_repo) as (client, manager):
        installed = client.post(
            "/api/plugins",
            json={
                "catalog_entry": "Weather",
                "config_values": {"WEATHER_LATITUDE": {"value": "51.5"}, "WEATHER_LONGITUDE": {"value": "-0.1"}},
            },
        ).json()
        plugin_id = installed["id"]
        assert manager.tools_schema

        response = client.put(f"/api/plugins/{plugin_id}/enabled", json={"enabled": False})
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == "disabled"
        assert body["tools"] == []
        assert manager.tools_schema == [], "a disabled plugin's tools must already be withdrawn"


def test_enabling_a_disabled_plugin_restarts_it_and_restores_its_tools(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()

    with _real_plugins_client(security, account_repo, plugin_repo) as (client, manager):
        installed = client.post(
            "/api/plugins",
            json={
                "catalog_entry": "Weather",
                "config_values": {"WEATHER_LATITUDE": {"value": "51.5"}, "WEATHER_LONGITUDE": {"value": "-0.1"}},
            },
        ).json()
        plugin_id = installed["id"]
        client.put(f"/api/plugins/{plugin_id}/enabled", json={"enabled": False})
        assert manager.tools_schema == []

        response = client.put(f"/api/plugins/{plugin_id}/enabled", json={"enabled": True})
        assert response.status_code == 200
        body = response.json()
        assert body["state"] == "running"
        assert body["tools"]
        assert manager.tools_schema


def test_saving_configuration_restarts_the_plugin_with_the_new_configuration(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """D-15's own worked example: a plugin installed with a blank secret,
    then configured afterward, must be running under the value just
    saved -- captured directly from the real child's own spawned
    environment, the same interception point `tests/test_plugin_manager.py`
    already uses."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()

    with _real_plugins_client(security, account_repo, plugin_repo) as (client, manager):
        installed = client.post(
            "/api/plugins",
            json={"catalog_entry": "Home Assistant", "config_values": {"HA_URL": {"value": "http://ha.invalid:8123"}}},
        ).json()
        plugin_id = installed["id"]

        captured_envs: list[dict] = []
        real_spawn = manager_module.McpToolHost._spawn

        async def _capturing_spawn(self, child_module, env):
            captured_envs.append(dict(env))
            await real_spawn(self, child_module, env)

        monkeypatch.setattr(manager_module.McpToolHost, "_spawn", _capturing_spawn)

        response = client.put(
            f"/api/plugins/{plugin_id}/config",
            json={"values": {"HA_TOKEN": {"value": "a-freshly-saved-token", "secret": True}}},
        )
        assert response.status_code == 200
        assert captured_envs, "saving configuration must restart the plugin's own child"
        assert captured_envs[-1]["HA_TOKEN"] == "a-freshly-saved-token"
        assert captured_envs[-1]["HA_URL"] == "http://ha.invalid:8123"


def test_a_newly_installed_plugin_that_will_not_start_is_still_created_and_reported_degraded(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()

    async def _always_fails(*args, **kwargs):
        raise RuntimeError("simulated: this plugin will never start")

    monkeypatch.setattr(manager_module, "start_plugin_host", _always_fails)

    with _real_plugins_client(security, account_repo, plugin_repo) as (client, manager):
        response = client.post(
            "/api/plugins",
            json={"display_name": "Never Starts", "transport": "command", "command": "-m spire_mcp.weather"},
        )
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["state"] == "degraded"
        assert "simulated: this plugin will never start" in body["reason"]
        assert plugin_repo.plugins  # the row is not lost


def test_editing_one_plugins_configuration_leaves_a_second_plugins_session_untouched(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """Task 3's own instruction: prove it -- edit one plugin's
    configuration and assert a second plugin's session is the same
    object afterwards."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()

    with _real_plugins_client(security, account_repo, plugin_repo) as (client, manager):
        ha = client.post(
            "/api/plugins",
            json={
                "catalog_entry": "Home Assistant",
                "config_values": {
                    "HA_URL": {"value": "http://ha.invalid:8123"},
                    "HA_TOKEN": {"value": "t", "secret": True},
                },
            },
        ).json()
        weather = client.post(
            "/api/plugins",
            json={
                "catalog_entry": "Weather",
                "config_values": {"WEATHER_LATITUDE": {"value": "51.5"}, "WEATHER_LONGITUDE": {"value": "-0.1"}},
            },
        ).json()

        weather_host_before = manager.tool_host_for(weather["slug"])
        assert weather_host_before is not None

        response = client.put(
            f"/api/plugins/{ha['id']}/config",
            json={"values": {"HA_URL": {"value": "http://ha.invalid:9999"}}},
        )
        assert response.status_code == 200

        weather_host_after = manager.tool_host_for(weather["slug"])
        assert weather_host_after is weather_host_before, "editing HA's config must not touch weather's own host"


def test_reinstalling_a_deleted_plugin_under_the_same_name_is_reported_as_running(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """WR-03 (code review): `stop_one` recorded a `DISABLED` bookkeeping
    entry for the deleted row and nothing ever removed one, while slugs
    are derived from the display name over the *current* rows -- so
    reinstalling a plugin with the same name reused the same slug under a
    new id, and `state_for`/`reason_for`/`tool_host_for` (which scan by
    slug, first match wins, insertion order) all answered for the dead
    row. `/api/plugins` then reported "Disabled" and "No tools right now"
    for a plugin that was genuinely running, and offered an Enable button
    for it. Deleting now forgets the row instead.
    """
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()

    install_body = {
        "catalog_entry": "Weather",
        "config_values": {"WEATHER_LATITUDE": {"value": "51.5"}, "WEATHER_LONGITUDE": {"value": "-0.1"}},
    }

    with _real_plugins_client(security, account_repo, plugin_repo) as (client, manager):
        first = client.post("/api/plugins", json=install_body).json()
        assert client.delete(f"/api/plugins/{first['id']}").status_code == 204

        second = client.post("/api/plugins", json=install_body)
        assert second.status_code == 201, second.text
        body = second.json()
        assert body["id"] != first["id"]
        assert body["slug"] == first["slug"], "the reinstall must reuse the freed slug"
        assert body["state"] == "running", (
            "the reinstalled plugin is running, but the deleted row's stale "
            "bookkeeping entry answered for its slug first (the WR-03 defect)"
        )
        assert body["tools"], "a running plugin's tools are not 'No tools right now'"

        listed = client.get("/api/plugins").json()
        assert [entry["state"] for entry in listed] == ["running"]

        # The manager keeps no state for a row that no longer exists.
        assert list(manager._plugins) == [body["id"]]


def test_a_save_cannot_rewrite_a_stored_secret_as_a_plaintext_row(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """WR-05 (code review): the route took `secret` straight off the
    request body and `set_config_values` overwrites `secret`/`value`/
    `ciphertext` wholesale, so a request naming an existing secret key
    with `"secret": false` replaced the encrypted row with a plaintext one
    -- and every later `GET /api/plugins` then returned that value, into
    the browser's query cache included. The value has to be supplied, so
    it is not a read primitive; it is worse in kind than that, because it
    silently defeats encryption at rest for the house's Home Assistant
    token and turns a write-only field into a readable one (D-03,
    PROV-04, T-06-27).

    The stored row is the authority now, and a request that contradicts it
    is refused by name rather than coerced.
    """
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.crypto.credentials import encrypt_credential
    from spire_voice.db.repository import Plugin, PluginConfigValue
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ha = Plugin(
        id=1, slug="ha", display_name="Home Assistant", transport="stdio", args=("-m", "spire_mcp.ha"),
        url=None, enabled=False, builtin=True, enforces_policy=True, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    ciphertext, key_version = encrypt_credential("a-plainly-fictional-hub-token", security)
    plugin_repo = fake_plugin_repository(
        plugins=[ha],
        config_values={
            1: [
                PluginConfigValue(
                    key="HA_TOKEN", secret=True, value=None,
                    ciphertext=ciphertext, key_version=key_version,
                )
            ]
        },
    )
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.put(
        "/api/plugins/1/config",
        json={"values": {"HA_TOKEN": {"value": "a-plainly-fictional-hub-token", "secret": False}}},
    )
    assert response.status_code == 409, response.text
    assert "HA_TOKEN" in response.json()["detail"]

    [stored] = plugin_repo.config_values[1]
    assert stored.secret is True, "the stored key was reclassified as a plain value"
    assert stored.value is None
    assert stored.ciphertext == ciphertext

    listing = client.get("/api/plugins")
    assert listing.status_code == 200
    assert "a-plainly-fictional-hub-token" not in listing.text
    [entry] = listing.json()[0]["config_values"]
    assert entry == {"key": "HA_TOKEN", "secret": True, "value": None, "is_set": True}


def test_a_save_cannot_reclassify_a_plain_key_as_secret_either(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """The same rule in the other direction -- a save changes a value,
    never whether it is secret. Reclassifying silently would leave the
    editor showing a key it can no longer read back, with no record of why."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin, PluginConfigValue
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1, slug="example", display_name="Example", transport="stdio", args=("-m", "example_module"),
        url=None, enabled=False, builtin=False, enforces_policy=False, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(
        plugins=[plugin],
        config_values={
            1: [PluginConfigValue(key="PLAIN_KEY", secret=False, value="v", ciphertext=None, key_version=None)]
        },
    )
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.put(
        "/api/plugins/1/config",
        json={"values": {"PLAIN_KEY": {"value": "w", "secret": True}}},
    )
    assert response.status_code == 409, response.text
    [stored] = plugin_repo.config_values[1]
    assert stored.secret is False
    assert stored.value == "v"


def test_a_key_the_plugin_does_not_have_yet_is_added_with_the_kind_the_request_names(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """06-UI-SPEC.md's own "Add configuration key (custom/hand-added
    plugins and any plugin's extra keys)" row: a genuinely new key has no
    stored classification to defer to, so the request is the only source
    there is. WR-05's rule is about reclassifying an existing key, not
    about refusing new ones."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin, PluginConfigValue
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    plugin = Plugin(
        id=1, slug="example", display_name="Example", transport="stdio", args=("-m", "example_module"),
        url=None, enabled=False, builtin=False, enforces_policy=False, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(
        plugins=[plugin],
        config_values={
            1: [PluginConfigValue(key="PLAIN_KEY", secret=False, value="v", ciphertext=None, key_version=None)]
        },
    )
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.put(
        "/api/plugins/1/config",
        json={
            "values": {
                "PLAIN_KEY": {"value": "v", "secret": False},
                "NEW_SECRET": {"value": "a-plainly-fictional-new-value", "secret": True},
            }
        },
    )
    assert response.status_code == 200, response.text
    by_key = {value.key: value for value in plugin_repo.config_values[1]}
    assert by_key["NEW_SECRET"].secret is True
    assert by_key["NEW_SECRET"].ciphertext is not None
    assert by_key["PLAIN_KEY"].value == "v"


def test_installing_a_url_plugin_with_two_unnamed_secrets_is_refused_by_name(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """WR-07 (code review): the remote transport sends exactly one bearer
    credential, and the install path accepted any number of secret keys --
    leaving which one this house sends to a third-party server decided by
    the order a `SELECT` with no `ORDER BY` returned rows."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    response = client.post(
        "/api/plugins",
        json={
            "display_name": "Remote Thing",
            "transport": "url",
            "url": "https://remote.invalid/mcp",
            "config_values": {
                "API_KEY": {"value": "a-plainly-fictional-first-value", "secret": True},
                "OTHER_KEY": {"value": "a-plainly-fictional-second-value", "secret": True},
            },
        },
    )
    assert response.status_code == 400, response.text
    assert "AUTH_TOKEN" in response.json()["detail"]
    assert plugin_repo.plugins == {}, "nothing is installed when the write is refused"

    named = client.post(
        "/api/plugins",
        json={
            "display_name": "Remote Thing",
            "transport": "url",
            "url": "https://remote.invalid/mcp",
            "config_values": {
                "AUTH_TOKEN": {"value": "a-plainly-fictional-first-value", "secret": True},
                "OTHER_KEY": {"value": "a-plainly-fictional-second-value", "secret": True},
            },
        },
    )
    assert named.status_code == 201, named.text


def test_a_reserved_configuration_key_is_refused_at_install_and_at_save(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """WR-08 (code review): `PYTHONPATH` decides where a plugin child
    imports `spire_mcp` -- `spire_mcp.safety` included -- and
    `SPIRE_SAFETY` carries the house policy the enforcing child applies to
    itself. Neither is a plugin's to set, so neither write boundary
    accepts one; `_env_from_config_values` writes both after the
    configuration loop as the backstop."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()

    from spire_voice.db.repository import Plugin
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc)
    ha = Plugin(
        id=1, slug="ha", display_name="Home Assistant", transport="stdio", args=("-m", "spire_mcp.ha"),
        url=None, enabled=False, builtin=True, enforces_policy=True, timeout_ms=5000,
        created_at=now, updated_at=now, created_by_user_id=None,
    )
    plugin_repo = fake_plugin_repository(plugins=[ha])
    manager = _FakePluginManagerForRoutes()

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    saved = client.put(
        "/api/plugins/1/config",
        json={"values": {"PYTHONPATH": {"value": "/tmp/a-directory-an-admin-can-write"}}},
    )
    assert saved.status_code == 400, saved.text
    assert "PYTHONPATH" in saved.json()["detail"]
    assert plugin_repo.config_values.get(1, []) == []

    installed = client.post(
        "/api/plugins",
        json={
            "display_name": "Sneaky",
            "transport": "command",
            "command": "-m example_module",
            "config_values": {"SPIRE_SAFETY": {"value": '{"mode": "allow_all"}'}},
        },
    )
    assert installed.status_code == 400, installed.text
    assert "SPIRE_SAFETY" in installed.json()["detail"]
    assert list(plugin_repo.plugins) == [1], "nothing is installed when the write is refused"


def test_an_unreadable_catalog_is_a_named_refusal_not_a_bare_500(
    monkeypatch, fake_account_repository, fake_plugin_repository
):
    """IN-01 (code review): `CatalogError` reached the client unhandled, so
    an image whose `config/` mount does not carry the catalog showed the
    plugins screen a raw 500 instead of one of this module's own named
    refusals."""
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)
    security = SecurityConfig()
    account_repo = fake_account_repository()
    plugin_repo = fake_plugin_repository()
    manager = _FakePluginManagerForRoutes()

    from spire_voice.routes import plugins as plugins_module

    monkeypatch.setattr(
        plugins_module, "DEFAULT_CATALOG_PATH", "/nonexistent/plugin-catalog.json"
    )

    app = _build_plugins_app(security, account_repo, plugin_repo, manager)
    client = _admin_client(app, security, account_repo)

    listed = client.get("/api/plugins/catalog")
    assert listed.status_code == 503, listed.text
    assert "/nonexistent/plugin-catalog.json" in listed.json()["detail"]

    installed = client.post("/api/plugins", json={"catalog_entry": "Weather"})
    assert installed.status_code == 503, installed.text
