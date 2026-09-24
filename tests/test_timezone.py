"""ATLAS tells the house's own local time, not UTC (260924-h2f, issue #1).

Four things, proved end to end as well as unit by unit:

1. With `server.timezone` unset, the boot falls back to Home Assistant's
   own `GET /api/config` `time_zone` field (`resolve_timezone`, `routes/
   wizard.py`).
2. One order decides the zone -- the webapp's own saved setting, then
   `server.timezone`, then Home Assistant, then the process's own zone --
   and a Home Assistant failure never stops the boot and never logs the
   token.
3. When nothing gives a zone and the process itself runs in UTC, the boot
   logs one `WARNING`, and `app.state.timezone_resolution.warning` carries
   the same text `GET /api/wizard/timezone` (plan Task 2) returns.
4. Every stdio MCP child -- `ha`, `weather`, and any plugin an admin
   installs -- gets `TZ` set to the resolved zone, on first start and on
   every respawn (`plugins/manager.py`).

The lifespan tests below reuse `tests/test_startup_smoke.py`'s own fake
builders (`import test_startup_smoke as smoke`, the same precedent
`tests/test_auth_setup.py` already established) -- booting the real
`lifespan` is the only way to prove `app.state.timezone_resolution`,
`app.state.server_timezone`, `WorkflowToolHost`'s own zone, and both
plugin children's `TZ` all agree, which is the actual acceptance
criterion (`_state_message` telling the house's own time, not four
independently-plausible unit tests that happen to agree by coincidence).

Only generic zones appear anywhere in this file (Europe/Berlin, Asia/
Tokyo, America/Chicago, America/New_York) -- never the operator's real
time zone (T-h2f-07, CLAUDE.md's own "no house data in git").
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import textwrap
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from zoneinfo import ZoneInfo

import httpx
import pytest
from fastapi.testclient import TestClient
from mcp.client.stdio import get_default_environment
from mcp.types import CallToolResult, TextContent

import conftest
import atlas.app as app_module
import atlas.routes.wizard as wizard_module
import test_startup_smoke as smoke
from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig, WorkflowConfig
from atlas.db.repository import PluginConfigValue
from atlas.plugins.manager import PluginManager
from atlas.routes.wizard import (
    TIMEZONE_SETTING_KEY,
    TIMEZONE_UTC_WARNING,
    TimezoneResolution,
    resolve_timezone,
)
from atlas.workflow.scheduler import WorkflowScheduler
from atlas.workflow.steps import execute_step
from atlas.workflow.tool import WorkflowToolHost
from test_plugin_env_isolation import _MCP_ROOT, _ha_plugin, _weather_plugin
from test_wizard_flow import _authed_app

_HA_URL = "http://ha.invalid:8123"
_HA_BEARER = "a-plainly-fictional-test-token-not-a-real-credential"


def _config(
    *, server_timezone: "str | None" = None, security: "SecurityConfig | None" = None
) -> SimpleNamespace:
    return SimpleNamespace(
        server=SimpleNamespace(timezone=server_timezone),
        security=security or SecurityConfig(),
    )


def _ha_plugin_repo(
    *, ha_url: str | None = _HA_URL, ha_token: str | None = _HA_BEARER
) -> conftest.FakePluginRepository:
    plugin = _ha_plugin()
    config_values: list[PluginConfigValue] = []
    if ha_url:
        config_values.append(
            PluginConfigValue(key="HA_URL", secret=False, value=ha_url, ciphertext=None, key_version=None)
        )
    if ha_token:
        config_values.append(
            PluginConfigValue(
                key="HA_TOKEN", secret=False, value=ha_token, ciphertext=None, key_version=None
            )
        )
    return conftest.FakePluginRepository(plugins=[plugin], config_values={plugin.id: config_values})


def _recording_transport(response_fn):
    """A `httpx.MockTransport` that records every request it answers --
    `requests` is the list a test asserts a call count against."""
    requests: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(request)
        return response_fn(request)

    return httpx.MockTransport(handler), requests


def _never_called_transport() -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError("Home Assistant must not be called")

    return httpx.MockTransport(handler)


# ---------------------------------------------------------------------
# resolve_timezone: the Home Assistant fallback (issue item 2)
# ---------------------------------------------------------------------


async def test_resolve_timezone_falls_back_to_home_assistant():
    def _answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config"
        assert request.headers["Authorization"] == f"Bearer {_HA_BEARER}"
        return httpx.Response(200, json={"time_zone": "Europe/Berlin"})

    transport, requests = _recording_transport(_answer)
    client = httpx.AsyncClient(transport=transport)
    try:
        resolution = await resolve_timezone(
            _config(), conftest.FakeSettingsRepository(), _ha_plugin_repo(), client=client
        )
    finally:
        await client.aclose()

    assert resolution.zone == ZoneInfo("Europe/Berlin")
    assert resolution.resolved_from == "home_assistant"
    assert resolution.name == "Europe/Berlin"
    assert resolution.child_tz == "Europe/Berlin"
    assert resolution.warning is None
    assert len(requests) == 1


# ---------------------------------------------------------------------
# Precedence
# ---------------------------------------------------------------------


async def test_a_stored_zone_wins_over_config_and_home_assistant():
    client = httpx.AsyncClient(transport=_never_called_transport())
    settings_repo = conftest.FakeSettingsRepository()
    await settings_repo.set_setting(
        TIMEZONE_SETTING_KEY,
        "Asia/Tokyo",
        updated_by_user_id=None,
        updated_at=datetime.now(timezone.utc),
    )
    try:
        resolution = await resolve_timezone(
            _config(server_timezone="America/Chicago"),
            settings_repo,
            _ha_plugin_repo(),
            client=client,
        )
    finally:
        await client.aclose()

    assert resolution.resolved_from == "database"
    assert resolution.zone == ZoneInfo("Asia/Tokyo")


async def test_config_wins_over_home_assistant_with_nothing_stored():
    client = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(server_timezone="America/Chicago"),
            conftest.FakeSettingsRepository(),
            _ha_plugin_repo(),
            client=client,
        )
    finally:
        await client.aclose()

    assert resolution.resolved_from == "config"
    assert resolution.zone == ZoneInfo("America/Chicago")


# ---------------------------------------------------------------------
# Home Assistant failures fall through, quietly about the token
# ---------------------------------------------------------------------


def _http_401(request: httpx.Request) -> httpx.Response:
    return httpx.Response(401, json={"message": "invalid auth token"})


def _connect_error(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectError("connection refused", request=request)


def _no_time_zone_key(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"something_else": "value"})


def _unknown_zone(request: httpx.Request) -> httpx.Response:
    return httpx.Response(200, json={"time_zone": "Not/AZone"})


@pytest.mark.parametrize(
    "case_id,handler",
    [
        ("http-401", _http_401),
        ("connect-error", _connect_error),
        ("no-time-zone-key", _no_time_zone_key),
        ("unknown-zone", _unknown_zone),
    ],
)
async def test_a_home_assistant_failure_falls_through_to_the_process_zone(
    case_id, handler, caplog, monkeypatch
):
    monkeypatch.delenv("TZ", raising=False)
    transport, requests = _recording_transport(handler)
    client = httpx.AsyncClient(transport=transport)
    try:
        with caplog.at_level(logging.WARNING, logger="atlas.routes.wizard"):
            resolution = await resolve_timezone(
                _config(), conftest.FakeSettingsRepository(), _ha_plugin_repo(), client=client
            )
    finally:
        await client.aclose()

    assert resolution.resolved_from == "process"
    assert resolution.zone is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert warnings, f"{case_id}: expected a WARNING"
    assert _HA_BEARER not in caplog.text


async def test_no_ha_url_configured_falls_through_with_no_request(monkeypatch):
    monkeypatch.delenv("TZ", raising=False)
    client = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(), conftest.FakeSettingsRepository(), _ha_plugin_repo(ha_url=None), client=client
        )
    finally:
        await client.aclose()

    assert resolution.resolved_from == "process"
    assert resolution.zone is None


# ---------------------------------------------------------------------
# The UTC warning (issue item 3)
# ---------------------------------------------------------------------


async def test_the_utc_warning_appears_when_the_process_zone_is_utc(monkeypatch):
    monkeypatch.setattr(wizard_module, "_process_zone_is_utc", lambda: True)
    client = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(), conftest.FakeSettingsRepository(), _ha_plugin_repo(ha_url=None), client=client
        )
    finally:
        await client.aclose()
    assert resolution.warning == TIMEZONE_UTC_WARNING


async def test_no_warning_when_the_process_zone_is_not_utc(monkeypatch):
    monkeypatch.setattr(wizard_module, "_process_zone_is_utc", lambda: False)
    client = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(), conftest.FakeSettingsRepository(), _ha_plugin_repo(ha_url=None), client=client
        )
    finally:
        await client.aclose()
    assert resolution.warning is None


# ---------------------------------------------------------------------
# child_tz on the process fallback
# ---------------------------------------------------------------------


async def test_child_tz_on_process_fallback_comes_from_the_parents_own_tz(monkeypatch):
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    client = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(), conftest.FakeSettingsRepository(), _ha_plugin_repo(ha_url=None), client=client
        )
    finally:
        await client.aclose()
    assert resolution.child_tz == "Asia/Tokyo"


async def test_child_tz_is_none_with_no_parent_tz(monkeypatch):
    monkeypatch.delenv("TZ", raising=False)
    client = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(), conftest.FakeSettingsRepository(), _ha_plugin_repo(ha_url=None), client=client
        )
    finally:
        await client.aclose()
    assert resolution.child_tz is None


# ---------------------------------------------------------------------
# PluginManager: TZ in every stdio child's own environment (issue item 4)
# ---------------------------------------------------------------------


async def _no_policy() -> "dict | None":
    return None


async def test_manager_writes_tz_to_every_plugins_child_env():
    ha = _ha_plugin()
    weather = _weather_plugin()
    repo = conftest.FakePluginRepository(
        plugins=[ha, weather],
        config_values={
            ha.id: [
                PluginConfigValue(key="HA_URL", secret=False, value=_HA_URL, ciphertext=None, key_version=None),
                # A plugin-declared TZ must be replaced, never win.
                PluginConfigValue(key="TZ", secret=False, value="Etc/GMT+5", ciphertext=None, key_version=None),
            ],
            weather.id: [],
        },
    )
    manager = PluginManager(
        repo,
        mcp_root=_MCP_ROOT,
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
        zone_name="Europe/Berlin",
    )

    ha_env = await manager._build_env(ha, safety_block=None)
    weather_env = await manager._build_env(weather, safety_block=None)

    assert ha_env["TZ"] == "Europe/Berlin"
    assert weather_env["TZ"] == "Europe/Berlin"


async def test_manager_writes_no_tz_key_with_no_resolved_zone():
    ha = _ha_plugin()
    repo = conftest.FakePluginRepository(
        plugins=[ha],
        config_values={
            ha.id: [
                PluginConfigValue(key="HA_URL", secret=False, value=_HA_URL, ciphertext=None, key_version=None),
            ],
        },
    )
    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=SecurityConfig(), safety_block_provider=_no_policy
    )

    ha_env = await manager._build_env(ha, safety_block=None)

    assert set(ha_env) == {"PYTHONPATH", "HA_URL"}


def test_a_real_child_process_sees_the_resolved_tz():
    """The exact shape a real spawned child sees: the MCP SDK's own
    default environment allow-list merged with the manager-built env
    (`test_plugin_env_isolation.py`'s own precedent), printing `TZ` and
    `time.tzname` back out."""
    manager_env = {"TZ": "Europe/Berlin"}
    env = get_default_environment() | manager_env
    code = textwrap.dedent(
        """
        import json
        import os
        import time
        print(json.dumps([os.environ.get("TZ"), list(time.tzname)]))
        """
    )
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, timeout=60
    )
    assert proc.returncode == 0, proc.stderr
    result = json.loads(proc.stdout.strip().splitlines()[-1])
    assert result == ["Europe/Berlin", ["CET", "CEST"]]


# ---------------------------------------------------------------------
# Lifespan, end to end -- the tracer check
# ---------------------------------------------------------------------


def _boot_common(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("TZ", raising=False)
    monkeypatch.setenv("ATLAS_SECRET_KEY", smoke._TEST_SECRET_KEY)
    monkeypatch.setattr(app_module, "_resolved_timezone", None)
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(smoke._write_fake_config(tmp_path)))
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)


def test_lifespan_resolves_the_zone_from_home_assistant_end_to_end(tmp_path, monkeypatch):
    _boot_common(tmp_path, monkeypatch)

    plugin_repo = conftest.FakePluginRepository(
        plugins=[smoke._ha_plugin_row(), smoke._weather_plugin_row()],
        config_values={
            1: [
                PluginConfigValue(key="HA_URL", secret=False, value=_HA_URL, ciphertext=None, key_version=None),
                PluginConfigValue(
                    key="HA_TOKEN", secret=False, value=_HA_BEARER, ciphertext=None, key_version=None
                ),
            ],
        },
    )

    def _repositories_with_ha_creds(config: object, engine: object) -> dict:
        repositories = smoke._fake_build_repositories(config, engine)
        repositories["plugin_repo"] = plugin_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _repositories_with_ha_creds)

    def _answer(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/api/config"
        return httpx.Response(200, json={"time_zone": "Europe/Berlin"})

    transport, requests = _recording_transport(_answer)
    app_module.app.state.ha_http_client = httpx.AsyncClient(transport=transport)

    recorded_envs: dict[str, dict] = {}

    async def _recording_start_plugin_host(plugin, **kwargs):
        recorded_envs[plugin.slug] = dict(kwargs.get("env") or {})
        return await smoke._fake_start_plugin_host(plugin, **kwargs)

    monkeypatch.setattr(
        smoke.plugin_manager_module, "start_plugin_host", _recording_start_plugin_host
    )

    try:
        with TestClient(app_module.app):
            resolution = app_module.app.state.timezone_resolution
            assert resolution.resolved_from == "home_assistant"
            assert app_module.app.state.server_timezone == ZoneInfo("Europe/Berlin")
            assert app_module._resolved_timezone == ZoneInfo("Europe/Berlin")
            assert app_module.app.state.workflow_tool_host._zone == ZoneInfo("Europe/Berlin")
            assert recorded_envs["ha"]["TZ"] == "Europe/Berlin"
            assert recorded_envs["weather"]["TZ"] == "Europe/Berlin"
            assert "Europe/Berlin" in app_module._state_message({})
    finally:
        del app_module.app.state.ha_http_client


def test_lifespan_logs_the_utc_warning_once_with_no_zone_anywhere(tmp_path, monkeypatch, caplog):
    _boot_common(tmp_path, monkeypatch)
    monkeypatch.setattr(smoke.plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "_build_repositories", smoke._fake_build_repositories)
    monkeypatch.setattr(wizard_module, "_process_zone_is_utc", lambda: True)

    with caplog.at_level(logging.WARNING, logger="atlas.app"):
        with TestClient(app_module.app):
            resolution = app_module.app.state.timezone_resolution
            assert resolution.warning == TIMEZONE_UTC_WARNING

    records = [
        r
        for r in caplog.records
        if r.name == "atlas.app"
        and r.levelno == logging.WARNING
        and r.getMessage() == TIMEZONE_UTC_WARNING
    ]
    assert len(records) == 1


# ---------------------------------------------------------------------
# GET/PUT /api/wizard/timezone (issue item 5, plan Task 2)
# ---------------------------------------------------------------------


def _process_fallback_resolution(*, warning: "str | None" = TIMEZONE_UTC_WARNING) -> TimezoneResolution:
    return TimezoneResolution(
        zone=None, name="the local zone", resolved_from="process", child_tz=None, warning=warning
    )


def test_get_timezone_reports_the_boot_value_and_warning(monkeypatch):
    app, client, *_ = _authed_app(monkeypatch)
    app.state.timezone_resolution = _process_fallback_resolution()

    response = client.get("/api/wizard/timezone")
    assert response.status_code == 200, response.text
    assert response.json() == {
        "zone": "the local zone",
        "resolved_from": "process",
        "stored": None,
        "warning": TIMEZONE_UTC_WARNING,
        "applies_live": False,
    }


def test_put_timezone_stores_the_zone_and_leaves_boot_values_unchanged(monkeypatch):
    app, client, security, admin, credential_repo, setup_repo, settings_repo = _authed_app(monkeypatch)
    app.state.timezone_resolution = _process_fallback_resolution()

    response = client.put("/api/wizard/timezone", json={"zone": "America/New_York"})
    assert response.status_code == 200, response.text
    body = response.json()
    assert body["stored"] == "America/New_York"
    assert body["applies_live"] is False
    assert body["zone"] == "the local zone"
    assert body["resolved_from"] == "process"
    assert body["warning"] == TIMEZONE_UTC_WARNING

    stored = settings_repo.settings[TIMEZONE_SETTING_KEY]
    assert stored.value == "America/New_York"
    assert stored.updated_by_user_id == admin["id"]


async def test_after_a_put_the_next_boot_resolves_the_stored_zone(monkeypatch):
    app, client, *_rest, settings_repo = _authed_app(monkeypatch)
    app.state.timezone_resolution = _process_fallback_resolution(warning=None)

    write = client.put("/api/wizard/timezone", json={"zone": "America/New_York"})
    assert write.status_code == 200, write.text

    client_for_ha = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(), settings_repo, _ha_plugin_repo(), client=client_for_ha
        )
    finally:
        await client_for_ha.aclose()

    assert resolution.resolved_from == "database"
    assert resolution.zone == ZoneInfo("America/New_York")


@pytest.mark.parametrize("bad_zone", ["Not/AZone", "", "../../etc/passwd", "/etc/localtime"])
def test_put_timezone_refuses_bad_zone_names(monkeypatch, bad_zone):
    app, client, *_rest, settings_repo = _authed_app(monkeypatch)
    app.state.timezone_resolution = _process_fallback_resolution()

    response = client.put("/api/wizard/timezone", json={"zone": bad_zone})
    assert response.status_code == 400, response.text
    assert settings_repo.settings == {}


def test_get_and_put_timezone_require_authentication(monkeypatch):
    app, client, *_ = _authed_app(monkeypatch)
    app.state.timezone_resolution = _process_fallback_resolution()
    anonymous = TestClient(app)

    assert anonymous.get("/api/wizard/timezone").status_code == 401
    assert anonymous.put("/api/wizard/timezone", json={"zone": "America/New_York"}).status_code == 401


async def test_put_timezone_refuses_an_operator(monkeypatch):
    app, client, security, admin, *_rest = _authed_app(monkeypatch)
    app.state.timezone_resolution = _process_fallback_resolution()

    # `current_user` re-reads the role fresh from `account_repo` by the
    # token's own user id (auth/dependencies.py's own docstring) -- a
    # token merely *claiming* role="operator" for the admin's own id
    # would still resolve to "admin", so this needs a real operator row.
    operator = await app.state.account_repo.create_user(
        email="operator@example.invalid",
        display_name="An Operator",
        password_hash="not-checked-by-this-test",
        role="operator",
    )
    operator_client = TestClient(app)
    operator_client.cookies.set(
        security.cookie_name,
        issue_access_token(user_id=operator.id, role="operator", security=security),
    )

    response = operator_client.put("/api/wizard/timezone", json={"zone": "America/New_York"})
    assert response.status_code == 403, response.text


# ---------------------------------------------------------------------
# A 22:00 local schedule fires at 22:00 local, across both DST directions
# ---------------------------------------------------------------------


class _RecordingToolHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.calls.append((name, dict(arguments)))
        return CallToolResult(
            content=[TextContent(type="text", text="ok")],
            structured_content={"changed": [arguments.get("entity_id")]},
        )


@pytest.mark.parametrize(
    "direction,clock_start,at,expected_due_at",
    [
        (
            "fall-back",
            datetime(2026, 10, 31, 16, 0, tzinfo=timezone.utc),
            "2026-11-01T22:00:00",
            datetime(2026, 11, 2, 3, 0, tzinfo=timezone.utc),
        ),
        (
            "spring-forward",
            datetime(2027, 3, 13, 17, 0, tzinfo=timezone.utc),
            "2027-03-14T22:00:00",
            datetime(2027, 3, 15, 2, 0, tzinfo=timezone.utc),
        ),
    ],
)
async def test_a_22_00_local_schedule_fires_at_22_00_local_across_dst(
    direction, clock_start, at, expected_due_at
):
    settings_repo = conftest.FakeSettingsRepository()
    await settings_repo.set_setting(
        TIMEZONE_SETTING_KEY,
        "America/New_York",
        updated_by_user_id=None,
        updated_at=datetime.now(timezone.utc),
    )
    client = httpx.AsyncClient(transport=_never_called_transport())
    try:
        resolution = await resolve_timezone(
            _config(), settings_repo, _ha_plugin_repo(), client=client
        )
    finally:
        await client.aclose()
    assert resolution.resolved_from == "database"
    zone = resolution.zone

    clock_time = [clock_start]
    workflow_repo = conftest.FakeWorkflowRepository()
    tool_host = WorkflowToolHost(workflow_repo, zone=zone, clock=lambda: clock_time[0])

    result = await tool_host.call_tool(
        "schedule_workflow",
        {
            "kind": "call_service",
            "arguments": {
                "domain": "switch",
                "service": "turn_off",
                "entity_id": "switch.example_dst_lamp",
            },
            "at": at,
            "summary": "turn off the example lamp",
        },
    )
    assert not getattr(result, "is_error", False), result

    run = await workflow_repo.get_run(1)
    assert run is not None
    due_at = run.steps[0].due_at
    assert due_at == expected_due_at, f"{direction}: due_at={due_at}"
    assert due_at.astimezone(zone).hour == 22

    recording_host = _RecordingToolHost()
    config = WorkflowConfig()
    scheduler = WorkflowScheduler(
        workflow_repo,
        lambda step, now: execute_step(step, recording_host, config, now),
        config,
        clock=lambda: clock_time[0],
    )

    clock_time[0] = due_at - timedelta(seconds=1)
    await scheduler._poll_once()
    assert scheduler.claimed_count == 0
    assert recording_host.calls == []

    clock_time[0] = due_at
    await scheduler._poll_once()
    assert scheduler.claimed_count == 1
    assert len(recording_host.calls) == 1
