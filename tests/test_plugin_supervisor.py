"""D-07: no plugin blocks startup, Home Assistant included.

`PluginManager.start_all()` races every enabled plugin's `start()` against
the same configured deadline (`config.plugins.startup_deadline_s`), one
plugin at a time (see `PluginManager.start_all`'s own docstring for the
verified `anyio`/`mcp` SDK reason literal task-based concurrency was
rejected here) -- a plugin whose start raises, or whose start is still
running when the deadline passes, is recorded `PluginState.DEGRADED` with
its own failure text kept verbatim, and every other plugin (Home Assistant
included, no exception) still gets its own full, independent budget.

Every test here uses fakes over `spire_voice.plugins.manager.
start_plugin_host` -- proving the manager's own deadline/state-recording
logic needs no real subprocess. `tests/test_plugin_manager.py` already
covers the real-child spawn path this file does not repeat.

This file also carries Task 2's watchdog/respawn tests (a following
commit) -- both tasks' own `<verify>` blocks run the whole file, and Task
2's own verify additionally asserts this file contains no wall-clock
`asyncio.sleep` of its own: every test below proves timing behavior
through event-driven synchronization (an `asyncio.Event` two coroutines
both wait on) or a real, short `PluginsConfig.startup_deadline_s`, never a
sleep call written directly in a test.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest

from spire_voice.config import PluginsConfig, SecurityConfig
from spire_voice.db.repository import Plugin
from spire_voice.plugins import manager as manager_module
from spire_voice.plugins.manager import PluginManager, PluginState

_MCP_ROOT = "/nonexistent/mcp-root"  # never read: start_plugin_host is faked in every test here


async def _no_policy() -> "dict | None":
    return None


def _plugin(
    plugin_id: int,
    slug: str,
    *,
    enabled: bool = True,
    enforces_policy: bool = False,
) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug=slug,
        display_name=slug,
        transport="stdio",
        args=("-m", f"spire_mcp.{slug}"),
        url=None,
        enabled=enabled,
        builtin=True,
        enforces_policy=enforces_policy,
        timeout_ms=5000,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )


class _FakeHost:
    """A minimal stand-in host -- distinct `.tools` per slug so a merged
    schema assertion can tell which plugins actually reached it."""

    def __init__(self, tool_name: str) -> None:
        from mcp.types import Tool

        self.tools = [Tool(name=tool_name, description="", inputSchema={"type": "object", "properties": {}})]
        self.aclose_count = 0

    async def call_tool(self, name: str, arguments: dict) -> object:
        raise AssertionError("not called in these tests")

    async def aclose(self) -> None:
        self.aclose_count += 1


def _manager(repo, *, plugins_config: PluginsConfig | None = None) -> PluginManager:
    return PluginManager(
        repo,
        mcp_root=_MCP_ROOT,
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
        plugins_config=plugins_config,
    )


async def test_a_plugin_that_raises_on_start_is_degraded_and_the_boot_continues(
    fake_plugin_repository, monkeypatch
):
    """Truth 1 (D-07): a plugin whose start raises leaves the application
    booting, recorded degraded with its own failure text kept unchanged,
    and every other plugin keeps its tools -- the rest of the assistant
    keeps answering."""
    good = _plugin(1, "weather")
    bad = _plugin(2, "broken")
    repo = fake_plugin_repository(plugins=[good, bad])

    async def _start_plugin_host(plugin, **kwargs):
        if plugin.slug == "broken":
            raise RuntimeError("simulated: broken plugin refused to start")
        return _FakeHost("weather_current")

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    manager = _manager(repo)
    await manager.start_all()

    assert manager.state_for("broken") is PluginState.DEGRADED
    assert manager.reason_for("broken") == "simulated: broken plugin refused to start"
    assert manager.tool_host_for("broken") is None

    assert manager.state_for("weather") is PluginState.RUNNING
    tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
    assert tool_names == {"weather_current"}


async def test_a_plugin_that_hangs_on_start_hits_the_deadline_and_is_degraded(
    fake_plugin_repository, monkeypatch
):
    """Truth 2 (D-07): a plugin whose start never returns is recorded
    degraded once the shared startup deadline passes, rather than holding
    the boot open forever."""
    hangs_forever = asyncio.Event()

    async def _start_plugin_host(plugin, **kwargs):
        await hangs_forever.wait()
        raise AssertionError("must never return -- the deadline should have won")

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    stuck = _plugin(1, "stuck")
    repo = fake_plugin_repository(plugins=[stuck])
    manager = _manager(repo, plugins_config=PluginsConfig(startup_deadline_s=0.05))

    await manager.start_all()

    assert manager.state_for("stuck") is PluginState.DEGRADED
    assert "did not start within" in manager.reason_for("stuck")
    assert manager.tool_host_for("stuck") is None
    assert manager.tools_schema == []


async def test_home_assistant_gets_no_special_treatment_when_it_fails_to_start(
    fake_plugin_repository, monkeypatch
):
    """D-07's own explicit reversal: the policy-enforcing plugin
    (Home Assistant) is degraded the same way any other plugin is, and the
    boot -- and the rest of the assistant -- continues without it."""
    ha = _plugin(1, "ha", enforces_policy=True)
    weather = _plugin(2, "weather")
    repo = fake_plugin_repository(plugins=[ha, weather])

    async def _start_plugin_host(plugin, **kwargs):
        if plugin.slug == "ha":
            raise RuntimeError("simulated: Home Assistant refused to start")
        return _FakeHost("weather_current")

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    manager = _manager(repo)
    await manager.start_all()

    assert manager.state_for("ha") is PluginState.DEGRADED
    assert manager.reason_for("ha") == "simulated: Home Assistant refused to start"
    assert manager.enforcing_host is None

    assert manager.state_for("weather") is PluginState.RUNNING
    assert manager.tool_host_for("weather") is not None
    tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
    assert "weather_current" in tool_names


async def test_a_disabled_plugin_is_recorded_disabled_without_being_started(
    fake_plugin_repository, monkeypatch
):
    """A disabled row is recorded `DISABLED` and never reaches
    `start_plugin_host` at all -- distinct from `DEGRADED`, which only
    ever follows an attempted, failed start."""
    started_slugs: list[str] = []

    async def _start_plugin_host(plugin, **kwargs):
        started_slugs.append(plugin.slug)
        return _FakeHost(f"{plugin.slug}_tool")

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    disabled = _plugin(1, "off", enabled=False)
    repo = fake_plugin_repository(plugins=[disabled])
    manager = _manager(repo)
    await manager.start_all()

    assert started_slugs == []
    assert manager.state_for("off") is PluginState.DISABLED
    assert manager.reason_for("off") is None
    assert manager.tool_host_for("off") is None


async def test_one_plugins_hang_never_affects_the_next_plugins_own_budget(
    fake_plugin_repository, monkeypatch
):
    """`start_all()` bounds each plugin one after another, in the same
    calling task (see `PluginManager.start_all`'s own docstring for the
    verified `anyio`/`mcp` SDK constraint that rules out literal
    task-based concurrency here) -- but a plugin that consumes its own
    entire deadline must never cost the *next* plugin any of its own
    budget: the first plugin's hang and the second plugin's real start
    are fully independent."""
    hangs_forever = asyncio.Event()

    async def _start_plugin_host(plugin, **kwargs):
        if plugin.slug == "stuck":
            await hangs_forever.wait()
            raise AssertionError("must never return -- the deadline should have won")
        return _FakeHost(f"{plugin.slug}_tool")

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    stuck = _plugin(1, "stuck")
    fine = _plugin(2, "fine")
    repo = fake_plugin_repository(plugins=[stuck, fine])
    manager = _manager(repo, plugins_config=PluginsConfig(startup_deadline_s=0.05))

    await manager.start_all()

    assert manager.state_for("stuck") is PluginState.DEGRADED
    assert manager.state_for("fine") is PluginState.RUNNING
    assert manager.tool_host_for("fine") is not None


async def test_stop_all_closes_only_running_hosts_and_never_raises_on_degraded_or_disabled(
    fake_plugin_repository, monkeypatch
):
    """`stop_all()` must tolerate a mix of running, degraded, and disabled
    rows -- only a `RUNNING` row has a host to close."""
    hosts: dict[str, _FakeHost] = {}

    async def _start_plugin_host(plugin, **kwargs):
        if plugin.slug == "broken":
            raise RuntimeError("simulated failure")
        host = _FakeHost(f"{plugin.slug}_tool")
        hosts[plugin.slug] = host
        return host

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    good = _plugin(1, "good")
    bad = _plugin(2, "broken")
    off = _plugin(3, "off", enabled=False)
    repo = fake_plugin_repository(plugins=[good, bad, off])
    manager = _manager(repo)
    await manager.start_all()

    await manager.stop_all()

    assert hosts["good"].aclose_count == 1
