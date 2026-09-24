"""`PluginManager` is the only thing in this process that spawns an MCP
plugin child (D-01, D-04, PLUG-09) -- these tests prove the vertical slice
this plan builds: a row in the `plugins` table reaches the assistant
through one loop, a disabled row spawns nothing, a secret config value is
decrypted only at the child's environment, and a respawn of the
policy-enforcing plugin carries a genuinely new safety block rather than
repeating the one the first child was started with.

Every single-plugin test spawns the real `atlas_mcp.ha` child module
already shipped in this repository's own `mcp/` directory (the same
real-subprocess discipline `tests/test_mcp_client.py` and
`tests/test_ha_tool.py` already use) -- never a fake tool host for the
Home Assistant side, since the whole point of this plan is that
`PluginManager` is the one thing that spawns a real child.

The one two-plugin test below spawns Home Assistant for real and
substitutes a fake for the second plugin: running two real MCP stdio
children (each holding its own `anyio` task-group-backed
`AsyncExitStack`) concurrently in one asyncio task hits a real, pre-existing
limitation in the installed `mcp`/`anyio` versions on this interpreter --
`RuntimeError: Attempted to exit a cancel scope that isn't the current
task's current cancel scope` at teardown -- reproducible with two bare
`McpToolHost` instances and no code this plan added. That defect predates
this plan (the same two-real-host shape already existed in `app.py`'s own
`lifespan`, Home Assistant plus weather, before this plan's refactor) and
is out of this task's scope to fix; the fake keeps this test's own claim
(the manager's loop treats every plugin identically, no per-slug branch)
provable without tripping over it.
"""

from __future__ import annotations

import asyncio
import json
import os
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest
from mcp.types import Tool

from atlas.config import SecurityConfig
from atlas.crypto.credentials import encrypt_credential
from atlas.db.repository import Plugin, PluginConfigValue
from atlas.mcp_client import McpToolHost, UnknownToolError
from atlas.plugins import manager as manager_module
from atlas.plugins.manager import PluginManager, PluginState, RunningPlugin
from atlas.plugins.naming import NAME_SEPARATOR

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_ROOT = os.path.join(_REPO_ROOT, "mcp")

# A plainly fictional secret -- `read_secret_key` has no structural
# requirement of its own (unlike `validate_secret_key_strength`, which
# this migration/manager path never calls), so a short literal is enough,
# matching `tests/test_credentials_crypto.py`'s own convention.
_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)


def _plugin(
    plugin_id: int,
    slug: str,
    *,
    args: list[str],
    enabled: bool = True,
    builtin: bool = True,
    enforces_policy: bool = False,
    timeout_ms: int = 5000,
) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug=slug,
        display_name=slug,
        transport="stdio",
        args=tuple(args),
        url=None,
        enabled=enabled,
        builtin=builtin,
        enforces_policy=enforces_policy,
        timeout_ms=timeout_ms,
        created_at=now,
        updated_at=now,
        created_by_user_id=None,
    )


def _plain_value(key: str, value: str) -> PluginConfigValue:
    return PluginConfigValue(key=key, secret=False, value=value, ciphertext=None, key_version=None)


def _secret_value(key: str, plaintext: str, security: SecurityConfig) -> PluginConfigValue:
    ciphertext, key_version = encrypt_credential(plaintext, security)
    return PluginConfigValue(key=key, secret=True, value=None, ciphertext=ciphertext, key_version=key_version)


async def _no_policy() -> "dict | None":
    return None


async def test_a_seeded_ha_row_spawns_the_child_and_its_tools_reach_the_lookup(
    fake_plugin_repository,
):
    """Truth 1: a row in the plugins table, not a hardcoded block, is what
    makes Home Assistant's tools reach the assistant."""
    security = SecurityConfig()
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"], enforces_policy=True)
    repo = fake_plugin_repository(
        plugins=[ha_plugin],
        config_values={
            1: [
                _plain_value("HA_URL", "http://ha.invalid:8123"),
                _secret_value("HA_TOKEN", "not-a-real-token", security),
            ]
        },
    )
    denylist_block = {"mode": "allow_all_except_denylist", "deny_entities": ["switch.example_denied"]}

    async def _denylist() -> "dict | None":
        return denylist_block

    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_denylist
    )
    try:
        await manager.start_all()

        assert manager.enforcing_host is not None
        tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert "ha_list_entities" in tool_names

        # A denied service call never issues a real HTTP request to the
        # fake, unreachable Home Assistant URL -- `Denied` is raised
        # inside the child before `httpx` is ever touched (`ha.py`'s own
        # rule), which is what makes this the safe way to prove the
        # lookup genuinely routes to the spawned child.
        result = await manager.tool_host_lookup.call_tool(
            "ha_call_service",
            {
                "domain": "switch",
                "service": "turn_off",
                "entity_id": "switch.example_denied",
            },
        )
        assert result.is_error
        assert "off limits" in result.content[0].text
    finally:
        await manager.stop_all()


class _FakeWeatherHost:
    """A second, distinctly-tooled host -- standing in only for the
    second real subprocess this test's own module docstring explains
    avoiding, never for Home Assistant, which this test still spawns for
    real."""

    def __init__(self) -> None:
        self.tools = [
            Tool(
                name="weather_current",
                description="Current outdoor weather.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        return SimpleNamespace(is_error=False, content=[])

    async def aclose(self) -> None:
        return None


async def test_a_second_seeded_row_reaches_the_assistant_through_the_same_loop(
    fake_plugin_repository, monkeypatch
):
    """Truth 2: the weather capability reaches the assistant the same way,
    through the same loop, with no second spawn path -- both plugins'
    tools end up in one merged schema and one merged lookup, built by the
    identical `_start_one` call regardless of which slug is being
    started. See this module's own docstring for why the weather side is
    a fake here rather than the second real subprocess."""
    security = SecurityConfig()
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"], enforces_policy=True)
    weather_plugin = _plugin(2, "weather", args=["-m", "atlas_mcp.weather"], builtin=True)
    repo = fake_plugin_repository(
        plugins=[ha_plugin, weather_plugin],
        config_values={
            1: [
                _plain_value("HA_URL", "http://ha.invalid:8123"),
                _secret_value("HA_TOKEN", "not-a-real-token", security),
            ],
            2: [
                _plain_value("WEATHER_LATITUDE", "0.0"),
                _plain_value("WEATHER_LONGITUDE", "0.0"),
            ],
        },
    )
    denylist_block = {"mode": "allow_all_except_denylist", "deny_entities": ["switch.example_denied"]}

    async def _denylist() -> "dict | None":
        return denylist_block

    real_start_plugin_host = manager_module.start_plugin_host

    async def _start_plugin_host_or_fake(plugin, **kwargs):
        # The identical call every plugin gets -- `plugin.slug` decides
        # nothing about *how* this is called, only that this particular
        # test substitutes a fake return value for one of the two, per
        # the module docstring's own explanation.
        if plugin.slug == "weather":
            return _FakeWeatherHost()
        return await real_start_plugin_host(plugin, **kwargs)

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host_or_fake)

    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_denylist
    )
    try:
        await manager.start_all()

        tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert "ha_list_entities" in tool_names
        assert "weather_current" in tool_names

        # Both hosts are reachable through the one merged lookup -- no
        # per-slug branch anywhere in this call. A denied service call
        # never issues a real HTTP request (`ha.py`'s own rule: `Denied`
        # raises before `httpx` is touched), which is what makes this the
        # safe way to prove the Home Assistant side of the lookup without
        # a real, unreachable Home Assistant URL or a real network call.
        result = await manager.tool_host_lookup.call_tool(
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_denied"},
        )
        assert result.is_error
        await manager.tool_host_lookup.call_tool("weather_current", {})

        # `app.state.tool_host`'s own definition: the running host of the
        # plugin that enforces policy, and only that one.
        assert manager.enforcing_host is manager.tool_host_for("ha")
        assert manager.tool_host_for("weather") is not manager.enforcing_host
    finally:
        await manager.stop_all()


async def test_a_disabled_plugin_row_contributes_no_tools_and_spawns_no_child(
    fake_plugin_repository,
):
    """Truth 3: a disabled plugin row contributes no tools and spawns no
    child."""
    security = SecurityConfig()
    disabled_ha = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"], enabled=False, enforces_policy=True)
    repo = fake_plugin_repository(plugins=[disabled_ha], config_values={})
    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_no_policy
    )
    try:
        await manager.start_all()

        assert manager.tools_schema == []
        assert manager.enforcing_host is None
        assert manager.tool_host_for("ha") is None
        with pytest.raises(UnknownToolError):
            await manager.tool_host_lookup.call_tool("ha_list_entities", {})
    finally:
        await manager.stop_all()


async def test_a_secret_config_value_is_decrypted_only_in_the_spawned_childs_environment(
    fake_plugin_repository, monkeypatch
):
    """Truth 4: a plugin whose config value is marked secret is stored as
    ciphertext and never as plaintext; the value that reaches the child's
    environment is the decrypted one.

    Captures the environment the manager actually spawned the child with
    (the same interception point `tests/test_mcp_client.py::
    test_respawned_child_environment_holds_exactly_four_keys_and_no_
    database_scheme` uses) -- `HA_TOKEN` must be present as the real
    decrypted plaintext there, and only there: the repository's own
    `PluginConfigValue` above never carries anything but ciphertext.
    """
    security = SecurityConfig()
    secret_plaintext = "a-plainly-fictional-decrypted-token"
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"], enforces_policy=True)
    stored_value = _secret_value("HA_TOKEN", secret_plaintext, security)
    assert stored_value.value is None, "a secret PluginConfigValue must never carry a plaintext value"
    assert stored_value.ciphertext is not None
    assert secret_plaintext.encode("utf-8") not in stored_value.ciphertext

    repo = fake_plugin_repository(
        plugins=[ha_plugin],
        config_values={1: [_plain_value("HA_URL", "http://ha.invalid:8123"), stored_value]},
    )

    captured_envs: list[dict] = []
    real_spawn = McpToolHost._spawn

    async def _capturing_spawn(self, child_module, env):
        captured_envs.append(dict(env))
        await real_spawn(self, child_module, env)

    monkeypatch.setattr(McpToolHost, "_spawn", _capturing_spawn)

    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_no_policy
    )
    try:
        await manager.start_all()
        assert len(captured_envs) == 1
        assert captured_envs[0]["HA_TOKEN"] == secret_plaintext
        assert captured_envs[0]["PYTHONPATH"] == str(_MCP_ROOT)
    finally:
        await manager.stop_all()


async def test_respawning_the_enforcing_plugin_carries_the_new_safety_block(
    fake_plugin_repository, monkeypatch
):
    """Truth 5: respawning the policy-enforcing plugin with a new safety
    block gives the replacement child that new block, not a repeat of the
    block the first child was started with -- proven against the captured
    environment `respawn()` actually built, not merely that `respawn()`
    ran without raising.

    Goes through `PluginManager.request_respawn`, not
    `McpToolHost.respawn()` directly: the host's stack is entered inside
    that plugin's own persistent lifecycle task (`PluginManager.
    start_all`'s own docstring), and the installed `mcp`/`anyio` SDK
    requires that same task to be the one that later replaces it. A small
    `timeout_ms` keeps this plugin's ping-watchdog interval short, so the
    request is serviced promptly rather than waiting out a five-second
    default.
    """
    security = SecurityConfig()
    ha_plugin = _plugin(
        1, "ha", args=["-m", "atlas_mcp.ha"], enforces_policy=True, timeout_ms=300
    )
    repo = fake_plugin_repository(
        plugins=[ha_plugin],
        config_values={
            1: [
                _plain_value("HA_URL", "http://ha.invalid:8123"),
                _secret_value("HA_TOKEN", "not-a-real-token", security),
            ]
        },
    )

    captured_envs: list[dict] = []
    real_spawn = McpToolHost._spawn

    async def _capturing_spawn(self, child_module, env):
        captured_envs.append(dict(env))
        await real_spawn(self, child_module, env)

    monkeypatch.setattr(McpToolHost, "_spawn", _capturing_spawn)

    first_block = {"mode": "allow_all_except_denylist", "deny_entities": ["switch.example_a"]}

    async def _first_block() -> "dict | None":
        return first_block

    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_first_block
    )
    try:
        await manager.start_all()
        assert len(captured_envs) == 1
        assert json.loads(captured_envs[0]["ATLAS_SAFETY"]) == first_block

        second_block = {
            "mode": "allow_all_except_denylist",
            "deny_entities": ["switch.example_b"],
        }
        await manager.request_respawn(ha_plugin.id, second_block)

        assert len(captured_envs) == 2, "respawn must spawn exactly one replacement child"
        assert json.loads(captured_envs[1]["ATLAS_SAFETY"]) == second_block
        assert captured_envs[1]["ATLAS_SAFETY"] != captured_envs[0]["ATLAS_SAFETY"], (
            "the replacement child must not repeat the first child's own safety block"
        )
        # The rest of the environment (HA_TOKEN decrypted, HA_URL, PYTHONPATH)
        # is rebuilt fresh too, from the same config values -- not merely
        # the safety block swapped into an otherwise-frozen mapping.
        assert captured_envs[1]["HA_TOKEN"] == "not-a-real-token"
        assert captured_envs[1]["HA_URL"] == "http://ha.invalid:8123"
    finally:
        await manager.stop_all()


async def test_rebuild_never_mutates_the_previous_lookup_or_schema_in_place(fake_plugin_repository):
    """D-08: `rebuild()` swaps `tool_host_lookup`/`tools_schema` for new
    objects rather than mutating the previous ones -- a caller holding a
    reference to the pre-rebuild lookup must keep seeing the old state."""
    security = SecurityConfig()
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"], enforces_policy=True)
    repo = fake_plugin_repository(
        plugins=[ha_plugin],
        config_values={
            1: [
                _plain_value("HA_URL", "http://ha.invalid:8123"),
                _secret_value("HA_TOKEN", "not-a-real-token", security),
            ]
        },
    )
    manager = PluginManager(
        repo, mcp_root=_MCP_ROOT, security=security, safety_block_provider=_no_policy
    )
    try:
        await manager.start_all()
        old_lookup = manager.tool_host_lookup
        old_schema = manager.tools_schema

        manager.rebuild()

        assert manager.tool_host_lookup is not old_lookup
        assert manager.tools_schema is not old_schema
        assert manager.tools_schema == old_schema
    finally:
        await manager.stop_all()


class _FakeCrashableHost:
    """A minimal fake host whose `ping()` can be told to fail on demand --
    standing in for a real spawned child only for this module's own D-05
    watchdog/rebuild test, which needs to control exactly when a "crash"
    is noticed without a real subprocess or any wall-clock wait."""

    def __init__(self, tool_name: str) -> None:
        self.tools = [Tool(name=tool_name, description="", inputSchema={"type": "object", "properties": {}})]
        self.ping_should_fail = False
        self.aclose_count = 0

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        return SimpleNamespace(is_error=False, content=[])

    async def ping(self) -> None:
        if self.ping_should_fail:
            raise RuntimeError("simulated: plugin ping failed -- child appears dead")

    async def aclose(self) -> None:
        self.aclose_count += 1


class _StepGate:
    """Stands in for `PluginManager`'s injected `sleep` -- each call
    blocks until the test calls `release()` exactly once, so the test
    drives the watchdog loop's own timing deterministically, with no real
    time passing and no standard-library timed-delay call anywhere in
    this test. `wait_until_entered()` (an `asyncio.Event`, never a sleep)
    lets the test know the watchdog task has actually reached this point
    before changing anything the next step depends on.
    """

    def __init__(self) -> None:
        self._entered = asyncio.Event()
        self._release = asyncio.Event()

    async def __call__(self, _seconds: float) -> None:
        self._entered.set()
        await self._release.wait()
        self._release.clear()

    async def wait_until_entered(self) -> None:
        # Clears `_entered` itself, synchronously, the instant this
        # returns -- so a second call in a row genuinely waits for the
        # *next* entry rather than observing the same still-set flag
        # from the entry this call just consumed (the watchdog task's own
        # `__call__` has not necessarily been scheduled again yet to
        # clear it itself).
        await self._entered.wait()
        self._entered.clear()

    def release(self) -> None:
        self._release.set()


async def test_a_rebuild_the_watchdog_triggers_mid_turn_leaves_that_turns_own_objects_unchanged(
    fake_plugin_repository, monkeypatch
):
    """D-08, proven against D-05's own real trigger rather than a bare
    `rebuild()` call: a turn captures `tools_schema`/`tool_host_lookup` at
    its own start (`run_turn`'s own contract, verified in 06-RESEARCH.md's
    Orchestrator Addendum) -- a plugin crash-and-respawn cycle the ping
    watchdog drives entirely on its own, mid-"turn", must never change
    what that turn was already handed. Uses a fake host (`_FakeCrashableHost`)
    and an injected `_StepGate` so the crash/respawn cycle is fully
    deterministic and needs no real subprocess or wall-clock wait.
    """
    host = _FakeCrashableHost("ha_list_entities")

    async def _start_plugin_host(plugin, **kwargs):
        return host

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    gate = _StepGate()
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"], enforces_policy=True)
    repo = fake_plugin_repository(plugins=[ha_plugin], config_values={})
    manager = PluginManager(
        repo,
        mcp_root=_MCP_ROOT,
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
        sleep=gate,
    )
    try:
        await manager.start_all()

        # "Start a turn": capture exactly what `run_turn` would be handed
        # at this instant, per `app.py`'s own call sites.
        turns_schema = manager.tools_schema
        turns_lookup = manager.tool_host_lookup
        assert {entry["function"]["name"] for entry in turns_schema} == {"ha_list_entities"}

        # The watchdog crashes and respawns this plugin entirely in the
        # background, with no turn or caller driving it.
        await gate.wait_until_entered()  # ping-interval sleep
        host.ping_should_fail = True
        gate.release()

        await gate.wait_until_entered()  # backoff sleep, plugin now CRASHED_RETRYING
        assert manager.state_for("ha") is PluginState.CRASHED_RETRYING
        gate.release()  # respawn attempt runs and succeeds (start_plugin_host above)

        await gate.wait_until_entered()  # ping-interval sleep, running again
        assert manager.state_for("ha") is PluginState.RUNNING

        # The turn's own captured objects are exactly what they were --
        # same objects, same content -- even though the manager has moved
        # on to a new schema/lookup built from the recovered plugin.
        assert manager.tool_host_lookup is not turns_lookup
        assert manager.tools_schema is not turns_schema
        assert {entry["function"]["name"] for entry in turns_schema} == {"ha_list_entities"}
        assert {entry["function"]["name"] for entry in manager.tools_schema} == {"ha_list_entities"}
    finally:
        await manager.stop_all()


# --- Plan 06-04, Task 2: the lookup is handed a name space it can route --


class _FakeToolHost:
    """A minimal fake host advertising a fixed tool list -- standing in
    for a real spawned child for this module's own collision-prefixing
    tests, which need two hosts publishing the same bare name with no
    real subprocess and no dependency on which transport started them
    (plan 06-04 prefixes tool names from `host.tools` regardless of
    transport, per `06-03-SUMMARY.md`'s own Next Phase Readiness note)."""

    def __init__(self, tool_specs: "list[tuple[str, str]]") -> None:
        self.tools = [
            Tool(name=name, description=description, inputSchema={"type": "object", "properties": {}})
            for name, description in tool_specs
        ]
        self.calls: list[str] = []

    async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
        self.calls.append(name)
        return SimpleNamespace(is_error=False, content=[], name=name)


def _running(plugin: Plugin, host) -> RunningPlugin:
    return RunningPlugin(plugin=plugin, host=host, state=PluginState.RUNNING)


def _bare_manager(fake_plugin_repository) -> PluginManager:
    """A `PluginManager` with no plugin ever actually started through
    `start_all()` -- these tests drive `rebuild()` directly over
    hand-built `RunningPlugin` entries, the same shape `_plugin_lifecycle`
    would have produced, so they can prove collision-prefixing with no
    real subprocess at all."""
    return PluginManager(
        fake_plugin_repository(plugins=[]),
        mcp_root=_MCP_ROOT,
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
    )


def test_a_second_plugin_publishing_a_name_the_first_already_published_does_not_stop_the_assistant(
    fake_plugin_repository,
):
    """T-06-18: a name collision must be a rename, never a refusal to
    start -- `rebuild()` itself must not raise, and both tools must reach
    the schema under distinct, owner-prefixed names."""
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"])
    weather_plugin = _plugin(2, "weather", args=["-m", "atlas_mcp.weather"])
    ha_host = _FakeToolHost([("notify", "Home Assistant's own notify")])
    weather_host = _FakeToolHost([("notify", "Weather's own notify")])

    manager = _bare_manager(fake_plugin_repository)
    manager._plugins[1] = _running(ha_plugin, ha_host)
    manager._plugins[2] = _running(weather_plugin, weather_host)

    manager.rebuild()  # must not raise AmbiguousToolError

    tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
    assert tool_names == {f"ha{NAME_SEPARATOR}notify", f"weather{NAME_SEPARATOR}notify"}


async def test_a_call_naming_a_prefixed_tool_reaches_the_plugin_that_owns_it_and_no_other(
    fake_plugin_repository,
):
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"])
    weather_plugin = _plugin(2, "weather", args=["-m", "atlas_mcp.weather"])
    ha_host = _FakeToolHost([("notify", "Home Assistant's own notify")])
    weather_host = _FakeToolHost([("notify", "Weather's own notify")])

    manager = _bare_manager(fake_plugin_repository)
    manager._plugins[1] = _running(ha_plugin, ha_host)
    manager._plugins[2] = _running(weather_plugin, weather_host)
    manager.rebuild()

    await manager.tool_host_lookup.call_tool(f"weather{NAME_SEPARATOR}notify", {"text": "hi"})

    # The call reached the weather host, and no other -- and the weather
    # host itself was handed its own bare name, never the prefixed one it
    # never advertised.
    assert weather_host.calls == ["notify"]
    assert ha_host.calls == []


async def test_a_call_naming_a_still_bare_uncontested_tool_reaches_its_one_owner(
    fake_plugin_repository,
):
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"])
    weather_plugin = _plugin(2, "weather", args=["-m", "atlas_mcp.weather"])
    ha_host = _FakeToolHost([("list_entities", "list Home Assistant entities")])
    weather_host = _FakeToolHost([("notify", "Weather's own notify")])

    manager = _bare_manager(fake_plugin_repository)
    manager._plugins[1] = _running(ha_plugin, ha_host)
    manager._plugins[2] = _running(weather_plugin, weather_host)
    manager.rebuild()

    await manager.tool_host_lookup.call_tool("list_entities", {})

    assert ha_host.calls == ["list_entities"]


def test_disabling_one_of_two_colliding_plugins_returns_the_survivor_to_its_bare_name(
    fake_plugin_repository,
):
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"])
    weather_plugin = _plugin(2, "weather", args=["-m", "atlas_mcp.weather"])
    ha_host = _FakeToolHost([("notify", "Home Assistant's own notify")])
    weather_host = _FakeToolHost([("notify", "Weather's own notify")])

    manager = _bare_manager(fake_plugin_repository)
    manager._plugins[1] = _running(ha_plugin, ha_host)
    manager._plugins[2] = _running(weather_plugin, weather_host)
    manager.rebuild()
    assert f"ha{NAME_SEPARATOR}notify" in {entry["function"]["name"] for entry in manager.tools_schema}

    # "Disabling" the weather plugin here means exactly what a future
    # disable route will do to this manager's own bookkeeping: it stops
    # being a `RunningPlugin` with a host at all.
    manager._plugins[2] = RunningPlugin(plugin=weather_plugin, host=None, state=PluginState.DISABLED)
    manager.rebuild()

    tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
    assert tool_names == {"notify"}


# --- Plan 06-04, Task 3: the ownership prompt rebuilds on the same swap --


def test_tool_ownership_prompt_is_empty_before_any_collision(fake_plugin_repository):
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"])
    ha_host = _FakeToolHost([("list_entities", "list entities")])

    manager = _bare_manager(fake_plugin_repository)
    manager._plugins[1] = _running(ha_plugin, ha_host)
    manager.rebuild()

    assert manager.tool_ownership_prompt == ""


def test_a_rebuild_that_adds_a_collision_updates_the_ownership_prompt_on_the_same_swap(
    fake_plugin_repository,
):
    ha_plugin = _plugin(1, "ha", args=["-m", "atlas_mcp.ha"])
    weather_plugin = _plugin(2, "weather", args=["-m", "atlas_mcp.weather"])
    ha_host = _FakeToolHost([("notify", "Home Assistant's own notify")])
    weather_host = _FakeToolHost([("notify", "Weather's own notify")])

    manager = _bare_manager(fake_plugin_repository)
    manager._plugins[1] = _running(ha_plugin, ha_host)
    manager.rebuild()
    assert manager.tool_ownership_prompt == ""

    # Installing the second, colliding plugin and rebuilding again updates
    # the schema/lookup and the ownership prompt in the exact same call --
    # never one without the other.
    manager._plugins[2] = _running(weather_plugin, weather_host)
    manager.rebuild()

    assert f"ha{NAME_SEPARATOR}notify" in {
        entry["function"]["name"] for entry in manager.tools_schema
    }
    assert f"ha{NAME_SEPARATOR}notify" in manager.tool_ownership_prompt
    assert f"weather{NAME_SEPARATOR}notify" in manager.tool_ownership_prompt
