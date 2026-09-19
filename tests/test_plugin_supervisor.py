"""D-07 (Task 1) and D-05 (Task 2): no plugin blocks startup, Home
Assistant included, and a crashed plugin's tools come back on its own
once it does.

`PluginManager.start_all()` starts every enabled plugin concurrently,
each in its own persistent lifecycle task racing the same configured
deadline (`config.plugins.startup_deadline_s`) independently -- a plugin
whose start raises, or whose start is still running when the deadline
passes, is recorded `PluginState.DEGRADED` with its own failure text kept
verbatim, and every other plugin (Home Assistant included, no exception)
still gets its own full, independent budget. See `PluginManager.
start_all`'s own docstring for why every plugin needs exactly one
persistent task for its whole life, not a short-lived one per start
attempt -- a real, verified `anyio`/`mcp` SDK constraint, not a style
choice.

Once running, that same task also runs the ping watchdog and
crash-driven respawn (D-05, Task 2): a ping failure withdraws the
plugin's tools immediately, and a bounded-backoff respawn loop restores
them once one succeeds.

Every test here uses fakes over `spire_voice.plugins.manager.
start_plugin_host` -- proving the manager's own deadline/state-recording/
watchdog logic needs no real subprocess. `tests/test_plugin_manager.py`
already covers the real-child spawn path this file does not repeat.

Both tasks' own `<verify>` blocks run this whole file, and Task 2's own
verify additionally asserts this file contains no wall-clock sleep call
of its own: every test below proves timing behavior through event-driven
synchronization (an `asyncio.Event`-backed `_StepGate`, standing in for
`PluginManager`'s own injected `sleep`, or two coroutines directly
waiting on an `asyncio.Event`) or a real, short
`PluginsConfig.startup_deadline_s` -- never a call to the standard
library's own real-time-delay primitive written directly in this file.
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
    schema assertion can tell which plugins actually reached it. `ping()`
    is controllable (`ping_should_fail`) for Task 2's watchdog tests --
    always succeeds by default, so Task 1's own tests (which never touch
    it) are unaffected."""

    def __init__(self, tool_name: str) -> None:
        from mcp.types import Tool

        self.tools = [Tool(name=tool_name, description="", inputSchema={"type": "object", "properties": {}})]
        self.aclose_count = 0
        self.ping_calls = 0
        self.ping_should_fail = False
        # CR-02 (code review): every safety block this host was actually
        # respawned with, in order -- the tests below assert on what
        # reached the child, not merely on what a caller asked for.
        self.respawn_blocks: "list[dict | None]" = []

    async def call_tool(self, name: str, arguments: dict) -> object:
        raise AssertionError("not called in these tests")

    async def ping(self) -> None:
        self.ping_calls += 1
        if self.ping_should_fail:
            raise RuntimeError("simulated: plugin ping failed -- child appears dead")

    async def respawn(self, safety_block: "dict | None") -> None:
        self.respawn_blocks.append(safety_block)

    async def aclose(self) -> None:
        self.aclose_count += 1


class _StepGate:
    """Stands in for `PluginManager`'s injected `sleep` -- each call
    blocks until the test calls `release()` exactly once, so a test
    drives the watchdog loop's own timing deterministically, with no real
    time passing and no standard-library timed-delay call anywhere in
    this file. `wait_until_entered()` (an `asyncio.Event`, never a sleep)
    lets a test know the watchdog task has actually reached this specific
    point before changing anything the next step depends on -- and
    consumes that signal itself (clears it synchronously, the instant it
    returns) so a second call in a row genuinely waits for the *next*
    entry rather than observing the same still-set flag the watchdog task
    has not yet been scheduled again to clear on its own.
    """

    def __init__(self) -> None:
        self._entered = asyncio.Event()
        self._release = asyncio.Event()

    async def __call__(self, _seconds: float) -> None:
        self._entered.set()
        await self._release.wait()
        self._release.clear()

    async def wait_until_entered(self) -> None:
        await self._entered.wait()
        self._entered.clear()

    def release(self) -> None:
        self._release.set()


def _manager(
    repo,
    *,
    plugins_config: PluginsConfig | None = None,
    sleep=None,
) -> PluginManager:
    kwargs: dict = {}
    if sleep is not None:
        kwargs["sleep"] = sleep
    return PluginManager(
        repo,
        mcp_root=_MCP_ROOT,
        security=SecurityConfig(),
        safety_block_provider=_no_policy,
        plugins_config=plugins_config,
        **kwargs,
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
    try:
        await manager.start_all()

        assert manager.state_for("broken") is PluginState.DEGRADED
        assert manager.reason_for("broken") == "simulated: broken plugin refused to start"
        assert manager.tool_host_for("broken") is None

        assert manager.state_for("weather") is PluginState.RUNNING
        tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert tool_names == {"weather_current"}
    finally:
        await manager.stop_all()


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
    try:
        await manager.start_all()

        assert manager.state_for("ha") is PluginState.DEGRADED
        assert manager.reason_for("ha") == "simulated: Home Assistant refused to start"
        assert manager.enforcing_host is None

        assert manager.state_for("weather") is PluginState.RUNNING
        assert manager.tool_host_for("weather") is not None
        tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert "weather_current" in tool_names
    finally:
        await manager.stop_all()


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
    """`start_all()` starts every enabled plugin concurrently, each in its
    own persistent lifecycle task (see `PluginManager.start_all`'s own
    docstring), racing the same deadline independently -- a plugin that
    consumes its own entire deadline must never cost the *next* plugin
    any of its own budget: the first plugin's hang and the second
    plugin's real start are fully independent."""
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

    try:
        await manager.start_all()

        assert manager.state_for("stuck") is PluginState.DEGRADED
        assert manager.state_for("fine") is PluginState.RUNNING
        assert manager.tool_host_for("fine") is not None
    finally:
        await manager.stop_all()


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


# ---------------------------------------------------------------------------
# Task 2 (D-05): the ping watchdog, crash-driven respawn, and shutdown.
# ---------------------------------------------------------------------------


async def test_an_idle_dead_plugin_is_noticed_by_the_watchdog_and_its_tools_are_withdrawn(
    fake_plugin_repository, monkeypatch
):
    """Truth 1 (D-05): a plugin's child dies while nothing is calling it
    -- the watchdog's own ping notices, and the plugin's tools are
    withdrawn from the schema before any turn could ever discover them
    missing."""
    host = _FakeHost("ha_list_entities")

    async def _start_plugin_host(plugin, **kwargs):
        return host

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    gate = _StepGate()
    ha = _plugin(1, "ha", enforces_policy=True)
    repo = fake_plugin_repository(plugins=[ha])
    manager = _manager(repo, sleep=gate)
    try:
        await manager.start_all()
        assert manager.state_for("ha") is PluginState.RUNNING

        # The watchdog is now blocked in its ping-interval sleep.
        await gate.wait_until_entered()
        host.ping_should_fail = True
        gate.release()  # wakes the sleep -- the watchdog pings and finds it dead

        # It loops back and blocks on the backoff sleep before its first
        # respawn attempt -- reaching that point proves withdrawal
        # (the state change and the schema rebuild) already happened.
        await gate.wait_until_entered()

        assert manager.state_for("ha") is PluginState.CRASHED_RETRYING
        assert manager.tool_host_for("ha") is None
        assert manager.tools_schema == []
        assert host.aclose_count == 1
        assert host.ping_calls == 1
    finally:
        await manager.stop_all()


async def test_a_dead_plugin_is_respawned_on_backoff_and_its_tools_return(
    fake_plugin_repository, monkeypatch
):
    """Truth 2 (D-05): once a ping fails, the plugin is respawned on a
    bounded backoff -- a failed respawn attempt keeps it `CRASHED_RETRYING`
    and tries again, and a successful one restores its tools to the
    schema."""
    first_host = _FakeHost("ha_list_entities")
    second_host = _FakeHost("ha_list_entities")
    attempts: list[str] = []

    async def _start_plugin_host(plugin, **kwargs):
        if not attempts:
            attempts.append("initial")
            return first_host
        if len(attempts) == 1:
            attempts.append("retry-1-fails")
            raise RuntimeError("simulated: respawn attempt 1 failed")
        attempts.append("retry-2-succeeds")
        return second_host

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    gate = _StepGate()
    ha = _plugin(1, "ha", enforces_policy=True)
    repo = fake_plugin_repository(plugins=[ha])
    manager = _manager(repo, sleep=gate)
    try:
        await manager.start_all()
        assert manager.tool_host_for("ha") is first_host

        await gate.wait_until_entered()  # ping-interval sleep
        first_host.ping_should_fail = True
        gate.release()  # ping fails -- withdrawn, backoff begins

        await gate.wait_until_entered()  # backoff sleep before retry 1
        assert manager.state_for("ha") is PluginState.CRASHED_RETRYING
        assert manager.tools_schema == []
        gate.release()  # retry 1 attempted -- fails

        await gate.wait_until_entered()  # backoff sleep before retry 2
        assert manager.state_for("ha") is PluginState.CRASHED_RETRYING, (
            "a failed respawn attempt must not flip the state back to running"
        )
        assert manager.tools_schema == []
        gate.release()  # retry 2 attempted -- succeeds

        await gate.wait_until_entered()  # ping-interval sleep, running again
        assert manager.state_for("ha") is PluginState.RUNNING
        assert manager.tool_host_for("ha") is second_host
        tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert "ha_list_entities" in tool_names
        assert attempts == ["initial", "retry-1-fails", "retry-2-succeeds"]
    finally:
        await manager.stop_all()


async def test_stop_all_cancels_every_watchdog_and_nothing_is_left_running(
    fake_plugin_repository, monkeypatch
):
    """Truth 5 (D-05): stopping the manager stops every watchdog; nothing
    is left running after shutdown."""
    host = _FakeHost("ha_list_entities")

    async def _start_plugin_host(plugin, **kwargs):
        return host

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    gate = _StepGate()
    ha = _plugin(1, "ha", enforces_policy=True)
    repo = fake_plugin_repository(plugins=[ha])
    manager = _manager(repo, sleep=gate)

    await manager.start_all()
    await gate.wait_until_entered()  # the watchdog is now blocked mid-sleep

    watchdog_task = manager._tasks[ha.id]
    await manager.stop_all()

    assert watchdog_task.done()
    assert manager._tasks == {}
    assert host.aclose_count == 1


# --- CR-02 (code review): the policy-respawn handshake is total ----------
#
# `request_respawn` stores one pending `(safety_block, future)` per plugin
# and awaits that future with no timeout, and the only code that ever
# resolved one is the top of that plugin's own watchdog loop, reached only
# while the plugin is `RUNNING`. Every sequence that ends, replaces or
# bypasses that loop therefore has to answer the request instead --
# otherwise `POST /api/policy/...` awaits forever and the operator gets
# neither the success nor the `_respawn_failed_error` this codebase's own
# "a write that cannot take live effect is a failed write" rule promises.
#
# Each test below drives one of those sequences and asserts the awaiting
# caller actually returns. `asyncio.wait_for` bounds the await so a
# regression fails this suite instead of hanging it; nothing here sleeps.


async def _pending_respawn_request(manager, safety_block):
    """Start a `request_policy_respawn` and return its task, only once the
    request is genuinely registered as pending.

    Deterministic with no sleep of its own: the event is set as the task's
    very first statement, and `request_policy_respawn`/`request_respawn`
    then run synchronously all the way to `await future` -- so by the time
    this function's own `await` resumes, `_pending_respawn` is populated.
    """
    started = asyncio.Event()

    async def _request() -> None:
        started.set()
        await manager.request_policy_respawn(safety_block)

    task = asyncio.create_task(_request())
    await started.wait()
    assert manager._pending_respawn, "the request never registered as pending"
    return task


async def _running_manager_with_one_enforcing_plugin(fake_plugin_repository, monkeypatch):
    """One running, policy-enforcing plugin whose watchdog is parked in its
    ping-interval sleep -- the exact state a policy write finds."""
    host = _FakeHost("ha_list_entities")

    async def _start_plugin_host(plugin, **kwargs):
        return host

    monkeypatch.setattr(manager_module, "start_plugin_host", _start_plugin_host)

    gate = _StepGate()
    ha = _plugin(1, "ha", enforces_policy=True)
    repo = fake_plugin_repository(plugins=[ha])
    manager = _manager(repo, sleep=gate)
    await manager.start_all()
    await gate.wait_until_entered()
    return manager, host, gate, ha


async def test_a_second_respawn_request_fails_the_first_rather_than_orphaning_it(
    fake_plugin_repository, monkeypatch
):
    """CR-02, path 1: the pending slot holds one request, so a second one
    overwrote the first and its caller awaited a future nobody would ever
    resolve. Two policy saves in flight -- or one double-click -- was
    enough. The displaced caller is now told its write did not land, and
    the surviving request still reaches the child."""
    manager, host, gate, _ha = await _running_manager_with_one_enforcing_plugin(
        fake_plugin_repository, monkeypatch
    )
    try:
        first = await _pending_respawn_request(manager, {"deny": ["light.example_first"]})
        second = await _pending_respawn_request(manager, {"deny": ["light.example_second"]})

        with pytest.raises(RuntimeError, match="superseded"):
            await asyncio.wait_for(first, 5.0)

        gate.release()  # the watchdog wakes, pings, and services the survivor
        await asyncio.wait_for(second, 5.0)
        assert host.respawn_blocks == [{"deny": ["light.example_second"]}]
    finally:
        await manager.stop_all()


async def test_disabling_a_plugin_fails_its_pending_respawn_rather_than_orphaning_it(
    fake_plugin_repository, monkeypatch
):
    """CR-02, path 2: a disable, a configuration save or a delete cancels
    the lifecycle task -- the only task that could ever service the pending
    request -- and used to neither pop nor reject it."""
    manager, host, _gate, ha = await _running_manager_with_one_enforcing_plugin(
        fake_plugin_repository, monkeypatch
    )
    try:
        pending = await _pending_respawn_request(manager, {"deny": ["light.example_kitchen"]})

        await manager.stop_one(ha)

        with pytest.raises(RuntimeError, match="stopped before its respawn"):
            await asyncio.wait_for(pending, 5.0)
        assert host.respawn_blocks == []
    finally:
        await manager.stop_all()


async def test_shutdown_fails_a_pending_respawn_rather_than_orphaning_it(
    fake_plugin_repository, monkeypatch
):
    """CR-02, path 3: `stop_all()` sets `_stopping`, so every watchdog
    returns without servicing the pending entry -- and `_pending_respawn`
    was the one piece of state `stop_all` never drained."""
    manager, host, _gate, _ha = await _running_manager_with_one_enforcing_plugin(
        fake_plugin_repository, monkeypatch
    )
    pending = await _pending_respawn_request(manager, {"deny": ["light.example_kitchen"]})

    await manager.stop_all()

    with pytest.raises(RuntimeError, match="shutting down"):
        await asyncio.wait_for(pending, 5.0)
    assert host.respawn_blocks == []


async def test_a_plugin_that_dies_with_a_respawn_pending_fails_that_request(
    fake_plugin_repository, monkeypatch
):
    """CR-02, path 4 (the same defect, reached through D-05's own watchdog):
    the loop only services a pending request while the plugin is `RUNNING`,
    so a child that died between the request and the next loop iteration
    left the caller awaiting a recovery that may never come. The backoff
    loop does start a new child carrying a freshly read policy block, but
    that is not this request succeeding -- the write is reported failed."""
    manager, host, gate, _ha = await _running_manager_with_one_enforcing_plugin(
        fake_plugin_repository, monkeypatch
    )
    try:
        pending = await _pending_respawn_request(manager, {"deny": ["light.example_kitchen"]})

        host.ping_should_fail = True
        gate.release()  # the watchdog wakes, pings, and finds the child dead

        with pytest.raises(RuntimeError, match="died before its respawn"):
            await asyncio.wait_for(pending, 5.0)
        assert host.respawn_blocks == []
    finally:
        await manager.stop_all()


class _FakeMcpToolHost:
    """Stands in for `McpToolHost` at exactly the surface
    `plugins/host.py::start_plugin_host` touches -- WR-02 (code review).

    `start()` blocks until this test releases it, which is what a real
    plugin whose child has spawned but whose `initialize()`/`list_tools()`
    handshake never answers does. `aclose_count` is the whole point: the
    real host has a live child (or a live HTTP client and session) by that
    point, so whether anything ever closed it is the difference between a
    failed start and an orphaned process.
    """

    def __init__(self) -> None:
        self.tools: list = []
        self.session = None
        self.started = asyncio.Event()
        self.aclose_count = 0

    async def start(self, *args: object, **kwargs: object) -> None:
        self.started.set()
        await asyncio.Event().wait()  # never returns: the deadline must win

    async def aclose(self) -> None:
        self.aclose_count += 1


async def test_a_start_that_misses_the_deadline_closes_the_host_it_already_opened(
    fake_plugin_repository, monkeypatch
):
    """WR-02 (code review): `PluginManager` races every start against its
    own startup deadline, and the cancellation lands *inside*
    `McpToolHost.start()` -- after the transport context was entered and
    the child spawned. The partially started host is a local inside
    `start_plugin_host`; it was never returned, never recorded in
    `_plugins`, and so never reached by `_plugin_lifecycle`'s `finally`,
    which closes `self._plugins[id].host` (still `None` at that point).
    The child kept running until the parent process exited, and every
    "Retry now" tap on a slow plugin added another orphan.

    Uses the real `start_plugin_host` -- the code under test -- with a
    fake `McpToolHost`, since the claim is about who closes the host, not
    about what a real child does.
    """
    from spire_voice.plugins import host as host_module

    fake_host = _FakeMcpToolHost()
    monkeypatch.setattr(host_module, "McpToolHost", lambda: fake_host)

    slow = _plugin(1, "slow")
    repo = fake_plugin_repository(plugins=[slow])
    manager = _manager(repo, plugins_config=PluginsConfig(startup_deadline_s=0.05))

    await manager.start_all()

    assert manager.state_for("slow") is PluginState.DEGRADED
    assert fake_host.started.is_set(), "the fake host's start was never even reached"
    assert fake_host.aclose_count == 1, (
        "the host that missed the startup deadline was never closed -- its child "
        "is orphaned for the life of the process (the WR-02 defect)"
    )

    await manager.stop_all()
    assert fake_host.aclose_count == 1


async def test_a_start_that_raises_after_the_child_spawned_closes_the_host(
    fake_plugin_repository, monkeypatch
):
    """The same window, reached the other way: `_spawn` succeeded and the
    session handshake then raised. Whoever built the host closes it."""
    from spire_voice.plugins import host as host_module

    class _RaisingHost(_FakeMcpToolHost):
        async def start(self, *args: object, **kwargs: object) -> None:
            self.started.set()
            raise RuntimeError("simulated: the child spawned, then the handshake failed")

    fake_host = _RaisingHost()
    monkeypatch.setattr(host_module, "McpToolHost", lambda: fake_host)

    broken = _plugin(1, "broken")
    repo = fake_plugin_repository(plugins=[broken])
    manager = _manager(repo)

    await manager.start_all()

    assert manager.state_for("broken") is PluginState.DEGRADED
    assert manager.reason_for("broken") == (
        "simulated: the child spawned, then the handshake failed"
    )
    assert fake_host.aclose_count == 1

    await manager.stop_all()
