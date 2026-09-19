"""`PluginManager`: the one thing in this process that spawns a plugin
child (D-01, D-04, PLUG-01, PLUG-09) -- `lifespan` constructs this and
nothing else that starts an MCP host.

`start_all()` reads every row and starts every enabled one concurrently,
each bounded by the same non-blocking startup deadline (D-07, plan
06-02). Every enabled plugin gets exactly one persistent task
(`_plugin_lifecycle`) that owns its host for that plugin's entire
remaining life -- the bounded initial start, the ping watchdog and
crash-driven respawn (D-05), and the final close on shutdown -- never
handed off between tasks; `_plugin_lifecycle`'s own docstring explains
why that single-task-per-plugin discipline is load-bearing, not merely
tidy. Disabled/starting/running/degraded/crashed-and-retrying are all
recorded as one of the small closed set of states in `PluginState` below
-- never a pair of loose booleans, so a state the screens branch on (a
following plan) cannot be spelled two ways.

Every plugin's environment is built literally -- the declared config
values (a secret one decrypted at this single point, D-03), `PYTHONPATH`
pointing at the repository's `mcp` directory, and the serialized safety
block for the row `enforces_policy` names -- and started through
`plugins.host.start_plugin_host` (never a second spawn path, D-04, Pitfall
2). `rebuild()` builds a new tools list and a new `McpToolHostLookup` and
assigns both -- never mutating either in place (D-08's own instruction:
`run_turn` reads `tools_schema`/`tool_host_lookup` exactly once per turn,
at the start, so an immutable swap needs no lock, generation counter, or
readers-writer gate at this level).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import os
from dataclasses import dataclass
from enum import Enum
from typing import Any, Awaitable, Callable, Sequence

from spire_voice.config import PluginsConfig, SecurityConfig
from spire_voice.crypto.credentials import decrypt_credential
from spire_voice.db.repository import Plugin, PluginConfigValue, PluginRepository
from spire_voice.mcp_client import McpToolHost, McpToolHostLookup, mcp_tools_to_openai_tools
from spire_voice.plugins.host import start_plugin_host

logger = logging.getLogger("spire_voice.plugins.manager")


class PluginState(str, Enum):
    """The small closed set a plugin's runtime state is one of (Task 1's
    own action text) -- never a pair of loose booleans a screen (a
    following plan) could branch on two different ways for the same
    underlying fact.

    `STARTING` is transient, held only while `start_all()`'s own deadline
    race for this plugin is still in flight. `RUNNING` and `DEGRADED` are
    the two boot-time outcomes (D-07). `CRASHED_RETRYING` is a plugin that
    started successfully and was later found dead by its own ping
    watchdog, currently between respawn attempts (D-05). `DISABLED` is a
    row nobody ever attempted to start.
    """

    STARTING = "starting"
    RUNNING = "running"
    DEGRADED = "degraded"
    CRASHED_RETRYING = "crashed_retrying"
    DISABLED = "disabled"


# The current house policy, as the JSON-shaped dict `Policy.from_config`
# expects -- built by `lifespan` from `safety_block_from_policy(await
# policy_repo.load_policy())`, the same one parser on both sides of the
# process boundary this codebase has always used. Read fresh on every
# call, never cached, so a policy write that lands between two plugin
# starts (or a later respawn) is never served a stale block.
SafetyBlockProvider = Callable[[], Awaitable["dict | None"]]


@dataclass
class RunningPlugin:
    """One plugin row paired with its current runtime state -- the
    manager's own bookkeeping unit, never exposed past this module except
    through the narrow `tool_host_for`/`state_for`/`reason_for` readers
    below.

    `host` is `None` for every state except `RUNNING` (and, from the next
    task onward, briefly during a respawn): a `DISABLED` row was never
    started, and a `DEGRADED`/`STARTING` row has no host yet worth
    routing a tool call to.
    """

    plugin: Plugin
    host: McpToolHost | None
    state: PluginState
    reason: str | None = None


class PluginManager:
    """Owns every plugin row's runtime state -- starting, running,
    degraded, crashed-and-retrying, disabled -- and the rebuildable tool
    schema and lookup the turn controller reads.
    """

    def __init__(
        self,
        repository: PluginRepository,
        *,
        mcp_root: "os.PathLike[str] | str",
        security: SecurityConfig,
        safety_block_provider: SafetyBlockProvider,
        plugins_config: PluginsConfig | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._repository = repository
        self._mcp_root = mcp_root
        self._security = security
        self._safety_block_provider = safety_block_provider
        # `None` (every caller that predates this task) resolves to the
        # declared defaults -- ten second startup deadline, one-to-thirty
        # second respawn backoff -- rather than making every existing
        # `PluginManager(...)` call site in this suite pass a config
        # object it has no opinion about.
        self._plugins_config = plugins_config or PluginsConfig()
        # Injectable, matching `WorkflowScheduler`'s own constructor
        # precedent (D-05, plan 06-02): every ping-watchdog interval and
        # every respawn backoff below awaits `self._sleep` rather than
        # `asyncio.sleep` directly, so a test drives the real watchdog
        # loop -- ping failure, backoff, respawn, tool withdrawal and
        # recovery -- with no wall-clock sleep of its own.
        self._sleep = sleep
        self._plugins: dict[int, RunningPlugin] = {}
        # One persistent task per enabled plugin (D-05, D-07) -- created
        # in `start_all()`, owning that plugin's host for its entire life
        # (`_plugin_lifecycle`'s own docstring explains why one task, not
        # a chain of short-lived ones). Cancelled (and awaited, so nothing
        # outlives `stop_all`) in `stop_all` below.
        self._tasks: dict[int, asyncio.Task[None]] = {}
        # A caller-requested respawn (`request_respawn`, e.g. a future
        # `routes/policy.py` wiring), serviced by that plugin's own
        # lifecycle task on its next loop iteration -- never performed
        # directly against `RunningPlugin.host` by an arbitrary caller's
        # own task, for the identical same-task reason `start_all`'s own
        # docstring gives for the initial spawn.
        self._pending_respawn: dict[int, tuple["dict | None", asyncio.Future[None]]] = {}
        # Set the instant `stop_all()` begins -- checked by every
        # lifecycle task's own watchdog loop both before and after its
        # own sleep, the same `_stopping` discipline
        # `FfmpegSupervisor._supervise` already uses, so a shutdown
        # mid-sleep never triggers one more ping or respawn.
        self._stopping = False
        # Built empty at construction, and rebuilt by `rebuild()` below --
        # never `None`, so a reader that runs before `start_all()` (there
        # is none today, but a future one should not need a `None` guard)
        # sees an unambiguous "nothing running yet" lookup instead.
        self.tool_host_lookup: McpToolHostLookup = McpToolHostLookup([])
        self.tools_schema: list[dict[str, Any]] = []

    @property
    def hosts(self) -> list[McpToolHost]:
        """Every currently-running plugin's own host, in start order --
        the raw list a caller (`lifespan`) combines with a non-plugin host
        (the in-process workflow tool host) into one final
        `McpToolHostLookup`, since this manager only knows about plugin
        rows. A disabled or degraded plugin contributes no host here --
        exactly the tool-withdrawal PLUG-05 asks for."""
        return [running.host for running in self._plugins.values() if running.host is not None]

    @property
    def enforcing_host(self) -> McpToolHost | None:
        """The running host of the plugin that enforces the house policy
        (`app.state.tool_host`'s own definition, per this plan's wiring
        note) -- `None` when that plugin is disabled, degraded, or not yet
        started. At most one plugin row has `enforces_policy=True` seeded
        by this plan's migration (Home Assistant); a future plan that lets
        an admin set this flag on more than one row would need to decide
        which wins, which this property does not attempt today."""
        for running in self._plugins.values():
            if running.plugin.enforces_policy and running.host is not None:
                return running.host
        return None

    def tool_host_for(self, slug: str) -> McpToolHost | None:
        """The running host for `slug`, or `None` if it is not running
        (disabled, degraded, or not yet started) -- a lookup by the
        plugin's own stable identifier, for callers that need one specific
        plugin's host rather than the merged `tool_host_lookup`."""
        for running in self._plugins.values():
            if running.plugin.slug == slug:
                return running.host
        return None

    def state_for(self, slug: str) -> PluginState | None:
        """`slug`'s current `PluginState`, or `None` if no plugin with
        that slug has ever been recorded (never started, never seeded) --
        the read half of the closed-set state model Task 1's own action
        text requires, for a following plan's admin screen and this same
        task's own tests."""
        for running in self._plugins.values():
            if running.plugin.slug == slug:
                return running.state
        return None

    def reason_for(self, slug: str) -> str | None:
        """`slug`'s degraded/crashed reason, kept byte for byte from the
        failure that produced it -- never paraphrased (this plan's own
        `must_haves`). `None` for a plugin that has no recorded failure
        (running, disabled, or never started)."""
        for running in self._plugins.values():
            if running.plugin.slug == slug:
                return running.reason
        return None

    async def request_respawn(self, plugin_id: int, safety_block: "dict | None") -> None:
        """Request a respawn of the currently-running plugin `plugin_id`
        with `safety_block` -- performed by that plugin's own persistent
        lifecycle task (`start_all`'s own docstring explains why a
        respawn cannot safely run in any other task), never by calling
        `McpToolHost.respawn()` directly against a host this manager
        owns. Awaits until that task has actually performed the respawn
        (raising whatever it raised) -- so a caller can still treat a
        respawn failure as a failed write, matching `routes/policy.py`'s
        existing "a write that cannot take live effect is a failed write"
        rule. Serviced on that plugin's own watchdog loop (`_run_
        watchdog`), so this can take up to one ping interval to be picked
        up. `routes/policy.py` reaches this through
        `request_policy_respawn`, which finds the enforcing row itself so
        the route never handles a plugin id.
        """
        running = self._plugins.get(plugin_id)
        if running is None or running.state is not PluginState.RUNNING:
            raise RuntimeError(f"plugin {plugin_id!r} is not currently running -- cannot respawn it")
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending_respawn[plugin_id] = (safety_block, future)
        await future

    async def request_policy_respawn(self, safety_block: "dict | None") -> None:
        """Respawn whichever running plugin enforces the house policy,
        carrying `safety_block` -- the one entry point `routes/policy.py`
        uses after a successful policy write.

        This exists so the route never has to learn a plugin id, and never
        reaches `McpToolHost.respawn()` directly. The direct call is not
        merely untidy here, it is broken: this manager gives every plugin
        one persistent lifecycle task that owns its host, and the
        installed `mcp==2.2.0` stdio transport binds its `anyio` cancel
        scope to whichever task entered it -- so a respawn driven from a
        request-handling task raises `RuntimeError: Attempted to exit
        cancel scope in a different task than it was entered in`, and
        every policy write in the webapp fails. Routing through
        `request_respawn` hands the work to the owning task, which is the
        only task allowed to do it.

        Raises `RuntimeError` when no running plugin enforces the policy
        (it is disabled, degraded, or not yet started). That is the same
        posture `routes/policy.py` already takes toward a respawn that
        cannot happen: a write whose new policy cannot reach the enforcing
        process is a failed write, never a silent success.
        """
        for plugin_id, running in self._plugins.items():
            if running.plugin.enforces_policy and running.state is PluginState.RUNNING:
                await self.request_respawn(plugin_id, safety_block)
                return
        raise RuntimeError(
            "no running plugin enforces the house policy -- cannot deliver the new "
            "policy to an enforcing process"
        )

    async def start_all(self) -> None:
        """Start every enabled plugin row concurrently, each bounded by
        the same startup deadline (D-07) -- a disabled row is recorded
        `DISABLED` and never gets a lifecycle task at all, so it spawns no
        child. Home Assistant gets no special treatment: every enabled
        row, `enforces_policy` or not, goes through the identical
        `_plugin_lifecycle` task below.

        Each plugin gets exactly one persistent task, created here, that
        owns that plugin's host for its *entire* remaining life -- the
        bounded initial start, every ping, every crash-driven respawn, and
        the final close on `stop_all()` -- never handed off. This is not
        an incidental implementation detail: the installed `mcp==2.2.0`
        SDK's stdio transport enters an `anyio` task group whose cancel
        scope is bound to whichever task entered it, and `anyio` raises
        `RuntimeError: Attempted to exit cancel scope in a different task
        than it was entered in` the moment a DIFFERENT task later tries to
        close or replace that host. A first attempt raced every plugin's
        start via `asyncio.gather`/`asyncio.wait_for` directly (no
        persistent task), and that broke this module's own
        respawn/`stop_all` contract for exactly this reason, since both
        wrap the awaited coroutine in a *new*, short-lived `asyncio.Task`
        that finishes the instant `start()` returns -- long before
        `stop_all()` (or, once D-05's watchdog respawns a crashed plugin,
        a *second* start for the same plugin) needs to touch that host
        again. One persistent task per plugin sidesteps the mismatch
        entirely: this method awaits each plugin's own `ready` future
        (resolved by that plugin's own task, not by wrapping the start
        coroutine in a second one), so plugins still start with genuine
        wall-clock overlap, but every later operation on a given plugin's
        host -- ping, respawn, close -- runs inside that same one task for
        as long as the plugin exists.
        """
        plugins = await self._repository.list_plugins()
        ready_futures: list[asyncio.Future[None]] = []
        for plugin in plugins:
            if not plugin.enabled:
                self._plugins[plugin.id] = RunningPlugin(
                    plugin=plugin, host=None, state=PluginState.DISABLED
                )
                continue
            self._plugins[plugin.id] = RunningPlugin(plugin=plugin, host=None, state=PluginState.STARTING)
            ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
            self._tasks[plugin.id] = asyncio.create_task(self._plugin_lifecycle(plugin, ready))
            ready_futures.append(ready)
        # Every `ready` here is already an `asyncio.Future` -- `gather`
        # does not wrap an existing Future in a further Task, so awaiting
        # this introduces no new task boundary of its own.
        if ready_futures:
            await asyncio.gather(*ready_futures)
        self.rebuild()

    async def _plugin_lifecycle(self, plugin: Plugin, ready: "asyncio.Future[None]") -> None:
        """Own `plugin`'s host for its entire life, in exactly one task
        (`start_all`'s own docstring explains why this must be one task,
        not a chain of short-lived ones): the bounded initial start
        (D-07), then the ping watchdog and crash-driven respawn (D-05)
        below, then a final close when `stop_all()` cancels this task.

        Resolves `ready` exactly once, as soon as the *initial* start's
        outcome (running or degraded) is known -- `start_all()` awaits
        every plugin's own `ready` to know when every plugin's boot-time
        outcome is settled; nothing after that point (a later crash, a
        later respawn) is awaited by `start_all()` at all, since those
        happen entirely in the background for the rest of this process's
        life.
        """
        host: McpToolHost | None = None
        try:
            deadline = self._plugins_config.startup_deadline_s
            try:
                async with asyncio.timeout(deadline):
                    host = await self._start_one(plugin)
            except TimeoutError:
                reason = f"did not start within {deadline:g}s"
                logger.error("plugin %r failed to start: %s", plugin.slug, reason)
                self._plugins[plugin.id] = RunningPlugin(
                    plugin=plugin, host=None, state=PluginState.DEGRADED, reason=reason
                )
                return
            except Exception as exc:  # noqa: BLE001 -- a plugin's own start failure, of any shape
                reason = str(exc)
                logger.error("plugin %r failed to start: %s", plugin.slug, reason)
                self._plugins[plugin.id] = RunningPlugin(
                    plugin=plugin, host=None, state=PluginState.DEGRADED, reason=reason
                )
                return
            else:
                self._plugins[plugin.id] = RunningPlugin(plugin=plugin, host=host, state=PluginState.RUNNING)
            finally:
                if not ready.done():
                    ready.set_result(None)

            await self._run_watchdog(plugin)
        finally:
            # Reached on every exit path -- the initial start failed, the
            # watchdog loop below returned because `stop_all()` set
            # `_stopping`, or this task was cancelled at any point in
            # between. Whichever host this plugin is *currently* holding
            # (the original one, or a later respawn's) was entered inside
            # THIS task, so this is the one place it is safe to close.
            current = self._plugins.get(plugin.id)
            if current is not None and current.host is not None:
                await current.host.aclose()

    async def _run_watchdog(self, plugin: Plugin) -> None:
        """The steady-state half of `_plugin_lifecycle`, once the initial
        start has already succeeded: ping on an interval well under
        `plugin.timeout_ms` while running (D-05); on a ping failure,
        withdraw this plugin's tools immediately (`rebuild()`) and drop
        into a bounded-backoff respawn loop until a respawn attempt
        succeeds, at which point tools are restored the same way.

        `host.ping()` (`mcp_client.py`) is what actually calls the SDK's
        own `session.send_ping()`, under the same readers-writer gate
        `respawn()` is the writer of -- this method never touches
        `host.session` directly, so a ping can never race a live
        respawn's own stack swap. `06-RESEARCH.md` Pattern 3 verified the
        installed SDK already resolves every in-flight/newly-attempted
        `call_tool` on its own when a child dies; this loop is the only
        mechanism that notices a plugin whose child died with no call in
        flight against it.

        Follows `FfmpegSupervisor._supervise`'s own shape: `self._stopping`
        checked both before and after every `self._sleep` call, so a
        shutdown mid-sleep never triggers one more ping or respawn
        attempt. `self._sleep` is injectable (constructor, matching
        `WorkflowScheduler`'s own precedent) so a test drives this whole
        loop -- ping failure, withdrawal, bounded backoff, respawn,
        recovery -- with no wall-clock sleep of its own. `stop_all()`
        additionally cancels this coroutine's own task directly, so a
        sleep or a ping in progress is interrupted rather than waited out.
        """
        backoff = self._plugins_config.respawn_backoff_min_s
        while not self._stopping:
            current = self._plugins[plugin.id]
            if current.state is PluginState.RUNNING:
                pending = self._pending_respawn.pop(plugin.id, None)
                if pending is not None:
                    safety_block, future = pending
                    try:
                        await current.host.respawn(safety_block)
                    except Exception as exc:  # noqa: BLE001 -- report to the requester, whatever it was
                        if not future.done():
                            future.set_exception(exc)
                    else:
                        if not future.done():
                            future.set_result(None)
                    continue
                await self._sleep(self._ping_interval_s(plugin))
                if self._stopping:
                    return
                current = self._plugins[plugin.id]
                assert current.host is not None
                try:
                    await current.host.ping()
                except BaseException as exc:  # noqa: BLE001 -- see below for why BaseException
                    # Plan 06-03: verified directly against the installed SDK -- a
                    # remote connection's own internal task group can surface a
                    # bare `asyncio.CancelledError` (not an `MCPError`) for a
                    # request in flight when the connection dies out from under
                    # it (its background GET-stream reconnect loop failing is
                    # what triggers this). `self._stopping` is set before
                    # `stop_all()` ever calls `task.cancel()` on this lifecycle
                    # task, so a `CancelledError` seen while it is still `False`
                    # is this plugin's connection dying, not a real shutdown --
                    # anything else (a genuine `Exception`, or a `CancelledError`
                    # seen while `self._stopping` is `True`) is handled below;
                    # a real shutdown-driven cancellation is re-raised, exactly
                    # as it would be with no `except` here at all.
                    if isinstance(exc, asyncio.CancelledError) and self._stopping:
                        raise
                    logger.warning(
                        "plugin %r ping failed (%s) -- withdrawing its tools and "
                        "scheduling a respawn",
                        plugin.slug,
                        exc,
                    )
                    await current.host.aclose()
                    self._plugins[plugin.id] = RunningPlugin(
                        plugin=plugin, host=None, state=PluginState.CRASHED_RETRYING
                    )
                    self.rebuild()
                    backoff = self._plugins_config.respawn_backoff_min_s
                continue

            # CRASHED_RETRYING: wait out the current backoff, then attempt
            # exactly one respawn -- success clears the backoff back to
            # its floor for the *next* crash; failure doubles it, bounded
            # at `respawn_backoff_max_s`, and this loop tries again.
            await self._sleep(backoff)
            if self._stopping:
                return
            try:
                new_host = await self._start_one(plugin)
            except Exception as exc:  # noqa: BLE001 -- a respawn attempt's own failure
                logger.warning(
                    "plugin %r respawn attempt failed (%s) -- retrying in %.1fs",
                    plugin.slug,
                    exc,
                    backoff,
                )
                backoff = min(backoff * 2, self._plugins_config.respawn_backoff_max_s)
                continue
            else:
                logger.info("plugin %r recovered -- tools restored", plugin.slug)
                self._plugins[plugin.id] = RunningPlugin(
                    plugin=plugin, host=new_host, state=PluginState.RUNNING
                )
                self.rebuild()
                backoff = self._plugins_config.respawn_backoff_min_s

    async def _start_one(self, plugin: Plugin) -> McpToolHost:
        safety_block = (
            await self._safety_block_provider() if plugin.enforces_policy else None
        )
        if plugin.transport == "remote":
            bearer_token = await self._remote_bearer_token(plugin)
            return await start_plugin_host(
                plugin,
                mcp_root=self._mcp_root,
                safety_block=safety_block,
                bearer_token=bearer_token,
            )
        env = await self._build_env(plugin, safety_block=safety_block)
        env_factory = self._make_env_factory(plugin) if plugin.enforces_policy else None
        return await start_plugin_host(
            plugin,
            mcp_root=self._mcp_root,
            safety_block=safety_block,
            env=env,
            env_factory=env_factory,
        )

    async def _remote_bearer_token(self, plugin: Plugin) -> "str | None":
        """`plugin`'s own secret config value, decrypted at exactly this
        point (D-03) -- building the connection, the same single point
        `_build_env` decrypts one for the stdio side -- and nowhere else;
        `start_plugin_host` never sees ciphertext, only this already-
        decrypted value. A remote plugin declares at most one secret
        config value (D-16's plain key/value shape, no OAuth flow in
        scope); `None` when it declares none, which is a real, supported
        shape and not an error -- a remote plugin can have no credential
        at all."""
        config_values = await self._repository.get_config_values(plugin.id)
        for value in config_values:
            if value.secret:
                if value.ciphertext is None or value.key_version is None:
                    raise RuntimeError(
                        f"plugin config value {value.key!r} is marked secret but has "
                        "no ciphertext -- a secret value must always be encrypted at rest"
                    )
                return decrypt_credential(value.ciphertext, value.key_version, self._security)
        return None

    async def _build_env(self, plugin: Plugin, *, safety_block: "dict | None") -> dict[str, str]:
        """The literal, key-by-key environment `plugin`'s child gets
        (SAFE-09, D-02): every declared config value (a secret one
        decrypted here, the single point D-03 names), plus `PYTHONPATH`
        pointing at this repository's own `mcp/` directory, plus the
        serialized safety block for the one row `enforces_policy` names.
        Never a filtered copy of this process's own `os.environ`.
        """
        config_values = await self._repository.get_config_values(plugin.id)
        return _env_from_config_values(
            config_values, security=self._security, mcp_root=self._mcp_root, safety_block=safety_block
        )

    def _make_env_factory(self, plugin: Plugin):
        """The `env_factory` `plugins.host.start_plugin_host` threads
        through to `McpToolHost.start()`/`respawn()` (`mcp_client.py`'s
        own hook) -- a fresh read of this plugin's own config values on
        every call, so a respawn carries whatever an admin most recently
        saved rather than the values this plugin was first started with.
        """

        async def _factory(safety_block: "dict | None") -> dict[str, str]:
            return await self._build_env(plugin, safety_block=safety_block)

        return _factory

    def rebuild(self) -> None:
        """Build a new tools list and a new lookup and assign both --
        never mutate either in place (D-08). `run_turn` reads
        `tools_schema`/`tool_host_lookup` exactly once per turn, at the
        start (the orchestrator addendum in `06-RESEARCH.md` verified
        this directly against `turn/controller.py`), so this immutable
        swap needs no lock, generation counter, or readers-writer gate of
        its own -- a turn already holding the pre-rebuild list and lookup
        keeps exactly those objects, unaffected by any later call here,
        including one the ping watchdog below triggers mid-turn."""
        hosts = self.hosts
        self.tool_host_lookup = McpToolHostLookup(hosts)
        schema: list[dict[str, Any]] = []
        for host in hosts:
            schema.extend(mcp_tools_to_openai_tools(host.tools))
        self.tools_schema = schema

    def _ping_interval_s(self, plugin: Plugin) -> float:
        """The ping watchdog's own poll interval for `plugin` -- well
        under that plugin's own `timeout_ms` call deadline (D-05's own
        action text), so a genuinely hung plugin is noticed by the
        watchdog before it could plausibly still be mid-call. A floor of
        half a second keeps a plugin configured with a very small
        `timeout_ms` from turning this into a busy loop."""
        return max(0.5, (plugin.timeout_ms / 1000.0) / 3.0)

    async def stop_all(self) -> None:
        """Cancel every plugin's own lifecycle task -- called from
        `lifespan`'s shutdown block in place of the old per-host
        `aclose()` calls. D-05's own guarantee: nothing is left running
        after shutdown.

        Each task's own `finally` (`_plugin_lifecycle`) closes whatever
        host it currently holds, from *within that same task* -- this
        method never calls `host.aclose()` directly, since a plugin that
        has been through even one watchdog-driven respawn holds a host
        whose stack was entered inside its own lifecycle task, not this
        one (see `start_all`'s own docstring). A disabled or degraded row
        has no task to cancel.
        """
        self._stopping = True
        tasks = list(self._tasks.values())
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._tasks.clear()
        self._plugins.clear()
        # Reset for a manager instance that starts again after stopping --
        # no current caller does this, but a later `start_all()`'s own
        # loop checks nothing about `_stopping`, so leaving it `True`
        # would silently make every future lifecycle task's watchdog loop
        # exit on its very first check.
        self._stopping = False


def _env_from_config_values(
    config_values: Sequence[PluginConfigValue],
    *,
    security: SecurityConfig,
    mcp_root: "os.PathLike[str] | str",
    safety_block: "dict | None",
) -> dict[str, str]:
    """The literal environment build itself, factored out of `_build_env`
    so it needs no `self` -- every declared config value, a secret one
    decrypted through `decrypt_credential` (D-03, the single point this
    codebase ever turns plugin ciphertext back into a usable value), plus
    `PYTHONPATH`, plus the serialized safety block when one is given.
    """
    env: dict[str, str] = {"PYTHONPATH": str(mcp_root)}
    for value in config_values:
        if value.secret:
            if value.ciphertext is None or value.key_version is None:
                raise RuntimeError(
                    f"plugin config value {value.key!r} is marked secret but has "
                    "no ciphertext -- a secret value must always be encrypted at rest"
                )
            env[value.key] = decrypt_credential(value.ciphertext, value.key_version, security)
        else:
            env[value.key] = value.value or ""
    if safety_block is not None:
        env["SPIRE_SAFETY"] = json.dumps(safety_block)
    return env
