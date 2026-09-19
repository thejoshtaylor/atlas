"""`PluginManager`: the one thing in this process that spawns a plugin
child (D-01, D-04, PLUG-01, PLUG-09) -- `lifespan` constructs this and
nothing else that starts an MCP host.

`start_all()` reads every row and starts every enabled one, each bounded
by the same non-blocking startup deadline (D-07, plan 06-02) -- one after
another rather than as literally overlapping `asyncio` tasks, a verified,
documented choice explained in `start_all`'s own docstring below, not an
oversight. Disabled/degraded/running are all recorded as one of the small
closed set of states in `PluginState` below -- never a pair of loose
booleans, so a state the screens branch on (a following plan) cannot be
spelled two ways.
Every plugin's environment is built literally -- the declared config
values (a secret one decrypted at this single point, D-03), `PYTHONPATH`
pointing at the repository's `mcp` directory, and the serialized safety
block for the row `enforces_policy` names -- and started through
`plugins.host.start_plugin_host` (never a second spawn path, D-04, Pitfall
2). `rebuild()` builds a new tools list and a new `McpToolHostLookup` and
assigns both -- never mutating either in place (D-08's own instruction:
`run_turn` reads `tools_schema`/`tool_host_lookup` exactly once per turn,
at the start, so an immutable swap needs no lock, generation counter, or
readers-writer gate at this level). The ping watchdog and crash-driven
respawn (D-05) land in this same module's next task; this task's own job
is the uniform non-blocking startup deadline (D-07) and the state model
that task builds on.
"""

from __future__ import annotations

import asyncio
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
    this task's two terminal-at-boot outcomes; `CRASHED_RETRYING` is the
    next task's own state, for a plugin that started successfully and was
    later found dead by the ping watchdog. `DISABLED` is a row nobody ever
    attempted to start.
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
    degraded, disabled (crashed-and-retrying from the next task onward) --
    and the rebuildable tool schema and lookup the turn controller reads.
    """

    def __init__(
        self,
        repository: PluginRepository,
        *,
        mcp_root: "os.PathLike[str] | str",
        security: SecurityConfig,
        safety_block_provider: SafetyBlockProvider,
        plugins_config: PluginsConfig | None = None,
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
        self._plugins: dict[int, RunningPlugin] = {}
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

    async def start_all(self) -> None:
        """Start every enabled plugin row, each bounded by the same
        startup deadline (D-07) -- a disabled row is recorded `DISABLED`
        and never passed to `_start_one` at all, so it spawns no child.
        Home Assistant gets no special treatment: every enabled row,
        `enforces_policy` or not, goes through the identical
        `_start_one_guarded` call below, in the same order `list_plugins`
        returned them.

        Plugins are raced against the deadline one after another, inside
        this method's own calling task, rather than as literally
        overlapping `asyncio` tasks -- a deliberate, verified choice, not
        an oversight:

        The plan's own action text says "concurrently". A first attempt
        did exactly that (`asyncio.gather`/`asyncio.wait_for` racing every
        plugin's `start()` at once) and broke this module's own,
        already-shipped respawn/`stop_all` contract
        (`tests/test_plugin_manager.py`): both `gather` and `wait_for`
        wrap the awaited coroutine in a *new* `asyncio.Task`, and the
        installed `mcp==2.2.0` SDK's stdio transport enters an `anyio`
        task group whose cancel scope is bound to whichever task entered
        it -- `anyio` raises `RuntimeError: Attempted to exit cancel scope
        in a different task than it was entered in` the moment
        `stop_all()`/`McpToolHost.respawn()` (called from THIS method's
        own caller, e.g. `lifespan`, possibly much later) tries to close
        or replace a host whose stack was entered inside a since-finished
        `gather`/`wait_for` task instead of that caller's own task.
        `asyncio.timeout()` (used below) bounds each plugin's own start
        from *within* this method's calling task -- no new task, so no
        such mismatch, and every existing single-task caller (this
        module's own tests, `lifespan`) keeps working unchanged. The one
        thing this trades away is wall-clock overlap between two slow
        starts; it does not trade away the actual guarantee D-07 asks
        for -- no plugin, Home Assistant included, can hold the boot open
        past its own configured deadline, and one plugin's failure or
        hang never affects any other plugin's own bounded attempt.
        """
        plugins = await self._repository.list_plugins()
        for plugin in plugins:
            if plugin.enabled:
                await self._start_one_guarded(plugin)
            else:
                self._plugins[plugin.id] = RunningPlugin(
                    plugin=plugin, host=None, state=PluginState.DISABLED
                )
        self.rebuild()

    async def _start_one_guarded(self, plugin: Plugin) -> None:
        """Race `_start_one(plugin)` against the configured startup
        deadline (D-07), from within the caller's own task
        (`asyncio.timeout`, not `asyncio.wait_for` -- see `start_all`'s
        own docstring for why): a raise or a timeout both end the same
        way -- `plugin` recorded `DEGRADED` with its own failure text kept
        verbatim, logged by name, and this coroutine returns normally so
        one plugin's failure never stops `start_all`'s loop from reaching
        the next plugin.
        """
        self._plugins[plugin.id] = RunningPlugin(plugin=plugin, host=None, state=PluginState.STARTING)
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
        except Exception as exc:  # noqa: BLE001 -- a plugin's own start failure, of any shape
            reason = str(exc)
            logger.error("plugin %r failed to start: %s", plugin.slug, reason)
            self._plugins[plugin.id] = RunningPlugin(
                plugin=plugin, host=None, state=PluginState.DEGRADED, reason=reason
            )
        else:
            self._plugins[plugin.id] = RunningPlugin(plugin=plugin, host=host, state=PluginState.RUNNING)

    async def _start_one(self, plugin: Plugin) -> McpToolHost:
        safety_block = (
            await self._safety_block_provider() if plugin.enforces_policy else None
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
        start, so this immutable swap needs no lock, generation counter,
        or readers-writer gate of its own."""
        hosts = self.hosts
        self.tool_host_lookup = McpToolHostLookup(hosts)
        schema: list[dict[str, Any]] = []
        for host in hosts:
            schema.extend(mcp_tools_to_openai_tools(host.tools))
        self.tools_schema = schema

    async def stop_all(self) -> None:
        """Close every running plugin child -- called from `lifespan`'s
        shutdown block in place of the old per-host `aclose()` calls. A
        disabled or degraded row has no host to close."""
        for running in self._plugins.values():
            if running.host is not None:
                await running.host.aclose()
        self._plugins.clear()


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
