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
from spire_voice.mcp_client import (
    McpToolHost,
    McpToolHostLookup,
    RenamedToolHostView,
    mcp_tools_to_openai_tools,
)
from spire_voice.plugins.host import start_plugin_host
from spire_voice.plugins.naming import NamingResult, PluginTool, PluginTools, rename_collisions

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

# WR-07 (code review): the configuration key a remote plugin's bearer
# credential is read from, when it declares one by name. A remote plugin
# carries no process and no environment (D-02), so its one secret is not
# an environment variable at all -- it is the `Authorization: Bearer`
# header `plugins/host.py` builds. Naming the key is what makes "which
# credential does this plugin send" a fact the row states rather than an
# artefact of the order a `SELECT` with no `ORDER BY` returned.
REMOTE_AUTH_KEY = "AUTH_TOKEN"


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
        on_rebuild: "Callable[[], None] | None" = None,
    ) -> None:
        self._repository = repository
        self._mcp_root = mcp_root
        self._security = security
        self._safety_block_provider = safety_block_provider
        # CR-01 (code review): called at the end of every `rebuild()`, and
        # only there -- the one hook `lifespan` uses to republish the live
        # view the running assistant actually reads (`app.state.
        # tool_host_lookup`/`tools_schema`/`catalog_prompt`/`tool_host`).
        # Before this existed, `lifespan` copied two of this manager's own
        # attributes onto `app.state` at boot and nothing ever reassigned
        # them, so every later rebuild -- an install, an enable, a disable,
        # a delete, a crash-driven withdrawal, a recovery -- changed what
        # this manager believed and nothing at all about what the model was
        # offered. A copy that must be kept in sync was the defect; one
        # callback fired by the single writer (`rebuild`) is what keeps a
        # second copy from drifting again. `None` for every caller that has
        # no second view to publish (every test that drives this manager
        # directly).
        self._on_rebuild = on_rebuild
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
        # Plan 06-04 (D-09): the naming-pre-pass view of every currently
        # running plugin's host, recomputed by `rebuild()` below --
        # `[]` before the first `rebuild()` call, exactly matching
        # `tool_host_lookup`/`tools_schema`'s own "nothing running yet"
        # starting point.
        self._hosts_view: list[Any] = []
        # Plan 06-04, Task 3 (D-10): one line per prefixed tool naming the
        # tool and its owner, for the cacheable system prompt -- rebuilt
        # on the exact same swap as `tool_host_lookup`/`tools_schema`
        # below, since a prompt describing a plugin set the schema does
        # not agree with is how a turn comes to misattribute a tool call.
        # `""` here and after every rebuild with no collision, so a
        # deployment with no colliding plugins gets a prompt
        # byte-identical to one with no ownership block at all.
        self.tool_ownership_prompt: str = ""
        # Plan 06-05 (D-12): the naming pre-pass's own answer to "how many
        # plugins currently publish this bare tool name", kept across
        # `rebuild()` calls so `owners_of_bare_name` below always answers
        # against the *current* plugin set -- never a second computation of
        # the fact `plugins/naming.py` already owns (06-04-SUMMARY.md's own
        # Next Phase Readiness note names this exact read as what plan
        # 06-05 needs). `rename_collisions([])` here matches every other
        # "nothing running yet" field above: every bare name has zero
        # owners before the first `rebuild()` call, which is the correct,
        # unambiguous answer, not a `None` a caller would need to guard.
        self._naming_result: NamingResult = rename_collisions([])

    @property
    def hosts(self) -> list[Any]:
        """Every currently-running plugin's own host, in start order,
        already passed through `plugins.naming`'s collision-only prefixing
        pre-pass (D-09) -- the list a caller (`lifespan`) combines with a
        non-plugin host (the in-process workflow tool host) into one final
        `McpToolHostLookup`, since this manager only knows about plugin
        rows. A disabled or degraded plugin contributes no host here --
        exactly the tool-withdrawal PLUG-05 asks for.

        Each entry is a `RenamedToolHostView` (`mcp_client.py`), not the
        bare `McpToolHost` a caller predating plan 06-04 might expect --
        duck-type compatible with the one shape `McpToolHostLookup` (and
        any caller combining this list with another host) actually needs:
        `.tools` and an `async def call_tool(name, arguments)`. Recomputed
        by `rebuild()` below; `[]` before the first call to it.
        """
        return self._hosts_view

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

    def owners_of_bare_name(self, bare_name: str) -> "tuple[str, ...]":
        """Every plugin slug that currently publishes `bare_name`, per the
        naming pre-pass's own answer (`plugins/naming.py::NamingResult.
        owners_of_bare_name`, plan 06-04) -- delegated directly, never
        recomputed here. A length of 2 or more is exactly what makes a
        stored macro action's or workflow step's bare tool name ambiguous
        (D-12, plan 06-05): `routes/conflict.py`'s shared annotator and
        `turn/macros.py`/`workflow/steps.py`'s fire-time refusals both read
        this, so the fact is computed in exactly one place regardless of
        which of those three callers asks."""
        return self._naming_result.owners_of_bare_name(bare_name)

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
        # CR-02 (code review): the slot holds exactly one request, so a
        # second one displaces the first. The displaced caller is told so,
        # rather than left awaiting a future nobody will ever resolve --
        # two policy saves in flight, or one double-click, was enough to
        # hang a request forever.
        self._fail_pending_respawn(
            plugin_id,
            "superseded by a later respawn request for the same plugin -- "
            "this write did not reach the enforcing process",
        )
        future: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._pending_respawn[plugin_id] = (safety_block, future)
        await future

    def _fail_pending_respawn(self, plugin_id: int, reason: str) -> None:
        """Resolve `plugin_id`'s pending respawn request, if it has one, as
        a failure -- CR-02 (code review).

        `request_respawn` above awaits its future with no timeout, and the
        only code that ever resolved one was the top of that plugin's own
        watchdog loop, reached only while the plugin is `RUNNING`. Every
        path that ends, replaces, or bypasses that loop therefore has to
        come through here: a second request displacing the first, a
        disable/config-save/delete cancelling the lifecycle task, the
        watchdog finding the child dead, and `stop_all()`. A respawn that
        will not happen is a failed write (`routes/policy.py`'s own rule);
        it is never a request left hanging with neither answer.
        """
        pending = self._pending_respawn.pop(plugin_id, None)
        if pending is not None and not pending[1].done():
            pending[1].set_exception(RuntimeError(reason))

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

    async def start_one(self, plugin: Plugin) -> None:
        """Start (or fully restart) `plugin`'s own lifecycle from scratch
        (plan 06-06, Task 3, D-15) -- the one primitive `routes/plugins.py`
        calls for an install, an enable, and a configuration edit alike,
        since all three need `plugin`'s environment or connection rebuilt
        fresh from its current row and its current configuration values,
        never a policy-block swap on an existing host (`request_respawn`'s
        own, narrower job).

        If `plugin.id` already has a running lifecycle task, it is
        cancelled and awaited first -- exactly `stop_all()`'s own
        cross-task cancellation shape, safe for the identical reason that
        method already is: cancelling a task and creating a new one never
        touches an `anyio` cancel scope from outside the task that entered
        it, unlike calling `host.respawn()`/`host.aclose()` directly ever
        would (`request_policy_respawn`'s own docstring). Whatever host
        that task held is closed inside that task's own `finally`
        (`_plugin_lifecycle`), before this method's own fresh task is
        ever created -- so a caller never observes two lifecycle tasks
        racing over the same plugin id.

        Awaits until the fresh start's own outcome (running or degraded)
        is settled, then rebuilds the schema -- an install or a
        configuration edit whose plugin will not start still returns
        normally: `_plugin_lifecycle`'s own try/except already turns a
        start failure into a `DEGRADED` row carrying its own reason,
        never an exception out of this method (D-07's posture, extended
        here from boot to an on-demand (re)start). The only thing this
        method itself raises is a caller error -- `plugin.enabled` must
        be `True`; a disabled row has nothing to start, and a caller
        wanting the other direction calls `stop_one` instead.

        Scoped entirely to `plugin.id` (Task 3's own "no write touches a
        plugin other than the one it named"): no other plugin's task,
        host, or `RunningPlugin` entry is read or written here, only
        `rebuild()`'s own read of `self._plugins.values()` at the very
        end -- which never replaces another plugin's own host object,
        only the views `rebuild()` always recomputes from scratch.
        """
        if not plugin.enabled:
            raise RuntimeError(
                f"plugin {plugin.slug!r} is not enabled -- start_one is for a plugin that "
                "should be running; use stop_one to take one down"
            )
        await self._stop_task_if_running(plugin.id)
        self._plugins[plugin.id] = RunningPlugin(plugin=plugin, host=None, state=PluginState.STARTING)
        ready: asyncio.Future[None] = asyncio.get_running_loop().create_future()
        self._tasks[plugin.id] = asyncio.create_task(self._plugin_lifecycle(plugin, ready))
        await ready
        self.rebuild()

    async def stop_one(self, plugin: Plugin) -> None:
        """Stop `plugin`'s own lifecycle task, if it has one, and record
        it `DISABLED` with no host (plan 06-06, Task 3, D-15, PLUG-05) --
        the live half of disabling or deleting a plugin: its tools are
        withdrawn from the schema before this method returns, the same
        guarantee `_run_watchdog`'s own crash-driven withdrawal already
        gives a plugin the ping watchdog finds dead.

        Safe to call for a plugin this manager has never started (no task
        to cancel) -- a caller disabling a plugin that failed to start at
        all, or deleting one, still ends up with a recorded `DISABLED`
        entry and a `rebuild()` that reflects it having no host, which is
        exactly what "no tools right now" needs to be true from the
        response the caller returns."""
        await self._stop_task_if_running(plugin.id)
        self._plugins[plugin.id] = RunningPlugin(plugin=plugin, host=None, state=PluginState.DISABLED)
        self.rebuild()

    async def forget(self, plugin_id: int) -> None:
        """Stop `plugin_id` and drop every trace of it -- WR-03 (code
        review), the live half of *deleting* a plugin, as distinct from
        disabling one.

        `stop_one` leaves a `DISABLED` bookkeeping entry behind, which is
        exactly right for a row that still exists and can be enabled
        again. For a deleted row it is wrong: nothing ever removed an
        entry from `self._plugins`, slugs are derived from the display
        name over the *current* rows, and `state_for`/`reason_for`/
        `tool_host_for` return the first slug match in insertion order --
        so reinstalling a plugin with the same name got the deleted row's
        `DISABLED` entry, and `/api/plugins` reported "Disabled" and "No
        tools right now" for a plugin that was genuinely running. It also
        meant `self._plugins` grew without bound across installs and
        deletes.

        Safe to call for a plugin this manager never started, and safe to
        call twice. Keying the three readers above by slug stays correct
        because this method is what keeps `self._plugins` holding live
        rows only, and `uq_plugins_slug` makes a slug unique among those.
        """
        await self._stop_task_if_running(plugin_id)
        self._plugins.pop(plugin_id, None)
        self.rebuild()

    async def _stop_task_if_running(self, plugin_id: int) -> None:
        """Cancel and await `plugin_id`'s own lifecycle task, if one
        exists -- shared by `start_one`/`stop_one` above, the identical
        cross-task cancellation `stop_all()` already performs from the
        lifespan's own shutdown task, narrowed to one plugin. The
        cancelled task's own `finally` (`_plugin_lifecycle`) closes
        whatever host it currently holds, from within that same task.

        CR-02 (code review): the task being cancelled is the only task that
        could ever have serviced this plugin's pending respawn request, so
        that request is failed here rather than abandoned -- a policy save
        awaiting a respawn on a plugin an admin disables (or saves
        configuration for, or deletes) at the same moment used to hang
        forever."""
        task = self._tasks.pop(plugin_id, None)
        if task is not None:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task
        self._fail_pending_respawn(
            plugin_id,
            "the plugin was stopped before its respawn could be performed -- "
            "this write did not reach the enforcing process",
        )

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
                    # CR-02 (code review): this loop only services a
                    # pending respawn while the plugin is `RUNNING`, and
                    # the host that request named is now closed. The
                    # backoff loop below will start a genuinely new child
                    # carrying a freshly read policy block, but that is not
                    # this request succeeding -- the caller is told its
                    # write did not land rather than waiting out however
                    # long the recovery takes.
                    self._fail_pending_respawn(
                        plugin.id,
                        f"plugin {plugin.slug!r} died before its respawn could be "
                        "performed -- this write did not reach the enforcing process",
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
        at all.

        WR-07 (code review): "at most one" is now enforced rather than
        assumed. This used to return the *first* row with `secret=True and
        ciphertext is not None` in whatever order the repository happened
        to return them (`SELECT ... WHERE plugin_id = ?`, no `ORDER BY`),
        so a plugin with two secret keys sent an arbitrary one of them --
        possibly a different one after a restart. `REMOTE_AUTH_KEY` names
        the credential explicitly when it is present; a single unnamed
        secret still works, because that is what every install made before
        this key existed; and two or more, with none of them named, is
        refused by name so the plugin is recorded degraded with an honest
        reason rather than connecting with a credential nobody chose.
        `routes/plugins.py` refuses to write that shape in the first
        place, so this is the backstop, not the only check.

        Plan 06-06: a row declared secret with `ciphertext=None` is a
        *declared but never set* placeholder (`routes/plugins.py`'s own
        install path writes exactly this for a catalog-declared secret key
        an admin left blank, D-16) -- not a corrupted row. Treated as "no
        credential yet," the same as a plugin that declares no secret at
        all, so an admin can install a plugin before filling in its
        token and see it start (and fail on ITS OWN terms, if the remote
        side actually requires one) rather than fail here on a shape this
        module invented."""
        config_values = await self._repository.get_config_values(plugin.id)
        set_secrets = [
            value for value in config_values if value.secret and value.ciphertext is not None
        ]
        if not set_secrets:
            return None
        named = [value for value in set_secrets if value.key == REMOTE_AUTH_KEY]
        if named:
            chosen = named[0]
        elif len(set_secrets) == 1:
            chosen = set_secrets[0]
        else:
            raise RuntimeError(
                f"plugin {plugin.slug!r} has {len(set_secrets)} secret configuration values "
                f"({sorted(value.key for value in set_secrets)!r}) and none of them is "
                f"{REMOTE_AUTH_KEY!r} -- a remote plugin sends exactly one bearer credential, "
                "and this row does not say which"
            )
        if chosen.key_version is None:
            raise RuntimeError(
                f"plugin config value {chosen.key!r} has ciphertext but no key_version -- "
                "a stored secret value must always carry both"
            )
        return decrypt_credential(chosen.ciphertext, chosen.key_version, self._security)

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
        including one the ping watchdog below triggers mid-turn.

        Plan 06-04 (D-09, D-10, Pitfall 6): the collision-only prefixing
        pre-pass (`plugins.naming.rename_collisions`) runs here, before
        `McpToolHostLookup` is ever constructed -- so the lookup is
        always handed a name space it can route without guessing, and
        its own construction-time `AmbiguousToolError` stays exactly the
        invariant check `06-CONTEXT.md`'s Specific Ideas says to keep,
        not a boot hazard. Recomputed fresh from `self._plugins` on
        *every* call, from scratch -- a plugin that stops running between
        two rebuilds (a crash, a future disable) simply is not in this
        pass's input the next time, which is what lets a survivor's name
        return to bare the moment the collision that prefixed it is
        gone."""
        running = [
            (running.plugin, running.host)
            for running in self._plugins.values()
            if running.host is not None
        ]
        naming_result = rename_collisions(
            [
                PluginTools(
                    slug=plugin.slug,
                    display_name=plugin.display_name,
                    tools=tuple(
                        PluginTool(name=tool.name, description=tool.description or "")
                        for tool in host.tools
                    ),
                )
                for plugin, host in running
            ]
        )

        hosts: list[Any] = []
        schema: list[dict[str, Any]] = []
        for plugin, host in running:
            renamed_tools_for_plugin = naming_result.tools_for(plugin.slug)
            # Feed the converter the already-renamed tools (Pitfall 6):
            # `mcp_tools_to_openai_tools` decides no names of its own, so
            # every renamed `Tool` handed to it here is what actually
            # reaches the schema and, through `RenamedToolHostView`
            # below, the model.
            renamed_mcp_tools = [
                tool.model_copy(
                    update={
                        "name": renamed.offered_name,
                        "description": renamed.offered_description,
                    }
                )
                for tool, renamed in zip(host.tools, renamed_tools_for_plugin)
            ]
            bare_name_by_offered_name = {
                renamed.offered_name: renamed.bare_name for renamed in renamed_tools_for_plugin
            }
            hosts.append(RenamedToolHostView(host, renamed_mcp_tools, bare_name_by_offered_name))
            schema.extend(mcp_tools_to_openai_tools(renamed_mcp_tools))

        self._hosts_view = hosts
        self.tool_host_lookup = McpToolHostLookup(hosts)
        self.tools_schema = schema
        self.tool_ownership_prompt = naming_result.ownership_prompt
        # Plan 06-05 (D-12): kept on the exact same swap as the three
        # attributes above, for the identical reason `tool_ownership_prompt`
        # already is -- `owners_of_bare_name` must never answer against a
        # plugin set older than what `tool_host_lookup`/`tools_schema`
        # already reflect.
        self._naming_result = naming_result
        # CR-01 (code review): last, after all five attributes above are
        # assigned, so the callback can never observe a half-swapped view.
        # A raise here propagates to whoever triggered the rebuild -- an
        # install/enable/disable route reports it as a failed write
        # (`routes/plugins.py::_reconcile_failed_error`), which is the same
        # posture `lifespan` has always taken toward a tool set it cannot
        # merge into one unambiguous lookup.
        if self._on_rebuild is not None:
            self._on_rebuild()

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
        # CR-02 (code review): `_stopping` makes every watchdog loop return
        # without servicing a pending respawn, so any request still in
        # flight has to be answered here -- `_pending_respawn` was the one
        # piece of state this method never drained, and a policy save
        # racing a shutdown was left awaiting a future nothing would ever
        # resolve.
        for plugin_id in list(self._pending_respawn):
            self._fail_pending_respawn(
                plugin_id,
                "the assistant is shutting down -- this write did not reach the "
                "enforcing process",
            )
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

    Plan 06-06: a secret value with `ciphertext=None` is a declared-but-
    never-set placeholder row (`routes/plugins.py`'s own install path),
    not a corrupted one -- it reaches the child as an empty string, the
    same as an unset plain value already does, so an admin can install a
    plugin before filling in its token and let the CHILD decide whether
    it can start with nothing there, rather than this module refusing to
    even try.
    """
    env: dict[str, str] = {"PYTHONPATH": str(mcp_root)}
    for value in config_values:
        if value.secret:
            if value.ciphertext is None:
                env[value.key] = ""
                continue
            if value.key_version is None:
                raise RuntimeError(
                    f"plugin config value {value.key!r} has ciphertext but no key_version -- "
                    "a stored secret value must always carry both"
                )
            env[value.key] = decrypt_credential(value.ciphertext, value.key_version, security)
        else:
            env[value.key] = value.value or ""
    if safety_block is not None:
        env["SPIRE_SAFETY"] = json.dumps(safety_block)
    return env
