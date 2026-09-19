"""The stdio MCP client wrapper: owns `spire_mcp.ha`'s subprocess for the
process lifetime, and converts its tool schemas for the language model.

Opened once, in the FastAPI lifespan, and held until shutdown -- never
per-turn. The subprocess's `env` is built explicitly, with exactly three
keys, rather than inherited from this process's environment: `HA_URL` and
`HA_TOKEN` are what the child needs, and `PYTHONPATH` is what lets
`-m spire_mcp.ha` resolve. Building it literally, not by copying and
filtering `os.environ`, is what keeps `XAI_API_KEY` out of the child process
-- the parent's provider credential has no reason to exist inside the
process that only ever talks to Home Assistant.

The child runs under `sys.executable`, never a bare `python3`. The child
imports the same `mcp` SDK this process does, so it must be the same
interpreter. A bare `python3` is whatever comes first on PATH, which outside
an activated virtualenv is the system interpreter with no SDK installed --
and because this repository has its own top-level `mcp/` directory, the
failure surfaces as a namespace-package shadow (`No module named
'mcp.server'`) rather than an honest "SDK not installed". That import runs
inside the child, at startup, so no unit test that imports the tool handlers
directly can see it.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
from contextlib import AsyncExitStack, asynccontextmanager
from typing import Any, AsyncIterator, Awaitable, Callable, Mapping, Sequence

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import Tool

# The child `start()` spawns when a caller supplies neither `child_module`
# nor `env` -- today's Home Assistant child, unchanged from before this
# plan. A second `McpToolHost` instance (the weather server, D-13) passes
# both explicitly instead of subclassing.
_DEFAULT_CHILD_MODULE = "spire_mcp.ha"

# Plan 06-01 (D-01 .. D-04): the shape a caller supplies to have
# `respawn()` rebuild a fresh environment on every call, rather than
# repeating the mapping the first child was started with. Async because
# the plugin manager's own factory reads a plugin's config values (and
# decrypts a secret one) from a repository -- both `await`-requiring
# operations `start()`/`respawn()` are already inside.
EnvFactory = Callable[["dict | None"], Awaitable[Mapping[str, str]]]


def _default_ha_env(
    ha_url: str, ha_token: str, mcp_root: str | os.PathLike[str], safety_block: dict | None
) -> dict[str, str]:
    """Today's literal three-or-four-key Home Assistant child environment --
    factored out of `_spawn` so `start()` and `respawn()` can each build it
    fresh (a respawn's whole point is a *different* `safety_block`), while
    `_spawn` itself stays the single place that actually launches a child,
    generic over which environment it was handed (Task 3, plan 04-01).
    """
    env = {"HA_URL": ha_url, "HA_TOKEN": ha_token, "PYTHONPATH": str(mcp_root)}
    if safety_block is not None:
        env["SPIRE_SAFETY"] = json.dumps(safety_block)
    return env


class McpToolHost:
    """Wraps one MCP stdio child process for its whole lifetime -- and,
    since Phase 3, can replace that child with a fresh one carrying a new
    policy (`respawn`).

    Readers-writer coordination (Task 3, plan 04-01), replacing the single
    `asyncio.Lock` this class held through Phase 3: `call_tool()` is a
    reader, `respawn()` is the writer. Two or more `call_tool()` calls now
    genuinely overlap -- each registers in `_in_flight` and snapshots
    `self.session`, then awaits the actual stdio round trip outside that
    registration step. What the old single lock bought is preserved, not
    loosened: `respawn()` still never tears the stack down while a call is
    in flight against it, and a call still never lands on a session
    `respawn()` is mid-teardown on. `_writer_waiting` is set, and
    `_readers_admitted` cleared, the instant `respawn()` begins -- before it
    ever waits for the in-flight count to reach zero -- so a steady stream
    of turns cannot starve a policy change forever; new readers block from
    that moment, not merely while the writer holds exclusive access.

    No separate lock guards `_in_flight`/`_writer_waiting`: every mutation
    of either happens in a synchronous stretch of code with no `await` in
    between check and update, and asyncio's single-threaded, cooperative
    scheduler never preempts a coroutine mid-stretch -- the same property
    that already makes a bare Python `for` loop over `await` points safe
    elsewhere in this codebase. `self._stack`/`self.session` stay the single
    source of truth for whether a child is running, the same discipline
    `FfmpegSupervisor` already follows with its own subprocess.
    """

    def __init__(self) -> None:
        self._stack = AsyncExitStack()
        self.session: ClientSession | None = None
        self.tools: list[Tool] = []
        # Readers-writer state (Task 3, plan 04-01): `_in_flight` counts
        # calls currently past `call_tool`'s gate and not yet in their
        # `finally`; `_writer_waiting` is `True` from the instant `respawn()`
        # is entered until it finishes; `_readers_admitted` is the `asyncio.Event`
        # every `call_tool()` waits on -- set whenever no writer is active or
        # waiting (the initial state), cleared by `respawn()` the same
        # instant it sets `_writer_waiting`, so no new reader can be admitted
        # from that point until `respawn()` sets it again on its way out.
        self._in_flight = 0
        self._writer_waiting = False
        self._readers_admitted = asyncio.Event()
        self._readers_admitted.set()
        # Writer-vs-writer exclusion (CR-01, phase 4 code review): the
        # readers-writer state above protects readers from a writer, but
        # nothing in it stops a SECOND concurrent `respawn()` from entering
        # the same drain-teardown-spawn sequence in parallel with the
        # first -- both would read `self._stack` before either reassigns
        # it, so one respawn's fresh stack (and the child it just spawned)
        # can be silently overwritten by the other's, leaking a still-
        # running child enforcing a stale policy. This lock restores
        # exactly the full writer-vs-writer serialization the single
        # `asyncio.Lock` this class held through Phase 3 gave for free,
        # without touching the reader-side coordination above.
        self._respawn_lock = asyncio.Lock()
        # The spawn arguments `respawn()` needs to repeat -- stored only
        # after a successful `start()`, so a `respawn()` called before any
        # `start()` fails the same way `call_tool()` does rather than
        # spawning a child with `None`s baked into its environment.
        self._ha_url: str | None = None
        self._ha_token: str | None = None
        self._mcp_root: str | os.PathLike[str] | None = None
        self._child_module: str = _DEFAULT_CHILD_MODULE
        # `None` (the default, and every caller that predates plan 06-01)
        # means `respawn()` rebuilds today's literal Home Assistant
        # environment fresh from `_ha_url`/`_ha_token`/`_mcp_root` and
        # whatever new `safety_block` it was given. A caller that passed
        # `env=` explicitly to `start()` (with no `env_factory=`) gets
        # exactly that mapping repeated, every time -- no `HA_URL`, no
        # `HA_TOKEN`, no `SPIRE_SAFETY`, nothing merged in from anywhere
        # (SAFE-09); the weather child never respawns, so this path is
        # never exercised against it, but repeats the mapping unchanged if
        # it ever is.
        self._env_override: dict[str, str] | None = None
        # Plan 06-01 (D-01 .. D-04): when a caller supplies an
        # `env_factory`, `respawn()` calls it fresh with the new
        # `safety_block` instead of repeating `_env_override` -- this is
        # what lets a plugin's respawn carry a genuinely new environment
        # (a config value an admin just changed, or a new safety block for
        # the enforcing plugin) rather than the one the first child was
        # started with. `None` (every caller that predates this plan)
        # leaves `_env_override`/the default Home Assistant build as the
        # only two paths, unchanged.
        self._env_factory: "EnvFactory | None" = None

    async def start(
        self,
        ha_url: str,
        ha_token: str,
        mcp_root: str | os.PathLike[str],
        safety_block: dict | None = None,
        *,
        child_module: str = _DEFAULT_CHILD_MODULE,
        env: "Mapping[str, str] | None" = None,
        env_factory: "EnvFactory | None" = None,
    ) -> None:
        """Spawn the tool server. `safety_block` is the raw `safety:` config.

        `child_module` and `env` both default to exactly today's Home
        Assistant child and its literal environment (Task 3, plan 04-01), so
        every caller that predates this plan compiles and behaves
        unchanged. Given explicitly, they let this same class own a
        different child with no subclass -- D-13's second `McpToolHost`
        instance for the weather server passes both, and gets none of the
        Home Assistant keys (SAFE-09): `env`, when given, is used exactly as
        given, with nothing merged in from `ha_url`/`ha_token`/`safety_block`.

        `env_factory` (plan 06-01, D-01 .. D-04) takes precedence over
        `env` when both are given: this call awaits it once, with
        `safety_block`, to build the initial environment, and `respawn()`
        awaits it again on every later call -- the hook that lets a
        plugin's environment be rebuilt fresh (a changed config value, a
        new safety block) rather than replayed from this call's own `env`.

        The child is the process that actually calls Home Assistant, so for
        the default (Home Assistant) child it is the process whose `Policy`
        decides. It cannot read the config file -- it receives an explicit
        env, not an inherited one, which is what keeps `XAI_API_KEY` out of
        it -- so the block travels as JSON on that same explicit env under
        `SPIRE_SAFETY`.

        Passing `safety_block=None` leaves the default child on `safety.py`'s
        compiled defaults: the generic destructive domains and services, and
        no entity rules. An empty house policy is a real choice an operator
        can make; a policy the operator wrote and the enforcing process never
        received is not, which is why the child refuses to start on a
        malformed block rather than quietly falling back to defaults.
        """
        self._ha_url = ha_url
        self._ha_token = ha_token
        self._mcp_root = mcp_root
        self._child_module = child_module
        self._env_override = dict(env) if env is not None else None
        self._env_factory = env_factory
        if env_factory is not None:
            resolved_env = dict(await env_factory(safety_block))
        elif env is not None:
            resolved_env = dict(env)
        else:
            resolved_env = _default_ha_env(ha_url, ha_token, mcp_root, safety_block)
        await self._spawn(child_module, resolved_env)

    async def respawn(self, safety_block: dict | None) -> None:
        """Replace the running child with a fresh one carrying `safety_block`.

        A policy change reaches the enforcing process this way -- by
        replacing it -- and never by mutating a running child's policy in
        place; there is no in-place update path here to get wrong. Repeats
        whatever `child_module` and environment `start()` recorded: the
        default (Home Assistant) case rebuilds the environment fresh from
        `_ha_url`/`_ha_token`/`_mcp_root` and this call's own `safety_block`
        -- a respawn's whole point is a *different* policy -- so only the
        policy differs between the old child and the new one. A host started
        with an explicit `env=` (and no `env_factory=`) repeats that mapping
        unchanged; this plan does not exercise that combination, since the
        weather server (D-13) never respawns. A host started with
        `env_factory=` (plan 06-01) instead awaits it fresh, with this
        call's own `safety_block` -- the hook a plugin's manager uses to
        rebuild a genuinely new environment (a changed config value, a new
        safety block for the enforcing plugin) rather than repeating the
        mapping the first child was started with.

        Task 3 (plan 04-01): this is now the writer half of a readers-writer
        pair with `call_tool()`. `_writer_waiting` is set the instant this
        method is entered -- before it ever waits for the in-flight count to
        reach zero -- so new readers block from this call's very first
        instant, not merely once teardown actually starts (otherwise a
        steady stream of turns could starve a policy change forever). Only
        once every already-in-flight `call_tool()` has returned does this
        close the old stack and spawn the replacement, preserving the
        property the old single lock actually bought: a call never lands on
        a session mid-teardown.

        Writer-vs-writer exclusion (CR-01, phase 4 code review): the whole
        body below runs under `self._respawn_lock`, so a second concurrent
        `respawn()` call queues behind the first one's entire drain-
        teardown-spawn sequence rather than racing it -- neither
        `self._stack` nor `self.session` is ever read or reassigned by two
        writers at once, and no child gets silently orphaned by a losing
        writer's fresh `AsyncExitStack()` overwriting a winning writer's.
        """
        if self._ha_url is None or self._ha_token is None or self._mcp_root is None:
            raise RuntimeError("McpToolHost.respawn called before start()")
        async with self._respawn_lock:
            if self._env_factory is not None:
                resolved_env = dict(await self._env_factory(safety_block))
            elif self._env_override is not None:
                resolved_env = dict(self._env_override)
            else:
                resolved_env = _default_ha_env(self._ha_url, self._ha_token, self._mcp_root, safety_block)
            # Both statements below run synchronously, with no `await`
            # between them -- no reader can observe one without the other
            # (this class's own docstring). From this instant, `call_tool`'s
            # own gate rejects every new caller until this method sets
            # `_readers_admitted` again on its way out.
            self._writer_waiting = True
            self._readers_admitted.clear()
            # Drain: every already-in-flight `call_tool()` decrements
            # `_in_flight` in its own `finally`, synchronously, so this will
            # observe zero as soon as the last one runs. `asyncio.sleep(0)`
            # yields to the event loop without imposing any minimum delay --
            # this notices the drain on the very next loop iteration in which
            # one was possible, not on a polling timer.
            while self._in_flight > 0:
                await asyncio.sleep(0)
            try:
                await self._stack.aclose()
                self._stack = AsyncExitStack()
                await self._spawn(self._child_module, resolved_env)
            finally:
                self._writer_waiting = False
                self._readers_admitted.set()

    async def _spawn(self, child_module: str, env: Mapping[str, str]) -> None:
        """The real-subprocess spawn both `start()` and `respawn()` perform --
        factored out so there is exactly one place that launches a child.
        Takes the module path and the environment as arguments (Task 3, plan
        04-01) rather than building either itself: `start()`/`respawn()` are
        the only two places that decide what a child's environment holds --
        the literal-not-inherited discipline this module's docstring states
        -- and two places building an environment, rather than one, is how
        that discipline drifts.
        """
        server_params = StdioServerParameters(
            command=sys.executable,
            args=["-m", child_module],
            env=dict(env),
        )
        read, write = await self._stack.enter_async_context(stdio_client(server_params))
        self.session = await self._stack.enter_async_context(ClientSession(read, write))
        await self.session.initialize()
        result = await self.session.list_tools()
        self.tools = result.tools

    @asynccontextmanager
    async def _as_reader(self) -> "AsyncIterator[ClientSession]":
        """Admit the caller as a reader under the readers-writer gate
        `respawn()` is the writer of, and yield the live session to call
        through -- shared by `call_tool()` and `ping()` (plan 06-02, D-05)
        so a lightweight liveness probe can never race a `respawn()`'s
        stack swap either, the identical discipline `call_tool()` always
        used before this method existed.

        Task 3 (plan 04-01): this is the reader half of a readers-writer
        pair with `respawn()`. Waits on `_readers_admitted` -- cleared from
        the instant a `respawn()` call begins, not only while it holds
        exclusive access, so this blocks from that same instant. Once
        admitted, this re-checks `_writer_waiting` before registering:
        `_readers_admitted` can be set again by a `respawn()` finishing at
        the same moment this call's `wait()` was already resolving on the
        OLD `set()` from before that same `respawn()` started, so a bare
        `wait()` return is not, by itself, proof no writer is active.
        Snapshotting `self.session` and incrementing `_in_flight` both
        happen in the same synchronous stretch as that re-check -- no
        `await` in between -- so a `respawn()` reading `_in_flight`
        afterwards never observes a half-registered call. The actual round
        trip runs OUTSIDE that stretch (inside the caller's own `async
        with` body), so two or more concurrent readers genuinely overlap
        rather than serializing behind each other. Deregisters in a
        `finally`, so a reader cancelled mid-flight still decrements the
        count and cannot wedge a later `respawn()` waiting on it to reach
        zero.
        """
        while True:
            await self._readers_admitted.wait()
            if self._writer_waiting:
                # A writer started (and cleared the event) in the gap
                # between this `wait()` resolving and this check -- loop
                # back and wait again rather than proceeding on stale
                # admission.
                continue
            session = self.session
            if session is None:
                raise RuntimeError("McpToolHost call made before start()")
            self._in_flight += 1
            break
        try:
            yield session
        finally:
            self._in_flight -= 1

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call one tool and return the SDK's own `CallToolResult` unchanged.

        This is what makes a refusal a distinguishable result rather than a
        generic error string: `ClientSession.call_tool` never raises for a
        tool-level failure. A `Denied` raised inside `spire_mcp.ha` becomes,
        on the wire, `CallToolResult(is_error=True, content=[TextContent(text=
        str(exc))])` -- and `str(exc)` on a `Denied` is exactly `reason` (see
        `spire_mcp.safety.Denied.__init__`). The turn controller reads
        `is_error` and `content[0].text` straight off this return value, so
        the reason crosses this boundary unchanged, not paraphrased.

        Reader half of the readers-writer pair with `respawn()` -- see
        `_as_reader`'s own docstring for the full discipline this shares
        with `ping()`.
        """
        async with self._as_reader() as session:
            return await session.call_tool(name, arguments)

    async def ping(self) -> None:
        """Cheap liveness probe (plan 06-02, D-05): the ping watchdog's own
        detection mechanism for a plugin that has died with no call
        pending against it -- `06-RESEARCH.md` Pattern 3 verified the
        installed SDK resolves every already-in-flight/newly-attempted
        `call_tool` automatically on child death, but nothing pushes that
        fact to a caller not currently calling. Raises whatever
        `session.send_ping()` raises (an `MCPError` on a dead session) --
        the watchdog interprets any raise as "this plugin is dead," never
        this method's job to classify.
        """
        async with self._as_reader() as session:
            await session.send_ping()

    async def aclose(self) -> None:
        await self._stack.aclose()


class UnknownToolError(Exception):
    """Raised by `McpToolHostLookup.call_tool` when no host in the lookup
    advertises the requested tool name.

    Never picks the first host as a fallback: a misrouted call is the
    failure mode that turns a weather question into a service call
    (T-04-12), so an unrecognized name must stop the turn rather than
    guess which child was meant.
    """


class AmbiguousToolError(Exception):
    """Raised by `McpToolHostLookup.__init__` when two or more of the
    hosts it was built over advertise the same tool name.

    Raised at construction, not at call time: a name landing on the wrong
    child by silent last-write-wins is exactly the misrouting T-04-12
    names, and a construction-time raise is the earliest point this
    lookup can refuse to be built ambiguous in the first place.
    """


class McpToolHostLookup:
    """A name-to-host lookup over an explicit, ordered sequence of
    already-started `McpToolHost` instances (D-13).

    Exposes exactly one method, `call_tool(name, arguments)` -- the same
    structural shape `turn/controller.py`'s `_ToolHost` Protocol already
    expects from a single host, so every `run_turn` call site changes
    only which object it passes, never how it calls it.

    Reads no configuration: it takes `hosts` as a constructor argument and
    builds its name-to-host map from each host's own `.tools` list
    (populated by that host's own `start()`), never from a server-block
    mapping parsed anywhere else in this codebase. Building a
    configuration-driven router that spawns hosts of its own is Phase 6's
    job, with Phase 6's information (04-CONTEXT.md D-13) -- this object
    only ever routes among hosts it was handed, already running.
    """

    def __init__(self, hosts: "Sequence[Any]") -> None:
        self._by_tool_name: dict[str, Any] = {}
        claimed_by_index: dict[str, int] = {}
        for index, host in enumerate(hosts):
            for tool in host.tools:
                if tool.name in self._by_tool_name:
                    raise AmbiguousToolError(
                        f"tool {tool.name!r} is advertised by more than one host in "
                        f"this lookup (host {claimed_by_index[tool.name]} and host "
                        f"{index}) -- a name two hosts both advertise cannot be "
                        "routed unambiguously"
                    )
                self._by_tool_name[tool.name] = host
                claimed_by_index[tool.name] = index

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        host = self._by_tool_name.get(name)
        if host is None:
            raise UnknownToolError(
                f"no host in this lookup advertises tool {name!r} -- known tools: "
                f"{sorted(self._by_tool_name)!r}"
            )
        return await host.call_tool(name, arguments)


_MISSING = object()


def mcp_tools_to_openai_tools(tools: list[Tool]) -> list[dict[str, Any]]:
    """Rename and nest each MCP tool's schema into the `tools=[...]` shape
    a chat-completions call expects. A rename and a nest, not a rewrite --
    the schema already is a JSON Schema object.

    The installed `mcp>=2.2,<3` SDK's `Tool` model exposes this field as the
    Python attribute `input_schema`; `inputSchema` is only its wire-format
    alias (`model_dump(by_alias=True)`), not an accessible attribute -- a
    bare `tool.inputSchema` raises `AttributeError` against a real `Tool`.
    `getattr` with a fallback, matching `app.py::_tool_result_json` and
    `turn/controller.py::_is_error`'s own camelCase/snake_case handling,
    keeps this working against either shape.

    The fallback checks presence with a sentinel, not truthiness with `or`
    -- a schema that legitimately serializes to `{}` must still win over
    the second attribute name, and a `Tool` exposing neither name must
    still fail loudly here rather than silently sending `None` as a tool's
    `parameters` inside a live chat-completions call.
    """
    results = []
    for tool in tools:
        schema = getattr(tool, "input_schema", _MISSING)
        if schema is _MISSING:
            schema = getattr(tool, "inputSchema", _MISSING)
        if schema is _MISSING:
            raise AttributeError(f"MCP Tool {tool.name!r} exposes neither input_schema nor inputSchema")
        results.append(
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description or "",
                    "parameters": schema,
                },
            }
        )
    return results
