"""Real assertions for `mcp_tools_to_openai_tools` against a real MCP `Tool`.

CR-01 (phase 01 code review): the function once read `tool.inputSchema`
(camelCase) off each MCP `Tool`. The installed `mcp>=2.2,<3` SDK exposes
that field as the Python attribute `input_schema` -- `inputSchema` is only
a wire-serialization alias, not an attribute name -- so a bare
`tool.inputSchema` raised `AttributeError` on every real `Tool` and crashed
app startup outright. No test caught this because every other test in this
suite builds tools with hand-written doubles that already use whichever
shape the test author typed, never a real `mcp.types.Tool` instance. This
test builds one for real, the way the installed SDK does, so this class of
attribute-name drift fails loudly again if it ever recurs.
"""

import asyncio
import inspect
import os
from types import SimpleNamespace

import pytest
from mcp.types import Tool

from spire_voice.mcp_client import mcp_tools_to_openai_tools


def test_mcp_tools_to_openai_tools_reads_real_tool_input_schema():
    tool = Tool(
        name="ha_call_service",
        description="Call a Home Assistant service.",
        inputSchema={
            "type": "object",
            "properties": {"entity_id": {"type": "string"}},
        },
    )

    openai_tools = mcp_tools_to_openai_tools([tool])

    assert openai_tools == [
        {
            "type": "function",
            "function": {
                "name": "ha_call_service",
                "description": "Call a Home Assistant service.",
                "parameters": {
                    "type": "object",
                    "properties": {"entity_id": {"type": "string"}},
                },
            },
        }
    ]


def test_mcp_tools_to_openai_tools_defaults_missing_description_to_empty_string():
    tool = Tool(name="ha_list_entities", description=None, inputSchema={"type": "object", "properties": {}})

    (openai_tool,) = mcp_tools_to_openai_tools([tool])

    assert openai_tool["function"]["description"] == ""


def test_mcp_tools_to_openai_tools_keeps_a_falsy_but_present_schema():
    """WR-01 (phase 01 code review, iteration 2): CR-01's own fallback read
    `getattr(tool, "input_schema", None) or getattr(tool, "inputSchema",
    None)`. `or` is a truthiness test, not a presence test, so a schema
    that is genuinely present but falsy -- an empty dict -- was discarded
    and replaced with `None` from the second `getattr`, which resolves to
    `None` on the installed SDK because `inputSchema` is not a real
    attribute on `Tool`. That would send `{"parameters": None}` into a live
    `tools=[...]` array instead of the empty-but-valid schema the tool
    actually declared. A sentinel-based presence check keeps `{}` intact.
    """
    tool = Tool(name="ha_list_entities", description="List entities.", inputSchema={})

    (openai_tool,) = mcp_tools_to_openai_tools([tool])

    assert openai_tool["function"]["parameters"] == {}


def test_mcp_child_runs_under_this_interpreter_not_a_bare_python3():
    """The MCP child must inherit the parent's interpreter, not PATH's `python3`.

    The child imports the same `mcp` SDK this process does. A bare `python3`
    is whatever PATH resolves first, which outside an activated virtualenv is
    the system interpreter with no SDK installed. Because this repository has
    its own top-level `mcp/` directory, that failure does not surface as an
    honest "SDK not installed" -- the repo directory satisfies `import mcp` as
    a namespace package, and the child dies on `No module named 'mcp.server'`,
    taking the FastAPI lifespan down with it.

    No test that imports the tool handlers directly can catch this, because
    the import happens inside the child process at startup. This asserts on
    the spawn parameters instead, which is the one place the choice is made.

    Reads `McpToolHost._spawn`, not `.start` -- Phase 3's `respawn()` factors
    the literal spawn (env dict, `command=sys.executable`) into one private
    helper both `start()` and `respawn()` call, so there is exactly one
    place this choice is made rather than two that could drift apart.
    """
    from spire_voice.mcp_client import McpToolHost

    source = inspect.getsource(McpToolHost._spawn)
    assert "command=sys.executable" in source, (
        "McpToolHost's spawn helper must use sys.executable; a bare "
        '"python3" resolves to the system interpreter, which has no mcp SDK'
    )
    assert 'command="python3"' not in source, (
        "a bare python3 command is still present in McpToolHost's spawn helper"
    )


# --- Plan 03-02 Task 4: McpToolHost.respawn -- the only way a policy change
# reaches the enforcing process (D-10) ---

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MCP_ROOT = os.path.join(_REPO_ROOT, "mcp")


async def test_respawn_replaces_the_child_with_one_enforcing_the_new_policy():
    """A respawned host must expose the new policy, and the old child must
    be gone -- proven against a real spawned subprocess, the same way
    `tests/test_policy_repo.py`'s end-to-end test does."""
    from spire_voice.mcp_client import McpToolHost

    host = McpToolHost()
    try:
        await host.start(
            ha_url="http://ha.invalid:8123",
            ha_token="test-key",
            mcp_root=_MCP_ROOT,
            safety_block={"mode": "allow_all_except_denylist", "deny_entities": []},
        )
        old_pid = host.session
        assert old_pid is not None

        # Before respawn: the entity is not on the denylist. No real Home
        # Assistant is reachable at ha.invalid, so this may still come back
        # as an error (a connection failure) -- what matters is that it is
        # not *this* refusal, since the entity isn't denied yet.
        allowed_before = await host.call_tool(
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_respawn_target"},
        )
        before_text = allowed_before.content[0].text if allowed_before.content else ""
        assert "off limits" not in before_text

        await host.respawn(
            {"mode": "allow_all_except_denylist", "deny_entities": ["switch.example_respawn_target"]}
        )

        # After respawn: the same entity is denied by the new child -- the
        # old session object is gone, replaced by a session on the new
        # process (a different ClientSession instance).
        assert host.session is not old_pid
        denied_after = await host.call_tool(
            "ha_call_service",
            {"domain": "switch", "service": "turn_off", "entity_id": "switch.example_respawn_target"},
        )
        assert denied_after.is_error
        assert "off limits" in denied_after.content[0].text
    finally:
        await host.aclose()


async def test_respawned_child_environment_holds_exactly_four_keys_and_no_database_scheme():
    """SAFE-09/T-03-08: the environment dict the host builds for the
    respawned child, asserted against the dict itself -- not against source
    text -- must hold exactly `HA_URL`, `HA_TOKEN`, `PYTHONPATH`, and
    `SPIRE_SAFETY`, and no value may contain a database connection scheme.

    `_spawn` now takes `(child_module, env)` directly (Task 3, plan 04-01):
    the environment is built by `start()`/`respawn()` before `_spawn` is
    ever called, so intercepting `_spawn` itself is a simpler capture point
    than reimplementing the build logic here.
    """
    from spire_voice.mcp_client import McpToolHost

    captured_envs: list[dict] = []
    real_spawn = McpToolHost._spawn

    async def _capturing_spawn(self, child_module, env):
        captured_envs.append(dict(env))
        await real_spawn(self, child_module, env)

    host = McpToolHost()
    try:
        McpToolHost._spawn = _capturing_spawn
        await host.start(
            ha_url="http://ha.invalid:8123",
            ha_token="test-key",
            mcp_root=_MCP_ROOT,
            safety_block={"mode": "allow_all_except_denylist"},
        )
        await host.respawn({"mode": "allowlist_only", "allow_entities": ["light.example_x"]})
    finally:
        McpToolHost._spawn = real_spawn
        await host.aclose()

    assert len(captured_envs) == 2, "expected one env for start() and one for respawn()"
    respawned_env = captured_envs[-1]
    assert set(respawned_env.keys()) == {"HA_URL", "HA_TOKEN", "PYTHONPATH", "SPIRE_SAFETY"}
    for value in respawned_env.values():
        assert "postgresql" not in value.lower(), (
            f"a value in the respawned child's environment names a database scheme: {value!r}"
        )


# --- Task 3, plan 04-01: readers-writer coordination -- concurrent
# `call_tool` calls genuinely overlap, and `respawn()` waits for them to
# drain rather than tearing the session out from under one in flight ------


class _BlockingSession:
    """A fake `ClientSession` whose `call_tool` blocks on a per-`call_id`
    `asyncio.Event` until the test releases it. Records every `call_id` the
    instant it arrives, before blocking -- proving two callers both reached
    the session before either was released (genuine overlap), not two calls
    that happened to return at about the same wall-clock moment.
    """

    def __init__(self) -> None:
        self.arrived: list[str] = []
        self._gates: dict[str, asyncio.Event] = {}

    async def call_tool(
        self, name: str, arguments: dict, read_timeout_seconds: float | None = None
    ) -> SimpleNamespace:
        call_id = arguments["call_id"]
        self.arrived.append(call_id)
        gate = self._gates.setdefault(call_id, asyncio.Event())
        await gate.wait()
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=call_id)])

    def release(self, call_id: str) -> None:
        self._gates.setdefault(call_id, asyncio.Event()).set()


def _host_with_fake_session(session: _BlockingSession):
    """A `McpToolHost` wired directly to a fake session -- bypassing
    `start()`/`_spawn` entirely, since these tests exercise `call_tool`'s
    and `respawn()`'s own coordination logic, not a real subprocess. The
    spawn attributes are still set, the same way a real `start()` would set
    them, so `respawn()`'s own `None`-guard does not fire."""
    from spire_voice.mcp_client import McpToolHost

    host = McpToolHost()
    host.session = session
    host._ha_url = "http://ha.invalid:8123"
    host._ha_token = "test-key"
    host._mcp_root = _MCP_ROOT
    host._child_module = "spire_mcp.ha"
    return host


async def test_two_call_tool_calls_genuinely_overlap_at_the_session():
    """Both callers reach the fake session before either is released -- not
    two calls that happened to return at about the same moment (RESEARCH.md
    Pitfall 3's own warning sign)."""
    session = _BlockingSession()
    host = _host_with_fake_session(session)

    task_a = asyncio.create_task(host.call_tool("ha_call_service", {"call_id": "a"}))
    task_b = asyncio.create_task(host.call_tool("ha_call_service", {"call_id": "b"}))
    await asyncio.sleep(0)
    await asyncio.sleep(0)

    assert sorted(session.arrived) == ["a", "b"], (
        "both calls must have reached the fake session before either is released"
    )

    session.release("a")
    session.release("b")
    result_a = await task_a
    result_b = await task_b
    assert result_a.content[0].text == "a"
    assert result_b.content[0].text == "b"


async def test_respawn_started_while_a_call_is_in_flight_waits_for_it_before_tearing_down():
    """The property the old single lock actually bought -- a call never
    lands on a session `respawn()` is mid-teardown on -- must survive
    concurrent dispatch: `respawn()` must not spawn the replacement until
    the in-flight call has returned, and that call's own result must still
    arrive intact."""
    from spire_voice.mcp_client import McpToolHost

    old_session = _BlockingSession()
    host = _host_with_fake_session(old_session)
    new_session = _BlockingSession()
    spawn_calls: list[tuple[str, dict]] = []
    real_spawn = McpToolHost._spawn

    async def _fake_spawn(self, child_module, env):
        spawn_calls.append((child_module, dict(env)))
        self.session = new_session

    McpToolHost._spawn = _fake_spawn
    try:
        call_task = asyncio.create_task(host.call_tool("ha_call_service", {"call_id": "in_flight"}))
        await asyncio.sleep(0)
        assert old_session.arrived == ["in_flight"]

        respawn_task = asyncio.create_task(host.respawn({"mode": "allow_all_except_denylist"}))
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        # The respawn must still be waiting on the in-flight call to drain,
        # not racing ahead to spawn the replacement.
        assert not respawn_task.done()
        assert spawn_calls == []

        old_session.release("in_flight")
        result = await call_task
        assert result.content[0].text == "in_flight"

        await respawn_task
        assert len(spawn_calls) == 1
        assert host.session is new_session
    finally:
        McpToolHost._spawn = real_spawn
        await host.aclose()


async def test_a_call_cancelled_mid_flight_does_not_wedge_a_later_respawn():
    """A `call_tool` cancelled while awaiting the session must still
    decrement `_in_flight` in its `finally` -- otherwise one cancelled
    reader wedges every future `respawn()` forever, waiting on a count that
    can never reach zero again."""
    from spire_voice.mcp_client import McpToolHost

    session = _BlockingSession()
    host = _host_with_fake_session(session)
    new_session = _BlockingSession()
    spawn_calls: list[tuple[str, dict]] = []
    real_spawn = McpToolHost._spawn

    async def _fake_spawn(self, child_module, env):
        spawn_calls.append((child_module, dict(env)))
        self.session = new_session

    McpToolHost._spawn = _fake_spawn
    try:
        call_task = asyncio.create_task(host.call_tool("ha_call_service", {"call_id": "cancel_me"}))
        await asyncio.sleep(0)
        assert session.arrived == ["cancel_me"]
        assert host._in_flight == 1

        call_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await call_task
        assert host._in_flight == 0, "a cancelled reader must still deregister in its finally"

        # A respawn started AFTER the cancellation must proceed -- it must
        # not wait forever on an in-flight count the cancelled call left
        # stuck above zero.
        await asyncio.wait_for(host.respawn({"mode": "allow_all_except_denylist"}), timeout=1.0)
        assert len(spawn_calls) == 1
        assert host.session is new_session
    finally:
        McpToolHost._spawn = real_spawn
        await host.aclose()


async def test_two_concurrent_respawn_calls_serialize_instead_of_racing():
    """CR-01 (phase 4 code review): the readers-writer rewrite (Task 3, plan
    04-01) protects readers from a writer, but nothing in it stopped a
    SECOND concurrent `respawn()` from entering the same
    drain-teardown-spawn sequence in parallel with the first -- both would
    read `self._stack` before either reassigned it, so one respawn's fresh
    stack (and the child it just spawned) could be silently overwritten by
    the other's, leaking a still-running child enforcing a stale policy.

    This proves the fix with two real, concurrent `respawn()` calls raced
    via `asyncio.create_task` (not sequential awaits): the second must not
    enter `_spawn` while the first is still inside its own body, `_spawn`
    must be called exactly twice, in order, and `self.session` must end up
    on whichever spawn actually ran last (B's), not a reference clobbered
    by a race.
    """
    from spire_voice.mcp_client import McpToolHost

    old_session = _BlockingSession()
    host = _host_with_fake_session(old_session)
    spawn_envs: list[dict] = []
    first_spawn_entered = asyncio.Event()
    release_first_spawn = asyncio.Event()
    real_spawn = McpToolHost._spawn

    async def _fake_spawn(self, child_module, env):
        spawn_envs.append(dict(env))
        if len(spawn_envs) == 1:
            # Hold the first respawn inside `_spawn` -- the exact window
            # CR-01's interleaving needed a second writer to race through --
            # so the test can prove a second `respawn()` call queues behind
            # this one rather than entering `_spawn` concurrently.
            first_spawn_entered.set()
            await release_first_spawn.wait()
        self.session = SimpleNamespace(spawned_for=dict(env))

    McpToolHost._spawn = _fake_spawn
    try:
        respawn_a = asyncio.create_task(
            host.respawn({"mode": "allow_all_except_denylist", "deny_entities": ["switch.example_a"]})
        )
        await first_spawn_entered.wait()

        respawn_b = asyncio.create_task(
            host.respawn({"mode": "allow_all_except_denylist", "deny_entities": ["switch.example_b"]})
        )
        # Give B every chance to race ahead if the lock did not hold: two
        # bare yields is enough for B to clear the readers-admitted event,
        # observe zero in-flight readers, and reach `_stack.aclose()`/`_spawn`
        # if nothing serializes it behind A.
        await asyncio.sleep(0)
        await asyncio.sleep(0)

        assert len(spawn_envs) == 1, (
            "a second respawn() must not enter _spawn while the first is still inside its own body"
        )
        assert not respawn_b.done()

        release_first_spawn.set()
        await respawn_a
        await respawn_b

        assert len(spawn_envs) == 2, "each respawn() must reach _spawn exactly once, never zero, never racing in"
        assert "switch.example_a" in spawn_envs[0]["SPIRE_SAFETY"], "A must have spawned first"
        assert "switch.example_b" in spawn_envs[1]["SPIRE_SAFETY"], "B must have spawned second, not raced in early"
        # B ran strictly after A finished (the lock, not the event loop's
        # scheduling order, is what guarantees this) -- so the session B's
        # own _spawn call set must be the one still installed, not
        # overwritten by a stray reassignment from A.
        assert host.session.spawned_for == spawn_envs[1]
    finally:
        McpToolHost._spawn = real_spawn
        await host.aclose()


async def test_start_with_an_explicit_module_and_env_produces_exactly_that_environment():
    """D-13: a second `McpToolHost` instance (the weather server) owns a
    different child by parameter, not by subclass -- an explicit `env=`
    given to `start()` produces exactly that mapping, with no Home
    Assistant keys merged in from anywhere (SAFE-09). Does not actually
    launch a subprocess for the given `child_module` -- `spire_mcp.weather`
    is plan 04-02's own module, built in a parallel worktree, and this test
    only needs to prove what `start()` decided to spawn, not that the
    module exists in this checkout.
    """
    from spire_voice.mcp_client import McpToolHost

    captured: list[tuple[str, dict]] = []
    real_spawn = McpToolHost._spawn

    async def _capturing_spawn(self, child_module, env):
        captured.append((child_module, dict(env)))

    host = McpToolHost()
    try:
        McpToolHost._spawn = _capturing_spawn
        await host.start(
            ha_url="unused",
            ha_token="unused",
            mcp_root=_MCP_ROOT,
            child_module="spire_mcp.weather",
            env={"PYTHONPATH": _MCP_ROOT},
        )
    finally:
        McpToolHost._spawn = real_spawn
        await host.aclose()

    assert len(captured) == 1
    child_module, env = captured[0]
    assert child_module == "spire_mcp.weather"
    assert env == {"PYTHONPATH": _MCP_ROOT}
    assert "HA_URL" not in env
    assert "HA_TOKEN" not in env
    assert "SPIRE_SAFETY" not in env


class _PingRecordingSession:
    """Records exactly what `McpToolHost.ping()` hands the SDK -- WR-04
    (code review). The probe's deadline is the whole subject: a ping sent
    with no `request_read_timeout_seconds`, against a `ClientSession`
    constructed with no session-level `read_timeout_seconds` either, has
    no deadline anywhere, and a child that is alive but answering nothing
    wedges the watchdog that was supposed to notice it."""

    def __init__(self) -> None:
        self.requests: list[tuple[str, float | None]] = []

    async def send_request(self, request, result_type, request_read_timeout_seconds=None, **kwargs):
        self.requests.append((request.method, request_read_timeout_seconds))
        from mcp.types import EmptyResult

        return EmptyResult()

    async def send_ping(self):  # pragma: no cover -- must not be used any more
        raise AssertionError(
            "ping() must send a request carrying a read timeout, not the SDK's "
            "own send_ping(), which accepts no deadline"
        )


async def test_the_liveness_probe_carries_the_plugins_own_deadline():
    """WR-04 (code review): `_ping_interval_s`'s docstring promises a hung
    plugin is noticed "before it could plausibly still be mid-call", but
    the probe itself had no deadline of any kind -- so a wedged-but-alive
    child was never noticed at all, its tools were never withdrawn, no
    respawn was ever attempted, and the blocked probe (registered as a
    reader) left any later `respawn()` spinning in its drain loop.

    The deadline is the plugin's own `timeout_ms`, the same value
    `call_tool` passes, handed to the SDK rather than wrapped in an
    `asyncio.wait_for` of this codebase's own.
    """
    session = _PingRecordingSession()
    host = _host_with_fake_session(session)
    host._timeout_s = 2.5

    await host.ping()

    assert session.requests == [("ping", 2.5)]
