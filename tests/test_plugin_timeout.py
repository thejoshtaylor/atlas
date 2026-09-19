"""D-06, PLUG-06: a tool call that exceeds its plugin's own deadline fails
that call alone.

`McpToolHost.call_tool` passes each host's own `_timeout_s` (set at
`start()`, from `plugin.timeout_ms` -- `plugins/host.py`) straight to the
SDK's own `ClientSession.call_tool(..., read_timeout_seconds=...)`
parameter -- no `asyncio.wait_for` wrapper (`06-RESEARCH.md` Pattern 2:
the SDK's own timeout path already sends the courtesy cancel and
unconditionally deregisters the waiter). A timeout (`MCPError` with
`code == REQUEST_TIMEOUT`) becomes the same `CallToolResult(is_error=True,
...)` shape a `Denied` already crosses this boundary with; any other
`MCPError` still surfaces.

These tests wire a `McpToolHost` directly to a fake session (the same
`_host_with_fake_session`-shaped pattern `tests/test_mcp_client.py`
already uses) -- bypassing `start()`/`_spawn` entirely, since the property
under test is `call_tool`'s own timeout-conversion logic, not a real
subprocess.
"""

from __future__ import annotations

from typing import Any

import pytest
from mcp.shared.exceptions import MCPError
from mcp_types.jsonrpc import INVALID_PARAMS, REQUEST_TIMEOUT

from spire_voice.mcp_client import McpToolHost

_MCP_ROOT = "/nonexistent/mcp-root"  # never read: _spawn is bypassed in every test here


def _host_with_fake_session(session: Any, *, timeout_s: float | None) -> McpToolHost:
    """A `McpToolHost` wired directly to a fake session, with `_timeout_s`
    set the way a real `start()` would (via `plugins/host.py`, from
    `plugin.timeout_ms / 1000`) -- bypassing `start()`/`_spawn` since these
    tests exercise `call_tool`'s own timeout-conversion logic, not a real
    subprocess."""
    host = McpToolHost()
    host.session = session
    host._ha_url = "http://ha.invalid:8123"
    host._ha_token = "test-key"
    host._mcp_root = _MCP_ROOT
    host._child_module = "spire_mcp.ha"
    host._timeout_s = timeout_s
    return host


class _TimingOutSession:
    """A fake `ClientSession` whose `call_tool` raises the SDK's own
    timeout error on its first call, then answers normally -- proving
    both PLUG-06 halves against one session object: the timed-out call
    fails alone, and the very next call on that same session succeeds."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, dict, float | None]] = []

    async def call_tool(
        self, name: str, arguments: dict, read_timeout_seconds: float | None = None
    ):
        self.calls.append((name, dict(arguments), read_timeout_seconds))
        if len(self.calls) == 1:
            raise MCPError(code=REQUEST_TIMEOUT, message=f"Request {name!r} timed out")
        from types import SimpleNamespace

        return SimpleNamespace(is_error=False, content=[SimpleNamespace(text="ok")])


class _AlwaysProtocolErrorSession:
    """A fake `ClientSession` whose `call_tool` always raises a genuine
    (non-timeout) `MCPError` -- proving a real protocol error still
    surfaces rather than being swallowed as a timeout."""

    async def call_tool(
        self, name: str, arguments: dict, read_timeout_seconds: float | None = None
    ):
        raise MCPError(code=INVALID_PARAMS, message="simulated: genuinely malformed request")


class _CapturingSession:
    """Records the `read_timeout_seconds` it was called with, and never
    raises -- proving each host passes its own configured deadline."""

    def __init__(self) -> None:
        self.read_timeout_seconds: float | None = "not called"

    async def call_tool(
        self, name: str, arguments: dict, read_timeout_seconds: float | None = None
    ):
        from types import SimpleNamespace

        self.read_timeout_seconds = read_timeout_seconds
        return SimpleNamespace(is_error=False, content=[SimpleNamespace(text="ok")])


async def test_a_call_over_its_deadline_returns_an_error_result_rather_than_raising():
    """Truth 1 (PLUG-06): a tool whose plugin never answers within its own
    deadline returns a readable, error-flagged result -- never raises into
    the turn."""
    session = _TimingOutSession()
    host = _host_with_fake_session(session, timeout_s=5.0)

    result = await host.call_tool("ha_call_service", {"domain": "switch"})

    assert result.is_error is True
    assert "ha_call_service" in result.content[0].text
    assert "5000ms" in result.content[0].text


async def test_the_session_survives_a_timeout_and_answers_the_next_call_normally():
    """Truth 2 (PLUG-06): the session a timed-out call ran on answers the
    very next call normally -- a timeout never tears the child down."""
    session = _TimingOutSession()
    host = _host_with_fake_session(session, timeout_s=5.0)

    timed_out_result = await host.call_tool("ha_call_service", {"domain": "switch"})
    assert timed_out_result.is_error is True

    next_result = await host.call_tool("ha_call_service", {"domain": "switch"})
    assert not getattr(next_result, "is_error", False)
    assert next_result.content[0].text == "ok"
    assert len(session.calls) == 2


async def test_two_plugins_with_different_deadlines_each_get_their_own():
    """Truth 3 (D-06): two `McpToolHost` instances, each configured with a
    different `timeout_s`, pass their own value through to the SDK's own
    `read_timeout_seconds` parameter -- no shared global deadline."""
    fast_session = _CapturingSession()
    slow_session = _CapturingSession()
    fast_host = _host_with_fake_session(fast_session, timeout_s=1.0)
    slow_host = _host_with_fake_session(slow_session, timeout_s=30.0)

    await fast_host.call_tool("weather_current", {})
    await slow_host.call_tool("ha_call_service", {"domain": "switch"})

    assert fast_session.read_timeout_seconds == 1.0
    assert slow_session.read_timeout_seconds == 30.0


async def test_a_genuine_protocol_error_is_not_swallowed_as_a_timeout():
    """Truth 4 (PLUG-06): a real `MCPError` that is not a timeout still
    surfaces -- never quietly relabelled as one."""
    session = _AlwaysProtocolErrorSession()
    host = _host_with_fake_session(session, timeout_s=5.0)

    with pytest.raises(MCPError) as excinfo:
        await host.call_tool("ha_call_service", {"domain": "switch"})

    assert excinfo.value.code == INVALID_PARAMS


async def test_a_host_with_no_configured_timeout_passes_none_through():
    """A `McpToolHost` never given a `timeout_s` (every caller that
    predates this plan) passes `None` through unchanged -- the SDK's own
    "no deadline" default, not a newly-invented zero-second one."""
    session = _CapturingSession()
    host = _host_with_fake_session(session, timeout_s=None)

    await host.call_tool("ha_list_entities", {})

    assert session.read_timeout_seconds is None
