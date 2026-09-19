"""Plan 06-03 (D-02, D-04): the second transport PLUG-04 names -- a plugin
reachable only at a URL, with no process of its own. `McpToolHost`'s own
generalized connect step (`mcp_client.py::McpToolHost._spawn`, Task 1) is
what a remote plugin connects through; `plugins.host.start_plugin_host`
(Task 2) is what turns a `Plugin` row with `transport="remote"` into one,
building the caller-owned `httpx2.AsyncClient` and its bearer header, and
refusing a plain-http URL that is not on loopback.

Every test in this file runs a real Starlette/uvicorn MCP server bound to
127.0.0.1 -- a server this test itself starts and stops -- never a fake
transport and never a real external host (this plan's own `<verification>`
instruction). The assistant side never binds a listening socket for
either transport: every connection here is outbound, from the test's own
client code standing in for `McpToolHost`, to the test's own server.
"""

from __future__ import annotations

import asyncio
import socket
from dataclasses import dataclass, field
from datetime import datetime, timezone

import pytest
import uvicorn
from mcp.server.mcpserver import MCPServer
from mcp.shared._httpx_utils import create_mcp_http_client

from spire_voice.config import PluginsConfig, SecurityConfig
from spire_voice.crypto.credentials import encrypt_credential
from spire_voice.db.repository import Plugin, PluginConfigValue
from spire_voice.mcp_client import McpToolHost

_REPO_ROOT = __file__.rsplit("/tests/", 1)[0]
_MCP_ROOT = f"{_REPO_ROOT}/mcp"

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("SPIRE_SECRET_KEY", _TEST_SECRET_KEY)


def _free_port() -> int:
    """A loopback TCP port free at the instant this returns -- the usual
    bind-to-0-then-close trick. A real race window exists between this
    call and uvicorn's own bind, the same one every test in this suite
    that picks its own port already accepts; this file starts and stops
    its own server, never sharing a port with another test process."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


class _HeaderCapturingASGIApp:
    """Wraps an ASGI app, recording each HTTP request's `Authorization`
    header (or `None` if absent) into `captured` before forwarding --
    proves a remote plugin's secret config value reaches the outbound
    request as a bearer header and nowhere else this test can observe.
    Passes every non-`http` scope (`lifespan`, in particular -- the
    Starlette app this wraps starts its own `StreamableHTTPSessionManager`
    from an ASGI lifespan event) straight through unexamined.
    """

    def __init__(self, app, captured: list[str | None]) -> None:
        self._app = app
        self._captured = captured

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "http":
            headers = dict(scope.get("headers") or [])
            auth = headers.get(b"authorization")
            self._captured.append(auth.decode("utf-8") if auth is not None else None)
        await self._app(scope, receive, send)


@dataclass
class _RunningTestServer:
    """A real MCP server bound to loopback, for the duration of one test
    -- `url` is the streamable-HTTP endpoint `McpToolHost`/`start_plugin_
    host` connect to; `captured_auth` is every request's own `Authorization`
    header, in arrival order."""

    port: int
    _server: uvicorn.Server
    _task: "asyncio.Task[None]"
    captured_auth: list[str | None] = field(default_factory=list)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/mcp"

    async def stop(self) -> None:
        self._server.should_exit = True
        await self._task


async def _start_test_mcp_server(*, port: int | None = None) -> _RunningTestServer:
    """Build and start a real streamable-HTTP MCP server with two tools:
    `echo` (returns its own argument) and `slow` (sleeps for the given
    number of seconds before answering) -- `slow` is what proves a call
    over its own deadline fails alone (D-06, PLUG-06) without tearing the
    session down."""
    server = MCPServer("spire-voice-test-remote-plugin")

    @server.tool()
    def echo(text: str) -> str:
        return text

    @server.tool()
    async def slow(seconds: float) -> str:
        await asyncio.sleep(seconds)
        return "done"

    captured_auth: list[str | None] = []
    app = _HeaderCapturingASGIApp(server.streamable_http_app(), captured_auth)
    port = port if port is not None else _free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="error")
    uv_server = uvicorn.Server(config)
    task = asyncio.create_task(uv_server.serve())
    for _ in range(500):
        if uv_server.started:
            break
        await asyncio.sleep(0.01)
    else:
        task.cancel()
        raise RuntimeError("test MCP server did not start listening in time")
    return _RunningTestServer(port=port, _server=uv_server, _task=task, captured_auth=captured_auth)


def _plugin(
    plugin_id: int,
    slug: str,
    *,
    url: str | None,
    enabled: bool = True,
    enforces_policy: bool = False,
    timeout_ms: int = 5000,
) -> Plugin:
    now = datetime.now(timezone.utc)
    return Plugin(
        id=plugin_id,
        slug=slug,
        display_name=slug,
        transport="remote",
        args=(),
        url=url,
        enabled=enabled,
        builtin=False,
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


# ---------------------------------------------------------------------------
# Task 1: the generalized connect step, exercised directly against a real
# local HTTP MCP server -- proves `McpToolHost` itself (not yet the plugin-
# level wrapper) is transport-agnostic past its one connect step.
# ---------------------------------------------------------------------------


async def test_a_host_constructed_for_a_remote_plugin_connects_and_lists_tools():
    """Behavior 3 (Task 1): a host constructed for a remote plugin exposes
    the same tool list and the same call method a stdio one does -- no
    change to `McpToolHostLookup`/`mcp_tools_to_openai_tools` is needed."""
    test_server = await _start_test_mcp_server()
    try:
        host = McpToolHost()
        await host.start(
            "",
            "",
            mcp_root=_MCP_ROOT,
            transport="remote",
            url=test_server.url,
            http_client_factory=create_mcp_http_client,
        )
        try:
            assert {tool.name for tool in host.tools} == {"echo", "slow"}
            result = await host.call_tool("echo", {"text": "hello"})
            assert result.is_error is not True
            assert result.content[0].text == "hello"
        finally:
            await host.aclose()
    finally:
        await test_server.stop()
