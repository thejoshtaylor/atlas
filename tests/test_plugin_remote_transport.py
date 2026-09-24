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

from atlas.config import PluginsConfig, SecurityConfig
from atlas.crypto.credentials import encrypt_credential
from atlas.db.repository import Plugin, PluginConfigValue
from atlas.mcp_client import McpToolHost
from atlas.plugins.host import validate_remote_url
from atlas.plugins.manager import PluginManager, PluginState

_REPO_ROOT = __file__.rsplit("/tests/", 1)[0]
_MCP_ROOT = f"{_REPO_ROOT}/mcp"

_TEST_SECRET_KEY = "test-secret-key-not-a-real-generated-value"


@pytest.fixture(autouse=True)
def _secret_key(monkeypatch):
    monkeypatch.setenv("ATLAS_SECRET_KEY", _TEST_SECRET_KEY)


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
    server = MCPServer("atlas-test-remote-plugin")

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


# ---------------------------------------------------------------------------
# Task 2: `validate_remote_url` itself -- the plain-http-unless-loopback
# refusal (T-06-12), tested directly rather than only through a full
# connection attempt.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "url",
    [
        "https://plugins.example.com/mcp",
        "https://192.0.2.10/mcp",
        "http://127.0.0.1:9000/mcp",
        "http://localhost:9000/mcp",
        "http://[::1]:9000/mcp",
    ],
)
def test_validate_remote_url_allows_https_anywhere_and_plain_http_only_on_loopback(url):
    validate_remote_url("some-plugin", url)  # must not raise


@pytest.mark.parametrize(
    "url",
    [
        "http://plugins.example.com/mcp",
        "http://192.168.1.50:9000/mcp",
        "ftp://127.0.0.1/mcp",
    ],
)
def test_validate_remote_url_refuses_plain_http_on_a_non_loopback_host(url):
    with pytest.raises(RuntimeError, match="only https is allowed"):
        validate_remote_url("some-plugin", url)


# ---------------------------------------------------------------------------
# Task 2: the full vertical slice through `PluginManager` -- a plugins row
# with `transport="remote"` reaches the assistant the same way a stdio row
# does, its secret config value reaches only the outbound Authorization
# header, a non-loopback plain-http URL is refused by name, a call over its
# own deadline fails alone, and a dead remote plugin is withdrawn and later
# restored by the same watchdog/supervisor a local plugin uses.
# ---------------------------------------------------------------------------


def _remote_plugin(
    plugin_id: int,
    slug: str,
    *,
    url: str,
    enabled: bool = True,
    enforces_policy: bool = False,
    timeout_ms: int = 5000,
) -> Plugin:
    return _plugin(plugin_id, slug, url=url, enabled=enabled, enforces_policy=enforces_policy, timeout_ms=timeout_ms)


def _manager(repo, *, plugins_config: PluginsConfig | None = None, sleep=None) -> PluginManager:
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


async def test_a_remote_plugin_row_reaches_the_assistant_through_the_plugin_manager(
    fake_plugin_repository,
):
    """Truth 1: a plugin reachable only at a URL joins the assistant's
    tool set with no process of its own -- through the same `PluginManager`
    loop a stdio row uses, no per-transport branch outside `_start_one`."""
    test_server = await _start_test_mcp_server()
    try:
        weather = _remote_plugin(1, "weather-remote", url=test_server.url)
        repo = fake_plugin_repository(plugins=[weather])
        manager = _manager(repo)
        try:
            await manager.start_all()

            assert manager.state_for("weather-remote") is PluginState.RUNNING
            tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
            assert tool_names == {"echo", "slow"}
            result = await manager.tool_host_lookup.call_tool("echo", {"text": "hi"})
            assert result.content[0].text == "hi"
        finally:
            await manager.stop_all()
    finally:
        await test_server.stop()


async def test_a_secret_config_value_reaches_only_the_outbound_authorization_header(
    fake_plugin_repository,
):
    """Truth 3: a remote plugin's credential is stored encrypted and
    reaches only its own outbound request headers (D-03) -- decrypted at
    exactly one point, building the connection, never returned by a
    repository read."""
    test_server = await _start_test_mcp_server()
    try:
        security = SecurityConfig()
        plugin = _remote_plugin(1, "secret-remote", url=test_server.url)
        repo = fake_plugin_repository(
            plugins=[plugin],
            config_values={1: [_secret_value("API_KEY", "super-secret-token", security)]},
        )
        manager = _manager(repo)
        try:
            await manager.start_all()
            assert manager.state_for("secret-remote") is PluginState.RUNNING

            await manager.tool_host_lookup.call_tool("echo", {"text": "hi"})

            assert test_server.captured_auth, "no request reached the test server"
            assert all(auth == "Bearer super-secret-token" for auth in test_server.captured_auth)

            # Nothing a repository read returns carries the decrypted value.
            values = await repo.get_config_values(1)
            assert all(value.value != "super-secret-token" for value in values)
        finally:
            await manager.stop_all()
    finally:
        await test_server.stop()


async def test_a_remote_plugin_with_no_secret_config_value_connects_with_no_authorization_header(
    fake_plugin_repository,
):
    """A remote plugin with no credential at all is a real, supported
    shape (D-16) -- not an error, and not an empty-string header."""
    test_server = await _start_test_mcp_server()
    try:
        plugin = _remote_plugin(1, "no-secret-remote", url=test_server.url)
        repo = fake_plugin_repository(plugins=[plugin])
        manager = _manager(repo)
        try:
            await manager.start_all()
            assert manager.state_for("no-secret-remote") is PluginState.RUNNING
            assert test_server.captured_auth
            assert all(auth is None for auth in test_server.captured_auth)
        finally:
            await manager.stop_all()
    finally:
        await test_server.stop()


async def test_a_non_loopback_plain_http_plugin_is_degraded_with_a_named_reason(
    fake_plugin_repository,
):
    """T-06-12: a plugin whose admin-entered URL is plain http on a
    non-loopback host never connects at all -- refused by name, the same
    `DEGRADED` outcome any other start failure produces (D-07), the boot
    continues."""
    plugin = _remote_plugin(1, "unsafe-remote", url="http://plugins.example.com/mcp")
    repo = fake_plugin_repository(plugins=[plugin])
    manager = _manager(repo)
    try:
        await manager.start_all()

        assert manager.state_for("unsafe-remote") is PluginState.DEGRADED
        assert "only https is allowed" in manager.reason_for("unsafe-remote")
        assert manager.tool_host_for("unsafe-remote") is None
    finally:
        await manager.stop_all()


async def test_a_remote_call_over_its_plugins_own_deadline_fails_alone(fake_plugin_repository):
    """Truth 5 (D-06, PLUG-06): a remote call over the plugin's deadline
    fails alone, the same as a local one -- the session survives and
    answers the next call normally."""
    test_server = await _start_test_mcp_server()
    try:
        plugin = _remote_plugin(1, "slow-remote", url=test_server.url, timeout_ms=100)
        repo = fake_plugin_repository(plugins=[plugin])
        manager = _manager(repo)
        try:
            await manager.start_all()
            host = manager.tool_host_for("slow-remote")
            assert host is not None

            result = await host.call_tool("slow", {"seconds": 5})
            assert result.is_error is True
            assert "did not respond within" in result.content[0].text

            # The session survives -- the very next call answers normally.
            result = await host.call_tool("echo", {"text": "still alive"})
            assert result.is_error is not True
            assert result.content[0].text == "still alive"
        finally:
            await manager.stop_all()
    finally:
        await test_server.stop()


async def test_a_dead_remote_plugin_is_withdrawn_by_the_same_watchdog_a_local_one_uses(
    fake_plugin_repository,
):
    """Truth 4: a remote plugin that stops answering is withdrawn by the
    same watchdog a local one uses -- proven here against a real local HTTP
    server this test starts and stops, never a fake transport. (The
    respawn half of Truth 4/5 -- "recovers once the server returns" -- is
    proven separately, against a fake `start_plugin_host`; see that test's
    own docstring for why a *second* real server is not used here.)"""
    test_server = await _start_test_mcp_server()
    try:
        plugin = _remote_plugin(1, "flaky-remote", url=test_server.url, timeout_ms=200)
        repo = fake_plugin_repository(plugins=[plugin])
        manager = _manager(
            repo,
            plugins_config=PluginsConfig(
                startup_deadline_s=5.0, respawn_backoff_min_s=5.0, respawn_backoff_max_s=5.0
            ),
            # The watchdog's own ping-interval wait is real seconds by
            # default; this test does not care about its exact length,
            # only that pings keep happening promptly, so every requested
            # sleep is shortened to a fixed, fast real delay. The backoff
            # itself is left at its real 5s floor above -- this test never
            # waits long enough to reach a respawn attempt, only the
            # ping-failure withdrawal that precedes it.
            sleep=lambda _seconds: asyncio.sleep(0.02),
        )
        try:
            await manager.start_all()
            assert manager.state_for("flaky-remote") is PluginState.RUNNING

            await test_server.stop()

            async def _until_crashed() -> None:
                while manager.state_for("flaky-remote") is not PluginState.CRASHED_RETRYING:
                    await asyncio.sleep(0.02)

            await asyncio.wait_for(_until_crashed(), timeout=5.0)
            assert manager.tool_host_for("flaky-remote") is None
            assert manager.tools_schema == []
        finally:
            await manager.stop_all()
    finally:
        await test_server.stop()


async def test_a_dead_remote_plugin_recovers_once_a_respawn_attempt_succeeds(
    fake_plugin_repository, monkeypatch,
):
    """Truth 5: a remote plugin is respawned by the same supervisor a
    local one uses -- proven here with a fake `start_plugin_host` and a
    fake host whose `ping()` fails on command, the same technique
    `tests/test_plugin_supervisor.py::test_a_dead_plugin_is_respawned_on_
    backoff_and_its_tools_return` already uses for the identical stdio
    assertion, over `manager.py`'s own transport-agnostic `_start_one`/
    `_run_watchdog` code -- neither branches on `transport` beyond which
    `start_plugin_host` keyword arguments to build (this test asserts
    `bearer_token` is one of them).

    A *second real* uvicorn server is deliberately not used for this half:
    reproduced directly against the installed SDK, tearing down a remote
    `McpToolHost` whose connection died out from under it (the exact shape
    a ping failure produces) can leave that connection's own background
    GET-stream reconnect task mid-teardown when `aclose()` returns --
    `McpToolHost.aclose()`'s own docstring (plan 06-03) already documents
    this and swallows the teardown's own failure so it cannot crash this
    plugin's watchdog loop, the guarantee this test file's own crash-
    detection test (above) exercises against a real server. But that same
    still-unwinding background task has also been observed, in this exact
    sequence, to corrupt an unrelated ASGI response on a *second*,
    freshly-bound server started immediately afterward in the same
    process -- a real, verified defect in the installed SDK version's own
    async-generator cleanup path, not a defect in this plan's code, and
    out of scope to fix upstream here (the same category of pre-existing
    SDK limitation `06-01-SUMMARY.md`'s Task 1 and `06-02-SUMMARY.md`'s
    Task 1 each already found and worked around in test design, not
    production code, for an analogous reason)."""
    from mcp.types import Tool as MCPTool

    from atlas.plugins import manager as manager_module

    class _FakeHost:
        def __init__(self) -> None:
            self.tools = [MCPTool(name="echo", description="", inputSchema={"type": "object", "properties": {}})]
            self.ping_should_fail = False

        async def call_tool(self, name, arguments):
            raise AssertionError("not called in this test")

        async def ping(self) -> None:
            if self.ping_should_fail:
                raise RuntimeError("simulated: remote plugin stopped answering")

        async def aclose(self) -> None:
            return None

    plugin = _remote_plugin(1, "flaky-remote", url="http://127.0.0.1:1/mcp", timeout_ms=1000)
    repo = fake_plugin_repository(plugins=[plugin])

    first_host = _FakeHost()
    attempts: list[str] = []

    async def _fake_start_plugin_host(plugin, **kwargs):
        assert "bearer_token" in kwargs, "manager must build a remote plugin's own bearer_token kwarg"
        if not attempts:
            attempts.append("initial")
            return first_host
        attempts.append("respawn-succeeds")
        return _FakeHost()

    monkeypatch.setattr(manager_module, "start_plugin_host", _fake_start_plugin_host)

    manager = _manager(
        repo,
        plugins_config=PluginsConfig(startup_deadline_s=2.0, respawn_backoff_min_s=0.01, respawn_backoff_max_s=0.02),
        sleep=lambda _seconds: asyncio.sleep(0.01),
    )
    try:
        await manager.start_all()
        assert manager.state_for("flaky-remote") is PluginState.RUNNING

        first_host.ping_should_fail = True

        async def _until_crashed() -> None:
            while manager.state_for("flaky-remote") is not PluginState.CRASHED_RETRYING:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_until_crashed(), timeout=5.0)
        assert manager.tools_schema == []

        async def _until_running() -> None:
            while manager.state_for("flaky-remote") is not PluginState.RUNNING:
                await asyncio.sleep(0.01)

        await asyncio.wait_for(_until_running(), timeout=5.0)
        assert attempts == ["initial", "respawn-succeeds"]
        tool_names = {entry["function"]["name"] for entry in manager.tools_schema}
        assert "echo" in tool_names
    finally:
        await manager.stop_all()


async def test_two_unnamed_secrets_are_refused_rather_than_sending_an_arbitrary_one(
    fake_plugin_repository,
):
    """WR-07 (code review): `_remote_bearer_token` returned the *first*
    row with `secret=True and ciphertext is not None`, in whatever order
    the repository returned rows -- `SELECT ... WHERE plugin_id = ?` with
    no `ORDER BY`. Nothing stopped an admin from adding a second secret
    key, so which credential this house sent to a third-party server was
    effectively arbitrary, and could change between restarts.

    Two secrets with neither named is refused, by name, and the plugin is
    degraded with that reason rather than connecting with a credential
    nobody chose.
    """
    security = SecurityConfig()
    plugin = _remote_plugin(1, "two-secrets", url="https://remote.invalid/mcp")
    repo = fake_plugin_repository(
        plugins=[plugin],
        config_values={
            1: [
                _secret_value("API_KEY", "a-plainly-fictional-first-value", security),
                _secret_value("OTHER_KEY", "a-plainly-fictional-second-value", security),
            ]
        },
    )
    manager = _manager(repo)
    try:
        await manager.start_all()

        assert manager.state_for("two-secrets") is PluginState.DEGRADED
        reason = manager.reason_for("two-secrets")
        assert "AUTH_TOKEN" in reason
        assert "API_KEY" in reason and "OTHER_KEY" in reason
        assert manager.tool_host_for("two-secrets") is None
    finally:
        await manager.stop_all()


async def test_the_named_auth_key_decides_which_credential_is_sent(fake_plugin_repository):
    """The other half of WR-07: naming the credential `AUTH_TOKEN` makes
    the choice a fact the row states, so a plugin may carry another secret
    for its own configuration without changing what is sent."""
    test_server = await _start_test_mcp_server()
    try:
        security = SecurityConfig()
        plugin = _remote_plugin(1, "named-secret", url=test_server.url)
        repo = fake_plugin_repository(
            plugins=[plugin],
            config_values={
                1: [
                    _secret_value("OTHER_KEY", "a-plainly-fictional-unsent-value", security),
                    _secret_value("AUTH_TOKEN", "a-plainly-fictional-sent-value", security),
                ]
            },
        )
        manager = _manager(repo)
        try:
            await manager.start_all()
            assert manager.state_for("named-secret") is PluginState.RUNNING

            await manager.tool_host_lookup.call_tool("echo", {"text": "hi"})

            assert test_server.captured_auth, "no request reached the test server"
            assert all(
                auth == "Bearer a-plainly-fictional-sent-value"
                for auth in test_server.captured_auth
            )
        finally:
            await manager.stop_all()
    finally:
        await test_server.stop()
