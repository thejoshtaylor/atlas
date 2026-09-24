"""Per-plugin host: the thin wrapper that actually connects a plugin
through `McpToolHost` -- never a second spawn path beside it (D-01 .. D-04,
Pitfall 2 in 06-RESEARCH.md). Building a plugin's own literal environment
(the config-value read, the secret decryption) is `PluginManager`'s job
(`atlas.plugins.manager`), since that is the one place holding the
repository and the security config both -- this module only turns an
already-built environment (or an `env_factory` that builds one), or an
already-decrypted bearer token, into a real connection.

Plan 06-02 (D-06, PLUG-06): this is also the one place `plugin.timeout_ms`
turns into `McpToolHost`'s own `timeout_s` -- read from the plugin's own
row, in milliseconds, so a local Home Assistant call and a remote weather
API never share one deadline (D-06's own rejection of a single global
timeout). `McpToolHost.call_tool` passes this value straight through to
the SDK's own `read_timeout_seconds` parameter on every call this plugin's
host makes -- no wrapper, no second place a deadline could drift from
this one.

Plan 06-03 (D-02, D-04): a `transport="remote"` row connects over
streamable HTTP instead of spawning a child. The HTTP client is built
here, through the SDK's own `create_mcp_http_client` helper, with the
plugin's decrypted secret config value (if any) as a bearer authorization
header -- the one seam the SDK exposes for exactly this, per
`06-RESEARCH.md`'s own "don't hand-roll an auth client" finding. A URL
whose scheme is not HTTPS is refused unless its host is a loopback address
(T-06-12): an admin-entered URL is an outbound-request surface this
process reaches from inside the house network, and a credential sent
there in the clear would cross that network unencrypted. This does not,
and is not meant to, stop this process from reaching another address on
the house network at all (T-06-13) -- that is what an admin-entered URL is
for; only the wire encryption of a non-loopback destination is enforced
here.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from ipaddress import ip_address
from typing import TYPE_CHECKING, AsyncIterator, Mapping, Sequence
from urllib.parse import urlsplit

from mcp.shared._httpx_utils import create_mcp_http_client

from atlas.db.repository import Plugin
from atlas.mcp_client import McpToolHost

if TYPE_CHECKING:
    import httpx2

    from atlas.mcp_client import EnvFactory


def module_for_stdio_args(slug: str, args: Sequence[str]) -> str:
    """The stdio child's importable module name, from a plugin row's own
    `args` (`["-m", "atlas_mcp.ha"]`) -- the same `-m <module>` shape
    `McpServerConfig`/`McpToolHost._spawn` already require (D-02: the
    interpreter is `sys.executable`, never configurable, so `args` never
    carries anything else)."""
    if len(args) != 2 or args[0] != "-m":
        raise RuntimeError(
            f"plugin {slug!r} has stdio args {list(args)!r} -- expected exactly "
            "['-m', '<module>'], the only shape a stdio plugin child's args take"
        )
    return args[1]


def _is_loopback_host(hostname: str | None) -> bool:
    """Whether `hostname` names this machine itself -- `localhost` by
    convention, or any address `ipaddress` classifies loopback (127.0.0.0/8,
    `::1`). A hostname that fails to parse as an IP address and is not the
    literal string `localhost` is never loopback -- this never resolves
    DNS, so a hostname that merely *points at* loopback in someone's `/etc/
    hosts` is not trusted here; only the address forms an admin would type
    to run a plugin beside this process are."""
    if hostname is None:
        return False
    if hostname == "localhost":
        return True
    try:
        return ip_address(hostname).is_loopback
    except ValueError:
        return False


def validate_remote_url(slug: str, url: str) -> None:
    """Refuse a scheme that is not HTTPS unless `url`'s host is a loopback
    address (T-06-12) -- the one address-level restriction this phase adds.
    A plain-http credential crossing the house network in the clear is
    exactly what this stops; it does not, and is not meant to, stop this
    process from reaching any other address on the house network at all
    (T-06-13) -- an admin-entered URL is an outbound-request surface by
    design, bounded by the admin role and nothing else."""
    parsed = urlsplit(url)
    if parsed.scheme == "https":
        return
    if parsed.scheme == "http" and _is_loopback_host(parsed.hostname):
        return
    raise RuntimeError(
        f"plugin {slug!r} has remote url {url!r} -- only https is allowed for a "
        "non-loopback host, since a plain-http credential would cross the house "
        "network in the clear"
    )


async def start_plugin_host(
    plugin: Plugin,
    *,
    mcp_root: "os.PathLike[str] | str",
    safety_block: dict | None,
    env: "Mapping[str, str] | None" = None,
    env_factory: "EnvFactory | None" = None,
    bearer_token: str | None = None,
) -> McpToolHost:
    """Connect `plugin` through `McpToolHost` -- never a second spawn path
    (D-04, Pitfall 2). Exactly one of `env`/`env_factory` is meaningful per
    stdio call: `env_factory`, when given, takes precedence in
    `McpToolHost.start()` itself (see that method's own docstring), and is
    threaded through so a later `respawn()` rebuilds this same plugin's
    environment fresh rather than repeating the mapping this call built.

    `transport="remote"` (plan 06-03, D-02) ignores `env`/`env_factory`
    entirely -- neither has meaning for a connection with no process of
    its own -- and instead builds the caller-owned `httpx2.AsyncClient`
    `McpToolHost` will enter into its own exit stack (T-06-15), with
    `bearer_token` (already decrypted by `PluginManager`, at the same
    server-side decryption points D-03 names, the same set the stdio side
    decrypts from) as its outbound
    `Authorization` header, and nowhere else. `bearer_token=None` builds a
    client with no `Authorization` header at all -- a remote plugin with no
    secret config value is a real, supported shape (D-16's plain key/value
    configuration), not an error.
    """
    host = McpToolHost()
    timeout_s = plugin.timeout_ms / 1000.0
    if plugin.transport == "remote":
        if not plugin.url:
            raise RuntimeError(f"plugin {plugin.slug!r} has transport 'remote' but no url")
        validate_remote_url(plugin.slug, plugin.url)
        url = plugin.url

        def _http_client_factory() -> "httpx2.AsyncClient":
            headers = {"Authorization": f"Bearer {bearer_token}"} if bearer_token else {}
            return create_mcp_http_client(headers=headers)

        async with _closed_if_it_never_starts(host):
            await host.start(
                "",
                "",
                mcp_root=mcp_root,
                safety_block=safety_block,
                transport="remote",
                url=url,
                http_client_factory=_http_client_factory,
                timeout_s=timeout_s,
            )
        return host

    module = module_for_stdio_args(plugin.slug, plugin.args)
    async with _closed_if_it_never_starts(host):
        await host.start(
            ha_url="",
            ha_token="",
            mcp_root=mcp_root,
            safety_block=safety_block,
            child_module=module,
            env=env,
            env_factory=env_factory,
            timeout_s=timeout_s,
        )
    return host


@asynccontextmanager
async def _closed_if_it_never_starts(host: McpToolHost) -> "AsyncIterator[None]":
    """Close `host` if the start it is about to attempt does not finish --
    WR-02 (code review).

    `McpToolHost.start()` enters the transport's own context (spawning the
    stdio child, or opening the remote client and session) well before it
    returns: the child is running by the time `session.initialize()` and
    `list_tools()` are awaited. A start that is cancelled at
    `PluginManager`'s own startup deadline, or that raises anywhere after
    `_spawn` partially succeeded, therefore leaves a live child (or a live
    `httpx2.AsyncClient` plus an open session) attached to a host object
    that is never returned, never recorded in `PluginManager._plugins`,
    and so never reached by `_plugin_lifecycle`'s own `finally`. Nothing
    reaped it until the parent process exited -- one orphaned child per
    attempt, and every "Retry now" tap on a slow plugin added another.

    Whoever built the host closes it if it is not handed on. `BaseException`
    deliberately, not `Exception`: `CancelledError` is the deadline case,
    which is the one that actually happens. `aclose()` never raises
    (its own docstring), so the original failure is what propagates.
    """
    try:
        yield
    except BaseException:
        await host.aclose()
        raise
