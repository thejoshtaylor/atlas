"""Per-plugin subprocess host: the thin wrapper that actually spawns a
plugin through `McpToolHost` -- never a second spawn path beside it (D-01
.. D-04, Pitfall 2 in 06-RESEARCH.md). Building a plugin's own literal
environment (the config-value read, the secret decryption) is
`PluginManager`'s job (`spire_voice.plugins.manager`), since that is the
one place holding the repository and the security config both -- this
module only turns an already-built environment (or an `env_factory` that
builds one) into a real spawned child.

Plan 06-02 (D-06, PLUG-06): this is also the one place `plugin.timeout_ms`
turns into `McpToolHost`'s own `timeout_s` -- read from the plugin's own
row, in milliseconds, so a local Home Assistant call and a remote weather
API never share one deadline (D-06's own rejection of a single global
timeout). `McpToolHost.call_tool` passes this value straight through to
the SDK's own `read_timeout_seconds` parameter on every call this plugin's
host makes -- no wrapper, no second place a deadline could drift from
this one.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Mapping, Sequence

from spire_voice.db.repository import Plugin
from spire_voice.mcp_client import McpToolHost

if TYPE_CHECKING:
    from spire_voice.mcp_client import EnvFactory


def module_for_stdio_args(slug: str, args: Sequence[str]) -> str:
    """The stdio child's importable module name, from a plugin row's own
    `args` (`["-m", "spire_mcp.ha"]`) -- the same `-m <module>` shape
    `McpServerConfig`/`McpToolHost._spawn` already require (D-02: the
    interpreter is `sys.executable`, never configurable, so `args` never
    carries anything else)."""
    if len(args) != 2 or args[0] != "-m":
        raise RuntimeError(
            f"plugin {slug!r} has stdio args {list(args)!r} -- expected exactly "
            "['-m', '<module>'], the only shape a stdio plugin child's args take"
        )
    return args[1]


async def start_plugin_host(
    plugin: Plugin,
    *,
    mcp_root: "os.PathLike[str] | str",
    safety_block: dict | None,
    env: "Mapping[str, str] | None" = None,
    env_factory: "EnvFactory | None" = None,
) -> McpToolHost:
    """Spawn `plugin`'s stdio child through `McpToolHost` -- never a
    second spawn path (D-04, Pitfall 2). Exactly one of `env`/`env_factory`
    is meaningful per call: `env_factory`, when given, takes precedence in
    `McpToolHost.start()` itself (see that method's own docstring), and is
    threaded through so a later `respawn()` rebuilds this same plugin's
    environment fresh rather than repeating the mapping this call built.
    """
    module = module_for_stdio_args(plugin.slug, plugin.args)
    host = McpToolHost()
    await host.start(
        ha_url="",
        ha_token="",
        mcp_root=mcp_root,
        safety_block=safety_block,
        child_module=module,
        env=env,
        env_factory=env_factory,
        timeout_s=plugin.timeout_ms / 1000.0,
    )
    return host
