"""`PluginManager`: the one thing in this process that spawns a plugin
child (D-01, D-04, PLUG-01, PLUG-09) -- `lifespan` constructs this and
nothing else that starts an MCP host.

`start_all()` reads every enabled row, builds each child's environment
literally -- the declared config values (a secret one decrypted at this
single point, D-03), `PYTHONPATH` pointing at the repository's `mcp`
directory, and the serialized safety block for the row `enforces_policy`
names -- and starts it through `plugins.host.start_plugin_host` (never a
second spawn path, D-04, Pitfall 2). `rebuild()` builds a new tools list
and a new `McpToolHostLookup` and assigns both -- never mutating either in
place (D-08's own instruction: `run_turn` reads
`tools_schema`/`tool_host_lookup` exactly once per turn, at the start, so
an immutable swap needs no lock, generation counter, or readers-writer
gate at this level). The full failure-handling posture (non-blocking
startup, the watchdog, withdrawal) lands in a following plan; this
manager's `start_all()` may let a start failure propagate for now.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Awaitable, Callable, Sequence

from spire_voice.config import SecurityConfig
from spire_voice.crypto.credentials import decrypt_credential
from spire_voice.db.repository import Plugin, PluginConfigValue, PluginRepository
from spire_voice.mcp_client import McpToolHost, McpToolHostLookup, mcp_tools_to_openai_tools
from spire_voice.plugins.host import start_plugin_host

logger = logging.getLogger("spire_voice.plugins.manager")

# The current house policy, as the JSON-shaped dict `Policy.from_config`
# expects -- built by `lifespan` from `safety_block_from_policy(await
# policy_repo.load_policy())`, the same one parser on both sides of the
# process boundary this codebase has always used. Read fresh on every
# call, never cached, so a policy write that lands between two plugin
# starts (or a later respawn) is never served a stale block.
SafetyBlockProvider = Callable[[], Awaitable["dict | None"]]


@dataclass
class RunningPlugin:
    """One plugin row paired with the `McpToolHost` actually running it --
    the manager's own bookkeeping unit, never exposed past this module."""

    plugin: Plugin
    host: McpToolHost


class PluginManager:
    """Owns every running plugin child, and the rebuildable tool schema
    and lookup the turn controller reads.
    """

    def __init__(
        self,
        repository: PluginRepository,
        *,
        mcp_root: "os.PathLike[str] | str",
        security: SecurityConfig,
        safety_block_provider: SafetyBlockProvider,
    ) -> None:
        self._repository = repository
        self._mcp_root = mcp_root
        self._security = security
        self._safety_block_provider = safety_block_provider
        self._running: dict[int, RunningPlugin] = {}
        # Built empty at construction, and rebuilt by `rebuild()` below --
        # never `None`, so a reader that runs before `start_all()` (there
        # is none today, but a future one should not need a `None` guard)
        # sees an unambiguous "nothing running yet" lookup instead.
        self.tool_host_lookup: McpToolHostLookup = McpToolHostLookup([])
        self.tools_schema: list[dict[str, Any]] = []

    @property
    def hosts(self) -> list[McpToolHost]:
        """Every running plugin's own host, in start order -- the raw
        list a caller (`lifespan`) combines with a non-plugin host (the
        in-process workflow tool host) into one final
        `McpToolHostLookup`, since this manager only knows about plugin
        rows."""
        return [running.host for running in self._running.values()]

    @property
    def enforcing_host(self) -> McpToolHost | None:
        """The running host of the plugin that enforces the house policy
        (`app.state.tool_host`'s own definition, per this plan's wiring
        note) -- `None` when no such plugin is running. At most one
        plugin row has `enforces_policy=True` seeded by this plan's
        migration (Home Assistant); a future plan that lets an admin set
        this flag on more than one row would need to decide which wins,
        which this property does not attempt today."""
        for running in self._running.values():
            if running.plugin.enforces_policy:
                return running.host
        return None

    def tool_host_for(self, slug: str) -> McpToolHost | None:
        """The running host for `slug`, or `None` if it is not running
        (disabled, or failed to start) -- a lookup by the plugin's own
        stable identifier, for callers that need one specific plugin's
        host rather than the merged `tool_host_lookup`."""
        for running in self._running.values():
            if running.plugin.slug == slug:
                return running.host
        return None

    async def start_all(self) -> None:
        """Start every enabled plugin row, then rebuild the merged tool
        schema and lookup. A disabled row contributes no tools and spawns
        no child -- it is simply never passed to `_start_one`."""
        plugins = await self._repository.list_plugins()
        for plugin in plugins:
            if not plugin.enabled:
                continue
            await self._start_one(plugin)
        self.rebuild()

    async def _start_one(self, plugin: Plugin) -> None:
        safety_block = (
            await self._safety_block_provider() if plugin.enforces_policy else None
        )
        env = await self._build_env(plugin, safety_block=safety_block)
        env_factory = self._make_env_factory(plugin) if plugin.enforces_policy else None
        host = await start_plugin_host(
            plugin,
            mcp_root=self._mcp_root,
            safety_block=safety_block,
            env=env,
            env_factory=env_factory,
        )
        self._running[plugin.id] = RunningPlugin(plugin=plugin, host=host)

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
        shutdown block in place of the old per-host `aclose()` calls."""
        for running in self._running.values():
            await running.host.aclose()
        self._running.clear()


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
