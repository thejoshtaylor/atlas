"""Finding the google plugin row, and keeping its own child's access
tokens fresh through the plugin manager's own respawn path -- never a
second spawn mechanism (D-04, `plugins/host.py`'s own doctrine).
"""

from __future__ import annotations

import logging

from atlas_mcp.google_tools import GOOGLE_PLUGIN_MODULE

from atlas.db.repository import Plugin, PluginRepository
from atlas.plugins.host import module_for_stdio_args
from atlas.plugins.manager import PluginManager, PluginState

logger = logging.getLogger("atlas.google.plugin")


async def find_google_plugin(plugin_repo: PluginRepository) -> "Plugin | None":
    """The plugin row whose stdio args name `GOOGLE_PLUGIN_MODULE`, found
    by the module it runs -- never by slug (an admin can rename a
    plugin's display name, but not the module its args point at). `None`
    when no such row exists (this deployment never installed the Google
    plugin) -- a real, tolerated state, not an error."""
    for plugin in await plugin_repo.list_plugins():
        if plugin.transport != "stdio":
            continue
        try:
            module = module_for_stdio_args(plugin.slug, plugin.args)
        except RuntimeError:
            continue
        if module == GOOGLE_PLUGIN_MODULE:
            return plugin
    return None


async def refresh_google_plugin(plugin_manager: PluginManager, plugin_repo: PluginRepository) -> None:
    """Give the running google plugin's child a freshly-built environment
    (new access tokens, GOOG-12's current per-account reasons) -- called
    on `GoogleTokenRefreshScheduler`'s own interval.

    A `RUNNING` plugin is respawned through `request_respawn`, the one
    entry point that rebuilds an already-running plugin's environment
    without a second spawn path (`PluginManager.request_respawn`'s own
    docstring). An enabled plugin that is not currently running (starting,
    degraded, crashed-and-retrying) is started fresh through `start_one`
    instead -- `request_respawn` raises `RuntimeError` for a plugin that
    is not `RUNNING`, and a scheduled refresh finding a degraded plugin is
    exactly the moment a fresh start might recover it. Nothing happens
    when no google plugin row exists, or the row is disabled -- a missed
    refresh must not take anything else down, and there is nothing to
    refresh either way.
    """
    plugin = await find_google_plugin(plugin_repo)
    if plugin is None or not plugin.enabled:
        return
    if plugin_manager.state_for(plugin.slug) is PluginState.RUNNING:
        try:
            await plugin_manager.request_respawn(plugin.id, None)
        except RuntimeError as exc:
            logger.warning("google plugin respawn for a token refresh did not land: %s", exc)
        return
    await plugin_manager.start_one(plugin)
