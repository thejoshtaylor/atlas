"""Finding the google plugin row, and keeping its own child's access
tokens fresh through the plugin manager's own respawn path -- never a
second spawn mechanism (D-04, `plugins/host.py`'s own doctrine).

Plan 09-03 adds `ensure_google_plugin` (create the row the first time an
account is ever linked) and `reconcile_google_plugin` (the one entry point
every write route in `routes/google_accounts.py` ends with) beside plan
09-01's read-only `find_google_plugin`/`refresh_google_plugin`.
"""

from __future__ import annotations

import logging

from atlas_mcp.google_tools import GOOGLE_PLUGIN_MODULE

from atlas.db.repository import Plugin, PluginAlreadyExistsError, PluginRepository
from atlas.plugins.host import module_for_stdio_args
from atlas.plugins.manager import PluginManager, PluginState

logger = logging.getLogger("atlas.google.plugin")

# 09-03-PLAN.md's own action text: the plugin's per-call deadline, fixed
# at creation -- an admin can still edit it later through the ordinary
# plugin-configuration screen, the same as any other plugin.
_GOOGLE_PLUGIN_TIMEOUT_MS = 15000


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


async def ensure_google_plugin(plugin_repo: PluginRepository, *, created_by_user_id: "int | None") -> Plugin:
    """The google plugin row, creating it the first time any account is
    ever linked -- found by `find_google_plugin` first (never by slug), so
    a deployment that already has one keeps it untouched. `slug="google"`
    is tried first; a `PluginAlreadyExistsError` (an admin already used
    that slug for an unrelated plugin, however unlikely) retries once with
    `slug="google-accounts"` rather than failing the very first link."""
    existing = await find_google_plugin(plugin_repo)
    if existing is not None:
        return existing
    try:
        return await plugin_repo.create_plugin(
            slug="google",
            display_name="Google",
            transport="stdio",
            args=("-m", GOOGLE_PLUGIN_MODULE),
            url=None,
            timeout_ms=_GOOGLE_PLUGIN_TIMEOUT_MS,
            config_values=(),
            created_by_user_id=created_by_user_id,
        )
    except PluginAlreadyExistsError:
        return await plugin_repo.create_plugin(
            slug="google-accounts",
            display_name="Google",
            transport="stdio",
            args=("-m", GOOGLE_PLUGIN_MODULE),
            url=None,
            timeout_ms=_GOOGLE_PLUGIN_TIMEOUT_MS,
            config_values=(),
            created_by_user_id=created_by_user_id,
        )


async def reconcile_google_plugin(
    plugin_repo: PluginRepository,
    plugin_manager: PluginManager,
    *,
    created_by_user_id: "int | None",
    narrowing: bool = False,
) -> None:
    """The one entry point every write route in `routes/google_accounts.py`
    ends with (09-03-PLAN.md's own action text): ensure the plugin row
    exists, then reach the running child -- `request_respawn` when it is
    already `RUNNING`, `start_one` otherwise (covers "not yet started" and
    "degraded," where a fresh start might recover it, the same posture
    `refresh_google_plugin` above already takes).

    `narrowing=True` names a write whose new state grants *less* access
    than before (a calendar access downgrade, an unlink, clearing a
    default) -- when the reconcile itself fails, the child is stopped
    before the exception is re-raised, so it is never left running with
    wider access than the operator just chose (D-05, T-09-14). A widening
    write's reconcile failure is re-raised with nothing stopped: the
    child keeps whatever access it already had, which is never wider than
    what this write was trying to grant anyway.
    """
    plugin = await ensure_google_plugin(plugin_repo, created_by_user_id=created_by_user_id)
    try:
        if plugin_manager.state_for(plugin.slug) is PluginState.RUNNING:
            await plugin_manager.request_respawn(plugin.id, None)
        else:
            await plugin_manager.start_one(plugin)
    except Exception:
        if narrowing:
            await plugin_manager.stop_one(plugin)
        raise
