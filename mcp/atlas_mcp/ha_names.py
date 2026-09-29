"""Names of the Home Assistant tools, kept apart from `atlas_mcp.ha`.

`atlas_mcp.ha` builds a tool server and reads environment on import. The
Atlas server process must know these names without that. So this module has
no import side effects and imports nothing.
"""

from __future__ import annotations

HA_PLUGIN_MODULE = "atlas_mcp.ha"

# Read-only tool for code. It turns an area, device, or label into entity ids.
# A later plan hides it from the model.
HA_EXPAND_TARGET_TOOL = "ha_expand_target"

# The only tools that change a device, so the only tools that take an entity claim.
HA_WRITE_TOOL_NAMES: frozenset[str] = frozenset({"ha_call_service", "ha_play_spotify_playlist"})

HA_CODE_ONLY_TOOL_NAMES: frozenset[str] = frozenset({HA_EXPAND_TARGET_TOOL})
