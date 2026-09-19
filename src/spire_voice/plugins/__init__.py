"""Plugins: one configuration-driven manager that spawns every MCP child
this process ever runs (D-01, D-04, PLUG-01, PLUG-09).

`spire_voice.mcp_client`'s own docstring named this phase directly:
"Building a configuration-driven router that spawns hosts of its own is
Phase 6's job, with Phase 6's information." `plugins.host` builds one
plugin's own environment and spawns it through `McpToolHost` (never a
second spawn path -- Pitfall 2); `plugins.manager` owns every running
plugin and the rebuildable tool schema/lookup the turn controller reads.
"""

from __future__ import annotations
