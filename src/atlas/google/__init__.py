"""Phase 9's backend package: everything that runs in this process (never
the `atlas_mcp.google` child) and touches a refresh token, an OAuth
client secret, or the database -- `token_service.py` (decrypt/refresh/
cache), `env.py` (the child's own env builder), `plugin.py` (finding and
respawning the google plugin row), and `scheduler.py` (the periodic
token-refresh loop).
"""

from __future__ import annotations
