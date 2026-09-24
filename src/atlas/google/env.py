"""`GoogleEnvBuilder`: the `CustomEnvBuilder` (`src/atlas/plugins/manager.py`)
that turns every linked account into the `atlas_mcp.google` child's one
environment key, `GOOGLE_ACCOUNTS_JSON` (T-09-01) -- an access token only,
never a refresh token or the OAuth client secret, and only each account's
enabled calendars (D-05: an off calendar never reaches the child at all).
"""

from __future__ import annotations

import json
from typing import Any

from atlas_mcp.google_tools import GOOGLE_ACCOUNTS_ENV

from atlas.db.google_repository import GoogleAccountRepository
from atlas.google.token_service import GoogleTokenService


class GoogleEnvBuilder:
    """Reads every linked account and builds the one-key environment the
    google plugin's child receives -- registered against
    `PluginManager(custom_env_builders={GOOGLE_PLUGIN_MODULE: ...})`
    (`src/atlas/plugins/manager.py`), called fresh on every start and
    every respawn, the same "a fresh read of this plugin's own config
    values on every call" contract `PluginManager._make_env_factory`
    already documents for the generic env path.
    """

    def __init__(self, repo: GoogleAccountRepository, token_service: GoogleTokenService) -> None:
        self._repo = repo
        self._token_service = token_service

    async def __call__(self, plugin: Any) -> dict[str, str]:
        accounts = await self._repo.list_accounts()
        payload: list[dict[str, Any]] = []
        for account in accounts:
            access_token = await self._token_service.access_token_for(account)
            payload.append(
                {
                    "label": account.label,
                    "email": account.email,
                    "is_default": account.is_default,
                    "access_token": access_token.token,
                    "unreachable_reason": access_token.unreachable_reason,
                    "calendars": [
                        {
                            "calendar_id": calendar.google_calendar_id,
                            "name": calendar.name,
                            "primary": calendar.is_primary,
                            "access": calendar.access,
                        }
                        for calendar in account.calendars
                        if calendar.access != "off"
                    ],
                }
            )
        return {GOOGLE_ACCOUNTS_ENV: json.dumps({"accounts": payload})}
