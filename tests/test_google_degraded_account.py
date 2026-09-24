"""GOOG-12: an account whose token could not be refreshed, or whose
calendar call fails, is named in the result as unreachable -- never
silently left out, never a crash.

09-01-PLAN.md Task 2 adds `test_revoked_account_is_named_by_the_child`
(child-level, no network call for the revoked account itself). Plan 09-05
adds the turn-level `test_revoked_account_named_not_crashed` to this same
file later.
"""

from __future__ import annotations

import json
from zoneinfo import ZoneInfo

from atlas_mcp.google import handle_calendar_list_events
from atlas_mcp.google_boundary import parse_accounts_env

from google_fakes import FakeGoogle

_ZONE = ZoneInfo("UTC")


async def test_revoked_account_is_named_by_the_child():
    """An account the token service could not refresh (`access_token`
    `None`, `unreachable_reason="needs_relink"`) is named in
    `unreachable_accounts` with that exact reason, and no request is ever
    made for it -- while a healthy account alongside it still answers."""
    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-on",
        [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
    )

    accounts_env = json.dumps(
        {
            "accounts": [
                {
                    "label": "work",
                    "email": "work@example.com",
                    "is_default": True,
                    "access_token": "at-work",
                    "unreachable_reason": None,
                    "calendars": [
                        {"calendar_id": "cal-on", "name": "Team", "primary": True, "access": "read_only"}
                    ],
                },
                {
                    "label": "revoked",
                    "email": "revoked@example.com",
                    "is_default": False,
                    "access_token": None,
                    "unreachable_reason": "needs_relink",
                    "calendars": [
                        {"calendar_id": "cal-x", "name": "Personal", "primary": True, "access": "read_only"}
                    ],
                },
            ]
        }
    )
    accounts = parse_accounts_env(accounts_env)

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
    )

    assert {e["account"] for e in result["events"]} == {"work"}
    assert result["unreachable_accounts"] == [{"account": "revoked", "reason": "needs_relink"}]
    assert fake_google.requests == fake_google.requests_by_bearer("at-work")
