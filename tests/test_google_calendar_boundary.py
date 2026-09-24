"""09-01-PLAN.md Task 2 (D-05): the tool-boundary enforcement --
`handle_calendar_list_events` itself refuses to query a calendar whose
`access` is `"off"`, mirroring `ha.py`'s "the call site exists uniformly
here" doctrine -- never relying only on `GoogleEnvBuilder` having already
filtered it out upstream (defense in depth, the same posture
`allow_call`/`allow_read` take toward every Home Assistant handler).
"""

from __future__ import annotations

import json
from zoneinfo import ZoneInfo

from atlas_mcp.google import handle_calendar_list_events
from atlas_mcp.google_boundary import parse_accounts_env

from google_fakes import FakeGoogle

_ZONE = ZoneInfo("UTC")


def test_disabled_calendar_refused_at_handler():
    import asyncio

    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-read-only",
        [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
    )
    fake_google.add_events(
        "at-work",
        "cal-off",
        [{"id": "e2", "summary": "Should Never Appear", "start": {"dateTime": "2026-10-02T10:00:00Z"}, "end": {"dateTime": "2026-10-02T10:15:00Z"}}],
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
                        {
                            "calendar_id": "cal-read-only",
                            "name": "Team",
                            "primary": True,
                            "access": "read_only",
                        },
                        # A calendar the operator has not enabled -- a
                        # real env never includes this (GoogleEnvBuilder
                        # filters it), but the handler must refuse it on
                        # its own too, never trusting only the upstream
                        # filter.
                        {
                            "calendar_id": "cal-off",
                            "name": "Personal",
                            "primary": False,
                            "access": "off",
                        },
                    ],
                }
            ]
        }
    )
    accounts = parse_accounts_env(accounts_env)

    result = asyncio.run(
        handle_calendar_list_events(
            accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
        )
    )

    titles = {e["title"] for e in result["events"]}
    assert titles == {"Standup"}
    assert fake_google.requests_by_bearer("at-work") == [
        r for r in fake_google.requests_by_bearer("at-work") if "cal-read-only" in str(r.url)
    ]
    assert not any("cal-off" in str(r.url) for r in fake_google.requests)


def test_a_read_write_calendar_is_read_normally():
    import asyncio

    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-rw",
        [{"id": "e1", "summary": "Planning", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
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
                        {"calendar_id": "cal-rw", "name": "Team", "primary": True, "access": "read_write"}
                    ],
                }
            ]
        }
    )
    accounts = parse_accounts_env(accounts_env)

    result = asyncio.run(
        handle_calendar_list_events(
            accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
        )
    )

    assert {e["title"] for e in result["events"]} == {"Planning"}
