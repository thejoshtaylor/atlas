"""09-01-PLAN.md Task 2: every linked account read at once, concurrently,
each item labelled with its own account, each account's own bearer token
used for its own requests.
"""

from __future__ import annotations

from zoneinfo import ZoneInfo

import pytest

from atlas_mcp.google import handle_calendar_list_events
from atlas_mcp.google_boundary import parse_accounts_env
from atlas_mcp.safety import Denied

from google_fakes import FakeGoogle

_ZONE = ZoneInfo("UTC")


def _accounts_env(*accounts: dict) -> str:
    import json

    return json.dumps({"accounts": list(accounts)})


def _account(label: str, access_token: "str | None", calendar_id: str = "cal-on") -> dict:
    return {
        "label": label,
        "email": f"{label}@example.com",
        "is_default": label == "work",
        "access_token": access_token,
        "unreachable_reason": None if access_token else "unreachable",
        "calendars": [
            {"calendar_id": calendar_id, "name": "Team Offsite", "primary": True, "access": "read_only"}
        ],
    }


async def test_unnamed_read_queries_every_account():
    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work", "cal-on", [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}]
    )
    fake_google.add_events(
        "at-home", "cal-on", [{"id": "e2", "summary": "Dentist", "start": {"dateTime": "2026-10-02T15:00:00Z"}, "end": {"dateTime": "2026-10-02T16:00:00Z"}}]
    )

    accounts = parse_accounts_env(
        _accounts_env(_account("work", "at-work"), _account("home", "at-home"))
    )

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
    )

    assert result["unreachable_accounts"] == []
    labels = {e["account"] for e in result["events"]}
    assert labels == {"work", "home"}
    assert {e["title"] for e in result["events"]} == {"Standup", "Dentist"}

    work_requests = fake_google.requests_by_bearer("at-work")
    home_requests = fake_google.requests_by_bearer("at-home")
    assert len(work_requests) == 1
    assert len(home_requests) == 1


async def test_a_named_account_case_insensitive_reads_only_that_one():
    fake_google = FakeGoogle()
    fake_google.add_events("at-work", "cal-on", [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}])
    fake_google.add_events("at-home", "cal-on", [{"id": "e2", "summary": "Dentist", "start": {"dateTime": "2026-10-02T15:00:00Z"}, "end": {"dateTime": "2026-10-02T16:00:00Z"}}])

    accounts = parse_accounts_env(_accounts_env(_account("work", "at-work"), _account("home", "at-home")))

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03", account="WORK"
    )

    assert {e["account"] for e in result["events"]} == {"work"}
    assert fake_google.requests_by_bearer("at-home") == []


async def test_an_account_with_no_access_token_is_named_unreachable_and_others_still_answer():
    fake_google = FakeGoogle()
    fake_google.add_events("at-work", "cal-on", [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}])

    accounts = parse_accounts_env(_accounts_env(_account("work", "at-work"), _account("broken", None)))

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
    )

    assert {e["account"] for e in result["events"]} == {"work"}
    assert result["unreachable_accounts"] == [{"account": "broken", "reason": "unreachable"}]


async def test_a_calendar_401_marks_the_account_not_authorized():
    fake_google = FakeGoogle()
    fake_google.fail_events("at-work", status=401)
    accounts = parse_accounts_env(_accounts_env(_account("work", "at-work")))

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
    )

    assert result["events"] == []
    assert result["unreachable_accounts"] == [{"account": "work", "reason": "not authorized"}]


async def test_a_transport_error_marks_the_account_not_reachable():
    fake_google = FakeGoogle()
    fake_google.fail_events("at-work", raise_connect_error=True)
    accounts = parse_accounts_env(_accounts_env(_account("work", "at-work")))

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
    )

    assert result["events"] == []
    assert result["unreachable_accounts"] == [{"account": "work", "reason": "not reachable"}]


async def test_a_range_over_31_days_is_refused_with_no_request():
    fake_google = FakeGoogle()
    accounts = parse_accounts_env(_accounts_env(_account("work", "at-work")))

    with pytest.raises(Denied):
        await handle_calendar_list_events(
            accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-12-02"
        )
    assert fake_google.requests == []


async def test_an_end_not_after_the_start_is_refused_with_no_request():
    fake_google = FakeGoogle()
    accounts = parse_accounts_env(_accounts_env(_account("work", "at-work")))

    with pytest.raises(Denied):
        await handle_calendar_list_events(
            accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-01"
        )
    assert fake_google.requests == []


async def test_a_naive_start_is_sent_as_wall_clock_in_the_childs_zone():
    zone = ZoneInfo("America/New_York")
    fake_google = FakeGoogle()
    fake_google.add_events("at-work", "cal-on", [])
    accounts = parse_accounts_env(_accounts_env(_account("work", "at-work")))

    await handle_calendar_list_events(
        accounts, fake_google.client, zone, start="2026-10-02T09:00", end="2026-10-02T10:00"
    )

    [request] = fake_google.requests_by_bearer("at-work")
    query = dict(request.url.params)
    assert query["timeZone"] == "America/New_York"
    # 09:00 America/New_York (EDT, UTC-4) is 13:00Z.
    assert query["timeMin"] == "2026-10-02T13:00:00Z"


async def test_no_linked_accounts_refuses_with_a_spoken_reason():
    fake_google = FakeGoogle()
    with pytest.raises(Denied) as excinfo:
        await handle_calendar_list_events(
            (), fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
        )
    assert "link one in the admin webapp" in excinfo.value.reason


async def test_a_failing_calendar_marks_the_account_unreachable_but_keeps_its_other_events():
    """A single account with two enabled calendars: one succeeds, one
    fails. The successful calendar's own events still contribute, and the
    account is still named unreachable so the operator knows one
    calendar's own results are missing."""
    fake_google = FakeGoogle()
    fake_google.add_events(
        "at-work",
        "cal-good",
        [{"id": "e1", "summary": "Standup", "start": {"dateTime": "2026-10-02T09:00:00Z"}, "end": {"dateTime": "2026-10-02T09:15:00Z"}}],
    )
    fake_google.fail_events("at-work", "cal-bad", status=500)

    accounts_env = _accounts_env(
        {
            "label": "work",
            "email": "work@example.com",
            "is_default": True,
            "access_token": "at-work",
            "unreachable_reason": None,
            "calendars": [
                {"calendar_id": "cal-good", "name": "Team", "primary": True, "access": "read_only"},
                {"calendar_id": "cal-bad", "name": "Broken", "primary": False, "access": "read_only"},
            ],
        }
    )
    accounts = parse_accounts_env(accounts_env)

    result = await handle_calendar_list_events(
        accounts, fake_google.client, _ZONE, start="2026-10-02", end="2026-10-03"
    )

    assert {e["title"] for e in result["events"]} == {"Standup"}
    assert result["unreachable_accounts"] == [{"account": "work", "reason": "not reachable"}]
