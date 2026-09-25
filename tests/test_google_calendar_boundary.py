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

import pytest

from atlas_mcp.google import handle_calendar_delete_event, handle_calendar_insert_event, handle_calendar_list_events
from atlas_mcp.google_boundary import AccountGrant, CalendarGrant, parse_accounts_env, require_writable
from atlas_mcp.safety import Denied

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


# --- Task 1: require_writable / handle_calendar_insert_event / handle_calendar_delete_event


def _write_account(*, calendars=None, access_token: str | None = "at-work") -> AccountGrant:
    return AccountGrant(
        label="work",
        email="work@example.com",
        is_default=True,
        access_token=access_token,
        unreachable_reason=None if access_token is not None else "needs_relink",
        calendars=calendars
        if calendars is not None
        else (CalendarGrant(calendar_id="cal-rw", name="Team", primary=True, access="read_write"),),
    )


def test_require_writable_denies_an_unknown_account():
    account = _write_account()
    with pytest.raises(Denied):
        require_writable((account,), "nonexistent", "cal-rw")


def test_require_writable_denies_an_account_with_no_access_token():
    account = _write_account(access_token=None)
    with pytest.raises(Denied):
        require_writable((account,), "work", "cal-rw")


def test_require_writable_denies_a_calendar_not_in_the_env():
    account = _write_account()
    with pytest.raises(Denied):
        require_writable((account,), "work", "cal-nonexistent")


def test_require_writable_denies_a_read_only_calendar():
    account = _write_account(
        calendars=(CalendarGrant(calendar_id="cal-ro", name="Team", primary=True, access="read_only"),)
    )
    with pytest.raises(Denied):
        require_writable((account,), "work", "cal-ro")


def test_require_writable_returns_the_account_and_calendar_for_a_read_write_calendar():
    account = _write_account()
    resolved_account, resolved_calendar = require_writable((account,), "work", "cal-rw")
    assert resolved_account is account
    assert resolved_calendar.calendar_id == "cal-rw"


async def test_insert_into_a_read_only_calendar_makes_zero_http_requests():
    fake_google = FakeGoogle()
    account = _write_account(
        calendars=(CalendarGrant(calendar_id="cal-ro", name="Team", primary=True, access="read_only"),)
    )
    with pytest.raises(Denied):
        await handle_calendar_insert_event(
            (account,),
            fake_google.client,
            account="work",
            calendar_id="cal-ro",
            title="Standup",
            start="2026-10-02T09:00:00Z",
            end="2026-10-02T09:15:00Z",
            all_day=False,
            time_zone="UTC",
        )
    assert fake_google.requests == []


async def test_insert_event_posts_one_request_and_returns_the_created_event():
    fake_google = FakeGoogle()
    account = _write_account()

    result = await handle_calendar_insert_event(
        (account,),
        fake_google.client,
        account="work",
        calendar_id="cal-rw",
        title="Standup",
        start="2026-10-02T09:00:00Z",
        end="2026-10-02T09:15:00Z",
        all_day=False,
        time_zone="UTC",
    )

    assert result["created"]["account"] == "work"
    assert result["created"]["calendar"] == "Team"
    assert result["created"]["event_id"] is not None
    assert len(fake_google.inserted) == 1
    assert fake_google.inserted[0]["calendar_id"] == "cal-rw"
    assert fake_google.inserted[0]["body"]["summary"] == "Standup"


async def test_insert_event_encodes_a_calendar_id_containing_at_and_hash():
    fake_google = FakeGoogle()
    account = _write_account(
        calendars=(
            CalendarGrant(calendar_id="a@b.com#c", name="Team", primary=True, access="read_write"),
        )
    )

    await handle_calendar_insert_event(
        (account,),
        fake_google.client,
        account="work",
        calendar_id="a@b.com#c",
        title="Standup",
        start="2026-10-02T09:00:00Z",
        end="2026-10-02T09:15:00Z",
        all_day=False,
        time_zone="UTC",
    )

    assert fake_google.inserted[0]["calendar_id"] == "a@b.com#c"


async def test_delete_event_removes_it_and_records_the_id():
    fake_google = FakeGoogle()
    fake_google.add_event("at-work", "cal-rw", {"id": "evt-1", "summary": "Standup"})
    account = _write_account()

    result = await handle_calendar_delete_event(
        (account,), fake_google.client, account="work", calendar_id="cal-rw", event_id="evt-1"
    )

    assert result["deleted"]["event_id"] == "evt-1"
    assert fake_google.deleted_event_ids == ["evt-1"]


async def test_delete_event_already_gone_is_reported_not_as_success():
    fake_google = FakeGoogle()
    account = _write_account()

    with pytest.raises(Denied):
        await handle_calendar_delete_event(
            (account,), fake_google.client, account="work", calendar_id="cal-rw", event_id="evt-missing"
        )
    assert fake_google.deleted_event_ids == []


async def test_delete_from_a_read_only_calendar_makes_zero_http_requests():
    fake_google = FakeGoogle()
    account = _write_account(
        calendars=(CalendarGrant(calendar_id="cal-ro", name="Team", primary=True, access="read_only"),)
    )
    with pytest.raises(Denied):
        await handle_calendar_delete_event(
            (account,), fake_google.client, account="work", calendar_id="cal-ro", event_id="evt-1"
        )
    assert fake_google.requests == []
