"""Plan 09-08 Task 1: the child lists and searches Gmail by headers only,
hands every read to code as an `email_list`/`email_read` handoff, and
fetches a cleaned body only when code asks for one (`gmail_fetch_body`).

Every account, address, and name below is invented (`example.com`) -- no
real house appears in this file, the same convention every other test in
this suite follows.
"""

from __future__ import annotations

import json

from atlas_mcp.google import (
    handle_gmail_fetch_body,
    handle_gmail_list_unread,
    handle_gmail_read,
    handle_gmail_search,
)
from atlas_mcp.google_boundary import AccountGrant, parse_accounts_env
from atlas_mcp.safety import Denied

import pytest

from google_fakes import FakeGoogle

_ACCOUNTS_ENV = json.dumps(
    {
        "accounts": [
            {
                "label": "work",
                "email": "work@example.com",
                "is_default": True,
                "access_token": "at-work",
                "unreachable_reason": None,
                "calendars": [],
            },
            {
                "label": "home",
                "email": "home@example.com",
                "is_default": False,
                "access_token": "at-home",
                "unreachable_reason": None,
                "calendars": [],
            },
        ]
    }
)


def _accounts() -> "tuple[AccountGrant, ...]":
    return parse_accounts_env(_ACCOUNTS_ENV)


def _handoff(result: dict) -> dict:
    return result["atlas_handoff"]


# --- gmail_list_unread -------------------------------------------------


async def test_unread_query_is_exact():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", [])
    fake.add_gmail_messages("at-home", [])

    await handle_gmail_list_unread(accounts, fake.client)

    queries = {r.url.params.get("q") for r in fake.requests if r.url.path.endswith("/messages")}
    assert queries == {"in:inbox category:primary is:unread"}


async def test_every_request_is_a_get():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["m1"])
    fake.add_gmail_metadata("at-work", "m1", headers={"From": "Dana <dana@example.com>", "Subject": "Hi"})
    fake.add_gmail_messages("at-home", [])

    await handle_gmail_list_unread(accounts, fake.client)

    assert all(r.method == "GET" for r in fake.requests)


async def test_named_account_only_lists_that_account():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["m1"])
    fake.add_gmail_metadata("at-work", "m1", headers={"From": "Dana <dana@example.com>", "Subject": "Hi"})

    result = await handle_gmail_list_unread(accounts, fake.client, account="work")

    handoff = _handoff(result)
    assert handoff["kind"] == "email_list"
    assert handoff["source"] == "unread"
    assert [item["account"] for item in handoff["items"]] == ["work"]
    assert handoff["unreachable_accounts"] == []


async def test_items_newest_first():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["older", "newer"])
    fake.add_gmail_metadata(
        "at-work", "older", headers={"From": "Dana <dana@example.com>", "Subject": "Old"}, internal_date="1000"
    )
    fake.add_gmail_metadata(
        "at-work", "newer", headers={"From": "Lee <lee@example.com>", "Subject": "New"}, internal_date="2000"
    )
    fake.add_gmail_messages("at-home", [])

    result = await handle_gmail_list_unread(accounts, fake.client)

    items = _handoff(result)["items"]
    assert [i["message_id"] for i in items] == ["newer", "older"]


async def test_rfc2047_subject_and_from_are_decoded():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["m1"])
    fake.add_gmail_metadata(
        "at-work",
        "m1",
        headers={
            "From": "=?UTF-8?B?RMOhbmE=?= <dana@example.com>",
            "Subject": "=?UTF-8?B?SGVsbG8gdGjDqHJl?=",
        },
    )
    fake.add_gmail_messages("at-home", [])

    result = await handle_gmail_list_unread(accounts, fake.client)

    item = _handoff(result)["items"][0]
    assert item["from_name"] == "Dána"
    assert item["from_address"] == "dana@example.com"
    assert item["subject"] == "Hello thère"


async def test_from_header_with_no_display_name_keeps_the_address():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["m1"])
    fake.add_gmail_metadata("at-work", "m1", headers={"From": "dana@example.com", "Subject": "Hi"})
    fake.add_gmail_messages("at-home", [])

    result = await handle_gmail_list_unread(accounts, fake.client)

    item = _handoff(result)["items"][0]
    assert item["from_name"] == ""
    assert item["from_address"] == "dana@example.com"


async def test_more_than_max_ids_reports_has_more():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", [f"m{i}" for i in range(30)])
    for i in range(25):
        fake.add_gmail_metadata(
            "at-work", f"m{i}", headers={"From": "Dana <dana@example.com>", "Subject": "Hi"}
        )
    fake.add_gmail_messages("at-home", [])

    result = await handle_gmail_list_unread(accounts, fake.client)

    handoff = _handoff(result)
    assert len(handoff["items"]) == 25
    assert handoff["has_more"] == ["work"]


async def test_failing_account_is_reported_unreachable_others_still_answer():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.fail_gmail_list("at-work", status=401)
    fake.add_gmail_messages("at-home", ["m1"])
    fake.add_gmail_metadata("at-home", "m1", headers={"From": "Lee <lee@example.com>", "Subject": "Hi"})

    result = await handle_gmail_list_unread(accounts, fake.client)

    handoff = _handoff(result)
    assert [item["account"] for item in handoff["items"]] == ["home"]
    assert handoff["unreachable_accounts"] == [{"account": "work", "reason": "not authorized"}]


async def test_no_accounts_linked_refuses():
    fake = FakeGoogle()
    with pytest.raises(Denied):
        await handle_gmail_list_unread((), fake.client)


# --- gmail_search --------------------------------------------------------


async def test_search_sends_the_stripped_query_exactly():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_messages("at-work", ["m1"])
    fake.add_gmail_metadata("at-work", "m1", headers={"From": "Dana <dana@example.com>", "Subject": "Hi"})
    fake.add_gmail_messages("at-home", [])

    await handle_gmail_search(accounts, fake.client, query="  from:dana newer_than:7d  ")

    queries = {r.url.params.get("q") for r in fake.requests if r.url.path.endswith("/messages")}
    assert queries == {"from:dana newer_than:7d"}


async def test_search_refuses_empty_query():
    accounts = _accounts()
    fake = FakeGoogle()
    with pytest.raises(Denied):
        await handle_gmail_search(accounts, fake.client, query="   ")


async def test_search_refuses_a_query_over_two_hundred_characters():
    accounts = _accounts()
    fake = FakeGoogle()
    with pytest.raises(Denied):
        await handle_gmail_search(accounts, fake.client, query="x" * 201)


async def test_search_refuses_a_control_character():
    accounts = _accounts()
    fake = FakeGoogle()
    with pytest.raises(Denied):
        await handle_gmail_search(accounts, fake.client, query="from:dana\x00")


# --- gmail_read ------------------------------------------------------------


async def test_gmail_read_makes_no_request():
    fake = FakeGoogle()
    result = await handle_gmail_read(position=2)
    assert fake.requests == []
    handoff = _handoff(result)
    assert handoff["kind"] == "email_read"
    assert handoff["position"] == 2


async def test_gmail_read_refuses_both_position_and_sender():
    with pytest.raises(Denied):
        await handle_gmail_read(position=1, sender="dana")


async def test_gmail_read_refuses_neither_position_nor_sender():
    with pytest.raises(Denied):
        await handle_gmail_read()


async def test_gmail_read_refuses_position_out_of_range():
    with pytest.raises(Denied):
        await handle_gmail_read(position=51)
    with pytest.raises(Denied):
        await handle_gmail_read(position=0)


async def test_gmail_read_refuses_a_sender_over_sixty_characters():
    with pytest.raises(Denied):
        await handle_gmail_read(sender="x" * 61)


# --- gmail_fetch_body --------------------------------------------------


async def test_fetch_body_cleans_and_caps_the_text():
    accounts = _accounts()
    fake = FakeGoogle()
    body_text = "Hello there.\n\n> quoted history\n-- \nDana"
    fake.add_gmail_full(
        "at-work", "m1", headers={"From": "Dana <dana@example.com>", "Subject": "Hi"}, text=body_text
    )

    result = await handle_gmail_fetch_body(accounts, fake.client, account="work", message_id="m1")

    assert result["body"] == "Hello there."
    assert result["truncated"] is False
    assert result["from_name"] == "Dana"
    assert result["from_address"] == "dana@example.com"
    assert result["subject"] == "Hi"


async def test_fetch_body_with_no_token_is_denied():
    accounts = _accounts()
    unreachable = tuple(
        AccountGrant(
            label=a.label,
            email=a.email,
            is_default=a.is_default,
            access_token=None,
            unreachable_reason="needs_relink",
            calendars=a.calendars,
        )
        for a in accounts
    )
    fake = FakeGoogle()
    with pytest.raises(Denied):
        await handle_gmail_fetch_body(unreachable, fake.client, account="work", message_id="m1")


async def test_fetch_body_makes_only_get_requests():
    accounts = _accounts()
    fake = FakeGoogle()
    fake.add_gmail_full("at-work", "m1", headers={"From": "Dana <dana@example.com>"}, text="hi")

    await handle_gmail_fetch_body(accounts, fake.client, account="work", message_id="m1")

    assert all(r.method == "GET" for r in fake.requests)
