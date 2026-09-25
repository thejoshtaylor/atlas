"""The stdio MCP tool server for Google Calendar and Gmail, across every
linked account -- follows `mcp/atlas_mcp/ha.py`'s own shape exactly
(module-level `MCPServer`, `handle_*` functions taking explicit
parameters, `@mcp_server.tool()` wrappers converting `Denied` to
`ToolError`, `_startup()`/`_run()`), the one rule stated there applying
here unchanged: every handler, reads included, calls through this
module's own boundary (`atlas_mcp.google_boundary.resolve_accounts`, plus
each account's own per-calendar `access`, D-05) before touching `httpx`.

This process never sees a refresh token or the OAuth client secret
(T-09-01) -- `_startup()` reads `GOOGLE_ACCOUNTS_JSON`
(`atlas_mcp.google_tools.GOOGLE_ACCOUNTS_ENV`) into module state built
entirely from `google_boundary.parse_accounts_env`, and nothing here ever
reaches for the database or the plugin manager's own environment builder.
"""

from __future__ import annotations

import asyncio
import os
import re
from datetime import date, datetime, timedelta, timezone as _timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from atlas_mcp.google_api import GoogleApiError, GoogleAuthError, delete_event, get_event, insert_event, list_events
from atlas_mcp.google_boundary import (
    AccountGrant,
    Clarification,
    parse_accounts_env,
    require_writable,
    resolve_accounts,
    resolve_write_target,
)
from atlas_mcp.google_gmail_api import get_message_full, get_message_metadata, header_value, list_message_ids, parse_from
from atlas_mcp.google_tools import GOOGLE_ACCOUNTS_ENV, HANDOFF_KEY
from atlas_mcp.mail_clean import MODEL_INPUT_CAP, cap_text, clean_body, extract_text
from atlas_mcp.safety import Denied

_MAX_RANGE_DAYS = 31
_MAX_TITLE_LEN = 120

# Task 1 (D-14, D-15): the fixed query "any new email?" always sends --
# unread mail in the Primary inbox category only. Promotions/Social/
# Updates are excluded on purpose (09-CONTEXT.md D-14): a mailbox with
# hundreds of promotional unread messages must never drown out the two
# real ones.
_GMAIL_UNREAD_QUERY = "in:inbox category:primary is:unread"
_MAX_MESSAGE_IDS = 25
_METADATA_CONCURRENCY = 8
_MAX_SEARCH_QUERY_LEN = 200
_MAX_READ_POSITION = 50
_MAX_SENDER_LEN = 60
_CONTROL_CHAR_RE = re.compile(r"[\x00-\x1f\x7f]")


def _parse_boundary(value: str, zone: ZoneInfo) -> datetime:
    """Parse a start/end boundary as an ISO date or datetime. A naive
    value (no offset, and no "T" at all for a bare date) is local to
    `zone` -- an operator says "Friday at 3pm" in house time, never UTC.
    """
    try:
        if "T" in value:
            parsed = datetime.fromisoformat(value)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=zone)
            return parsed
        parsed_date = date.fromisoformat(value)
        return datetime(parsed_date.year, parsed_date.month, parsed_date.day, tzinfo=zone)
    except ValueError as exc:
        raise Denied(f"i couldn't understand the date or time {value!r}") from exc


def _reason_for(exc: Exception) -> str:
    """The GOOG-12 per-account degrade reason for a failed calendar call
    -- an auth failure (expired/revoked access token) reads distinctly
    from any other transport or API failure, the same "distinguishable,
    spoken reason" discipline `ha.py::_resolve_target` establishes for an
    unresolved target."""
    if isinstance(exc, GoogleAuthError):
        return "not authorized"
    return "not reachable"


async def _events_for_calendar(
    account: AccountGrant,
    calendar: "Any",
    client: httpx.AsyncClient,
    zone: ZoneInfo,
    *,
    time_min_s: str,
    time_max_s: str,
    query: "str | None",
) -> list[dict[str, Any]]:
    """Every event one calendar carries, formatted with its own account
    and calendar labels -- one `asyncio.gather` job (D-04's own fan-out
    unit is a calendar, not an account, so one account's second calendar
    can succeed even while its first one fails). A `GoogleAuthError`/
    `GoogleApiError`/`httpx.HTTPError` here propagates to the caller's
    `gather(..., return_exceptions=True)` -- never caught in this
    function, so the caller's own per-account bookkeeping is the one
    place a failure is turned into a reason.
    """
    raw_events = await list_events(
        client,
        access_token=account.access_token,
        calendar_id=calendar.calendar_id,
        time_min=time_min_s,
        time_max=time_max_s,
        time_zone=str(zone),
        query=query,
    )
    return [
        _format_event(raw_event, account=account.label, calendar=calendar.name)
        for raw_event in raw_events
    ]


async def _fan_out_events(
    resolved: "tuple[AccountGrant, ...]",
    client: httpx.AsyncClient,
    zone: ZoneInfo,
    *,
    time_min: datetime,
    time_max: datetime,
    query: "str | None",
) -> tuple[list[dict[str, Any]], list[dict[str, str]]]:
    """Every event across every resolved account, and the
    `{"account", "reason"}` entry for every account this call could not
    fully answer.

    `asyncio.gather(..., return_exceptions=True)` -- never the structured-
    concurrency construct whose first failure would cancel every other
    account and calendar still in flight (the same reasoning
    `turn/brain_race.py`'s own module docstring gives, and Task 2's own
    action text names directly) -- fans out one job per (account,
    calendar) pair, so one calendar failing never loses another
    calendar's already-succeeding events for the SAME account (D-04's
    "that account's successful calendars still contribute events").

    An account with no `access_token` (GOOG-12: `GoogleTokenService`
    could not refresh it) contributes no job at all -- no request is ever
    made for it -- and is reported unreachable with its own
    `unreachable_reason` directly.
    """
    time_min_s = time_min.astimezone(_timezone.utc).isoformat().replace("+00:00", "Z")
    time_max_s = time_max.astimezone(_timezone.utc).isoformat().replace("+00:00", "Z")

    unreachable_by_account: dict[str, str] = {}
    jobs: list[tuple[AccountGrant, Any]] = []
    for one_account in resolved:
        if one_account.access_token is None:
            unreachable_by_account[one_account.label] = one_account.unreachable_reason or "not reachable"
            continue
        for calendar in one_account.calendars:
            # D-05, defense in depth: the boundary enforces this itself,
            # never trusting only `GoogleEnvBuilder`'s own upstream
            # filter (`ha.py`'s own "the call site exists uniformly here"
            # doctrine).
            if calendar.access == "off":
                continue
            jobs.append((one_account, calendar))

    results = await asyncio.gather(
        *(
            _events_for_calendar(
                one_account, calendar, client, zone, time_min_s=time_min_s, time_max_s=time_max_s, query=query
            )
            for one_account, calendar in jobs
        ),
        return_exceptions=True,
    )

    all_events: list[dict[str, Any]] = []
    for (one_account, _calendar), result in zip(jobs, results):
        if isinstance(result, Exception):
            if one_account.label not in unreachable_by_account:
                unreachable_by_account[one_account.label] = _reason_for(result)
            continue
        all_events.extend(result)

    unreachable = [
        {"account": label, "reason": reason} for label, reason in sorted(unreachable_by_account.items())
    ]
    return all_events, unreachable


def _format_event(raw_event: dict[str, Any], *, account: str, calendar: str) -> dict[str, Any]:
    start = raw_event.get("start", {})
    end = raw_event.get("end", {})
    all_day = "date" in start
    return {
        "account": account,
        "calendar": calendar,
        "title": raw_event.get("summary", "(no title)"),
        "start": start.get("date") or start.get("dateTime"),
        "end": end.get("date") or end.get("dateTime"),
        "all_day": all_day,
        "location": raw_event.get("location"),
        "event_id": raw_event.get("id"),
        "calendar_id": raw_event.get("organizer", {}).get("email"),
        "recurring": bool(raw_event.get("recurringEventId")),
    }


async def handle_calendar_list_events(
    accounts: "tuple[AccountGrant, ...]",
    client: httpx.AsyncClient,
    zone: ZoneInfo,
    *,
    start: str,
    end: str,
    account: "str | None" = None,
    query: "str | None" = None,
) -> dict[str, Any]:
    """Answer a calendar-events question, gated by `resolve_accounts`
    (D-04) and this account's own per-calendar `access` (D-05, enforced
    both upstream by `GoogleEnvBuilder` -- an off calendar is normally
    never in `account.calendars` at all -- and again by `_fan_out_events`
    itself, never trusting only the upstream filter).

    Refuses (never queries) a range longer than `_MAX_RANGE_DAYS` days or
    an end not after the start, and refuses when no account is linked at
    all -- both name a spoken reason, neither raises a bare exception.
    """
    if not accounts:
        raise Denied("no google account is linked -- link one in the admin webapp under google accounts")
    resolved = resolve_accounts(accounts, account)

    time_min = _parse_boundary(start, zone)
    time_max = _parse_boundary(end, zone)
    if time_max <= time_min:
        raise Denied("the end of that range must be after the start")
    if time_max - time_min > timedelta(days=_MAX_RANGE_DAYS):
        raise Denied(f"i can only look across {_MAX_RANGE_DAYS} days at a time")

    all_events, unreachable = await _fan_out_events(
        resolved, client, zone, time_min=time_min, time_max=time_max, query=query
    )

    all_events.sort(key=lambda e: e["start"] or "")
    return {
        "time_zone": str(zone),
        "events": all_events,
        "unreachable_accounts": unreachable,
    }


def _sanitize_title(title: str) -> str:
    """A title with newlines flattened to spaces, stripped, and capped at
    `_MAX_TITLE_LEN` characters -- what a proposal's own title ends up as,
    verbatim, in the readback and in the stored pending action (D-07)."""
    cleaned = title.replace("\r\n", " ").replace("\n", " ").replace("\r", " ").strip()
    return cleaned[:_MAX_TITLE_LEN]


async def handle_calendar_propose_event(
    accounts: "tuple[AccountGrant, ...]",
    zone: ZoneInfo,
    *,
    title: str,
    start: str,
    end: "str | None" = None,
    duration_minutes: "int | None" = None,
    all_day: bool = False,
    account: "str | None" = None,
    calendar: "str | None" = None,
) -> dict[str, Any]:
    """Build a `pending_action` (or `needs_clarification`) handoff payload
    for adding one calendar event -- makes no HTTP call, ever (D-08). The
    event is only ever added by the executing tool this handoff's own
    `atlas.turn.pending_action.EXECUTING_TOOL_BY_ACTION` names, run later by
    code holding the exact arguments this call resolved, never by this
    function or by the model that called it.

    `resolve_write_target` (D-04, D-05) is the one gate: a calendar that is
    off or read-only, or an account this env does not carry, raises `Denied`
    with a spoken reason before any payload is ever built. An ambiguous
    target (no account named with more than one candidate, or a named
    account with more than one writable calendar) returns a
    `needs_clarification` handoff instead of a proposal.

    `duration_minutes`/`end` default to a 60-minute event when neither is
    given. `all_day` takes `start` as a bare date (no time) and reports one
    calendar day, end exclusive. `title` is sanitized by `_sanitize_title`.
    Every time in the returned payload is ISO 8601 with an offset in `zone`.
    """
    if not accounts:
        raise Denied("no google account is linked -- link one in the admin webapp under google accounts")

    target = resolve_write_target(accounts, account, calendar)
    if isinstance(target, Clarification):
        return {
            HANDOFF_KEY: {
                "kind": "needs_clarification",
                "about": target.about,
                "candidates": list(target.candidates),
            }
        }
    account_grant, calendar_grant = target

    clean_title = _sanitize_title(title)
    if not clean_title:
        raise Denied("i need a title for that event")

    if all_day:
        start_date = date.fromisoformat(start)
        end_date = start_date + timedelta(days=1)
        start_out = start_date.isoformat()
        end_out = end_date.isoformat()
    else:
        start_dt = _parse_boundary(start, zone)
        if end is not None:
            end_dt = _parse_boundary(end, zone)
        elif duration_minutes is not None:
            end_dt = start_dt + timedelta(minutes=duration_minutes)
        else:
            end_dt = start_dt + timedelta(minutes=60)
        if end_dt <= start_dt:
            raise Denied("the end of that event must be after the start")
        start_out = start_dt.isoformat()
        end_out = end_dt.isoformat()

    return {
        HANDOFF_KEY: {
            "kind": "pending_action",
            "action": "calendar_create",
            "account": account_grant.label,
            "calendar_id": calendar_grant.calendar_id,
            "calendar_name": calendar_grant.name,
            "calendar_primary": calendar_grant.primary,
            "title": clean_title,
            "start": start_out,
            "end": end_out,
            "all_day": bool(all_day),
            "time_zone": str(zone),
        }
    }


async def handle_calendar_propose_delete(
    accounts: "tuple[AccountGrant, ...]",
    client: httpx.AsyncClient,
    zone: ZoneInfo,
    *,
    account: str,
    calendar_id: str,
    event_id: str,
) -> dict[str, Any]:
    """Build a `pending_action` (`calendar_delete`) handoff for deleting
    one calendar event -- fetches the event once, to read its title,
    time, and whether it is one instance of a recurring series, but makes
    no DELETE request, ever (D-08). `event_id` is the model's own choice,
    always found first through `calendar_list_events` (this tool's own
    description says so) -- `account`/`calendar_id` are therefore already
    exact, never ambiguous the way `calendar_propose_event`'s own
    ``resolve_write_target`` has to be.

    `require_writable` gates this call exactly the way it gates the
    executing insert/delete handlers -- an unknown account, an
    unreachable one, a calendar this env does not carry, or a read-only
    calendar all refuse with a spoken reason before any request exists.
    A cancelled or already-gone event (Google answers 404/410, or the
    fetched event's own `status` is `"cancelled"`) raises `Denied("i
    can't find that event any more")`.
    """
    if not accounts:
        raise Denied("no google account is linked -- link one in the admin webapp under google accounts")
    account_grant, calendar_grant = require_writable(accounts, account, calendar_id)
    try:
        raw_event = await get_event(
            client,
            access_token=account_grant.access_token,
            calendar_id=calendar_id,
            event_id=event_id,
            time_zone=str(zone),
        )
    except GoogleAuthError as exc:
        raise Denied(f"i can't reach your {account_grant.label} account right now") from exc
    except GoogleApiError as exc:
        if exc.status in (404, 410):
            raise Denied("i can't find that event any more") from exc
        raise Denied(f"google calendar couldn't look up that event: {exc.message}") from exc

    if raw_event.get("status") == "cancelled":
        raise Denied("i can't find that event any more")

    start = raw_event.get("start", {})
    end = raw_event.get("end", {})
    all_day = "date" in start

    return {
        HANDOFF_KEY: {
            "kind": "pending_action",
            "action": "calendar_delete",
            "account": account_grant.label,
            "calendar_id": calendar_grant.calendar_id,
            "calendar_name": calendar_grant.name,
            "calendar_primary": calendar_grant.primary,
            "title": raw_event.get("summary", "(no title)"),
            "start": start.get("date") or start.get("dateTime"),
            "end": end.get("date") or end.get("dateTime"),
            "all_day": all_day,
            "time_zone": str(zone),
            "event_id": raw_event.get("id") or event_id,
            "recurring_instance": bool(raw_event.get("recurringEventId")),
        }
    }


async def handle_calendar_insert_event(
    accounts: "tuple[AccountGrant, ...]",
    client: httpx.AsyncClient,
    *,
    account: str,
    calendar_id: str,
    title: str,
    start: str,
    end: str,
    all_day: bool,
    time_zone: str,
) -> dict[str, Any]:
    """The executing half of `calendar_propose_event`'s handoff (D-08) --
    posts one `events.insert`, with the exact arguments the stored
    `PendingProposal` resolved. Called by code only, after a spoken
    confirmation; never by the model, never with an argument this call
    itself resolves.

    `require_writable` (T-09-28) re-checks THIS moment's env -- an
    account or calendar the operator narrowed after the readback still
    refuses here, even though the proposal-time `resolve_write_target`
    call already passed.
    """
    account_grant, calendar_grant = require_writable(accounts, account, calendar_id)
    if all_day:
        body: dict[str, Any] = {"summary": title, "start": {"date": start}, "end": {"date": end}}
    else:
        zone = ZoneInfo(time_zone)
        start_dt = _parse_boundary(start, zone)
        end_dt = _parse_boundary(end, zone)
        body = {
            "summary": title,
            "start": {"dateTime": start_dt.isoformat(), "timeZone": time_zone},
            "end": {"dateTime": end_dt.isoformat(), "timeZone": time_zone},
        }
    try:
        raw_event = await insert_event(
            client, access_token=account_grant.access_token, calendar_id=calendar_id, body=body
        )
    except GoogleAuthError as exc:
        raise Denied(f"i can't reach your {account_grant.label} account right now") from exc
    except GoogleApiError as exc:
        raise Denied(f"google calendar couldn't add that event: {exc.message}") from exc
    return {
        "created": {
            "event_id": raw_event.get("id"),
            "account": account_grant.label,
            "calendar": calendar_grant.name,
        }
    }


async def handle_calendar_delete_event(
    accounts: "tuple[AccountGrant, ...]",
    client: httpx.AsyncClient,
    *,
    account: str,
    calendar_id: str,
    event_id: str,
) -> dict[str, Any]:
    """The executing half of `calendar_propose_delete`'s handoff (plan
    09-05's own delete mechanism, D-08) -- sends one `events.delete` for
    `event_id`. `require_writable` (T-09-28) re-checks THIS moment's env
    exactly like the insert handler above.

    A 404 or 410 from Google (the event was already deleted, or a
    recurring instance already cancelled) is reported as "that event is
    already gone" -- never as success, and never a bare exception.
    """
    account_grant, calendar_grant = require_writable(accounts, account, calendar_id)
    try:
        await delete_event(client, access_token=account_grant.access_token, calendar_id=calendar_id, event_id=event_id)
    except GoogleAuthError as exc:
        raise Denied(f"i can't reach your {account_grant.label} account right now") from exc
    except GoogleApiError as exc:
        if exc.status in (404, 410):
            raise Denied("that event is already gone") from exc
        raise Denied(f"google calendar couldn't delete that event: {exc.message}") from exc
    return {
        "deleted": {"event_id": event_id, "account": account_grant.label, "calendar": calendar_grant.name}
    }


def _format_message_item(message: dict[str, Any], *, account: str) -> dict[str, Any]:
    """One `email_list` item's own shape (D-16: no body field, by
    construction) -- `received_at` comes from Gmail's own `internalDate`
    (epoch milliseconds, present on every format, metadata included),
    formatted as an ISO 8601 UTC instant so items sort newest first with
    a plain string comparison."""
    from_name, from_address = parse_from(header_value(message, "From"))
    received_at = ""
    internal_date = message.get("internalDate")
    if internal_date:
        try:
            received_at = (
                datetime.fromtimestamp(int(internal_date) / 1000, tz=_timezone.utc)
                .isoformat()
                .replace("+00:00", "Z")
            )
        except (TypeError, ValueError):
            received_at = ""
    return {
        "account": account,
        "message_id": message.get("id"),
        "thread_id": message.get("threadId"),
        "from_name": from_name,
        "from_address": from_address,
        "subject": header_value(message, "Subject"),
        "received_at": received_at,
    }


async def _account_messages(
    account: AccountGrant, client: httpx.AsyncClient, query: str, semaphore: asyncio.Semaphore
) -> "tuple[list[dict[str, Any]], bool]":
    """Every matching message's own headers for one account -- `has_more`
    true when this account carries more than `_MAX_MESSAGE_IDS` matching
    ids. A `GoogleAuthError`/`GoogleApiError`/`httpx.HTTPError` here
    propagates to the caller's own `gather(..., return_exceptions=True)`,
    the same "never caught in this function" discipline
    `_events_for_calendar` already follows."""
    ids, has_more = await list_message_ids(
        client, access_token=account.access_token, query=query, max_results=_MAX_MESSAGE_IDS
    )

    async def _one(message_id: str) -> dict[str, Any]:
        async with semaphore:
            metadata = await get_message_metadata(client, access_token=account.access_token, message_id=message_id)
        return _format_message_item(metadata, account=account.label)

    items = list(await asyncio.gather(*(_one(entry["id"]) for entry in ids if entry.get("id"))))
    return items, has_more


async def _fan_out_messages(
    resolved: "tuple[AccountGrant, ...]", client: httpx.AsyncClient, *, query: str
) -> "tuple[list[dict[str, Any]], list[str], list[dict[str, str]]]":
    """Every matching message across every resolved account, the labels
    of every account that carried more than `_MAX_MESSAGE_IDS` matches,
    and the `{"account", "reason"}` entry for every account this call
    could not fully answer -- the Gmail-read equivalent of
    `_fan_out_events` (D-04), one job per account rather than per
    calendar (Gmail carries no calendar concept)."""
    unreachable_by_account: dict[str, str] = {}
    jobs: list[AccountGrant] = []
    for one_account in resolved:
        if one_account.access_token is None:
            unreachable_by_account[one_account.label] = one_account.unreachable_reason or "not reachable"
            continue
        jobs.append(one_account)

    semaphore = asyncio.Semaphore(_METADATA_CONCURRENCY)
    results = await asyncio.gather(
        *(_account_messages(one_account, client, query, semaphore) for one_account in jobs),
        return_exceptions=True,
    )

    all_items: list[dict[str, Any]] = []
    has_more_labels: list[str] = []
    for one_account, result in zip(jobs, results):
        if isinstance(result, Exception):
            if one_account.label not in unreachable_by_account:
                unreachable_by_account[one_account.label] = _reason_for(result)
            continue
        items, has_more = result
        all_items.extend(items)
        if has_more:
            has_more_labels.append(one_account.label)

    unreachable = [
        {"account": label, "reason": reason} for label, reason in sorted(unreachable_by_account.items())
    ]
    return all_items, has_more_labels, unreachable


async def handle_gmail_list_unread(
    accounts: "tuple[AccountGrant, ...]", client: httpx.AsyncClient, *, account: "str | None" = None
) -> dict[str, Any]:
    """Answer "any new email?" -- unread mail in the Primary inbox category
    of every linked account, or the one named (D-14, D-15). Returns an
    `email_list` handoff; code, never this function, decides what to say
    (`src/atlas/turn/email_handoff.py::handle_email_list`)."""
    if not accounts:
        raise Denied("no google account is linked -- link one in the admin webapp under google accounts")
    resolved = resolve_accounts(accounts, account)
    items, has_more, unreachable = await _fan_out_messages(resolved, client, query=_GMAIL_UNREAD_QUERY)
    items.sort(key=lambda item: item["received_at"], reverse=True)
    return {
        HANDOFF_KEY: {
            "kind": "email_list",
            "source": "unread",
            "items": items,
            "has_more": has_more,
            "unreachable_accounts": unreachable,
        }
    }


async def handle_gmail_search(
    accounts: "tuple[AccountGrant, ...]",
    client: httpx.AsyncClient,
    *,
    query: str,
    account: "str | None" = None,
) -> dict[str, Any]:
    """A Gmail search the model itself built (for example `from:dana
    newer_than:7d`), sent exactly as given after stripping and a length/
    control-character check -- never queried when empty, too long, or
    carrying a control character (T-09-45). Returns an `email_list`
    handoff, same shape as `handle_gmail_list_unread`."""
    if not accounts:
        raise Denied("no google account is linked -- link one in the admin webapp under google accounts")
    stripped = query.strip()
    if not stripped:
        raise Denied("i need something to search for")
    if len(stripped) > _MAX_SEARCH_QUERY_LEN:
        raise Denied(f"that search is too long -- keep it under {_MAX_SEARCH_QUERY_LEN} characters")
    if _CONTROL_CHAR_RE.search(stripped):
        raise Denied("that search has characters i can't use")
    resolved = resolve_accounts(accounts, account)
    items, has_more, unreachable = await _fan_out_messages(resolved, client, query=stripped)
    items.sort(key=lambda item: item["received_at"], reverse=True)
    return {
        HANDOFF_KEY: {
            "kind": "email_list",
            "source": "search",
            "query": stripped,
            "items": items,
            "has_more": has_more,
            "unreachable_accounts": unreachable,
        }
    }


async def handle_gmail_read(
    *, position: "int | None" = None, sender: "str | None" = None, word_for_word: bool = False
) -> dict[str, Any]:
    """Read one message from the list the operator last heard, by
    position or by sender -- makes no request at all (D-16): resolving
    "the second one" against the last spoken list is code's own job
    (`src/atlas/turn/email_memory.py::EmailListMemory.resolve`), never
    this handler's. Refuses both or neither of `position`/`sender`, a
    position outside `1..50`, and a sender longer than
    `_MAX_SENDER_LEN` characters."""
    if (position is None) == (sender is None):
        raise Denied("tell me which email, by its position in the list or by who sent it -- not both")
    if position is not None and not (1 <= position <= _MAX_READ_POSITION):
        raise Denied("that's not a position in the last list i read")
    if sender is not None and len(sender) > _MAX_SENDER_LEN:
        raise Denied("that sender name is too long")
    return {
        HANDOFF_KEY: {
            "kind": "email_read",
            "position": position,
            "sender": sender,
            "word_for_word": bool(word_for_word),
        }
    }


async def handle_gmail_fetch_body(
    accounts: "tuple[AccountGrant, ...]", client: httpx.AsyncClient, *, account: str, message_id: str
) -> dict[str, Any]:
    """Code-only (`atlas_mcp.google_tools.CODE_ONLY_TOOL_NAMES`): fetch one
    message's full payload, clean its body (`mail_clean.clean_body`, D-17),
    and cap it at `MODEL_INPUT_CAP` before it can ever reach a model round
    -- called by `src/atlas/turn/email_handoff.py` only, after the
    operator's own list or read request named this exact message."""
    normalized = account.strip().casefold()
    account_grant = next((a for a in accounts if a.label.casefold() == normalized), None)
    if account_grant is None:
        linked = ", ".join(sorted(a.label for a in accounts)) or "none linked"
        raise Denied(f"i don't have a google account called {account!r} -- linked accounts: {linked}")
    if account_grant.access_token is None:
        raise Denied(f"i can't reach your {account_grant.label} account right now")
    try:
        raw_message = await get_message_full(
            client, access_token=account_grant.access_token, message_id=message_id
        )
    except GoogleAuthError as exc:
        raise Denied(f"i can't reach your {account_grant.label} account right now") from exc
    except GoogleApiError as exc:
        raise Denied(f"gmail couldn't fetch that message: {exc.message}") from exc

    from_name, from_address = parse_from(header_value(raw_message, "From"))
    subject = header_value(raw_message, "Subject")
    raw_text = extract_text(raw_message.get("payload") or {})
    cleaned = clean_body(raw_text)
    capped, truncated = cap_text(cleaned, MODEL_INPUT_CAP)
    return {
        "body": capped,
        "truncated": truncated,
        "from_name": from_name,
        "from_address": from_address,
        "subject": subject,
    }


mcp_server = MCPServer("atlas-google")

_accounts: tuple[AccountGrant, ...] = ()
_zone: ZoneInfo = ZoneInfo("UTC")
_http_client: httpx.AsyncClient | None = None


@mcp_server.tool()
async def calendar_list_events(
    start: str, end: str, account: "str | None" = None, query: "str | None" = None
) -> dict[str, Any]:
    """Answer a question about calendar events in a time range, across
    every linked Google account unless `account` names one. Always name
    each returned event's own `account` label in your reply (D-04) --
    the same event title can exist on more than one account. `start` and
    `end` are ISO dates or datetimes; a time with no offset is in the
    house's own time zone. Any account this call could not reach is
    listed in `unreachable_accounts` with why -- say so, never silently
    leave it out.
    """
    assert _http_client is not None, "calendar_list_events invoked before startup"
    try:
        return await handle_calendar_list_events(
            _accounts, _http_client, _zone, start=start, end=end, account=account, query=query
        )
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def calendar_propose_event(
    title: str,
    start: str,
    end: "str | None" = None,
    duration_minutes: "int | None" = None,
    all_day: bool = False,
    account: "str | None" = None,
    calendar: "str | None" = None,
) -> dict[str, Any]:
    """Propose adding one calendar event. This does NOT add the event --
    the operator hears a spoken readback of exactly this proposal and must
    confirm it before anything changes in Google Calendar. Never tell the
    operator the event was added; it has not been, and only their spoken
    confirmation makes it so. `start`/`end` are ISO dates or datetimes; a
    time with no offset is in the house's own time zone. Leave `account`/
    `calendar` unset unless the operator named one."""
    try:
        return await handle_calendar_propose_event(
            _accounts,
            _zone,
            title=title,
            start=start,
            end=end,
            duration_minutes=duration_minutes,
            all_day=all_day,
            account=account,
            calendar=calendar,
        )
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def calendar_propose_delete(account: str, calendar_id: str, event_id: str) -> dict[str, Any]:
    """Propose deleting one calendar event -- found first with
    `calendar_list_events`, whose own results carry each event's
    `account`, `calendar_id`, and `event_id`. This does NOT delete the
    event -- the operator hears a spoken readback and must confirm it
    before anything changes in Google Calendar. One event per call: if
    the operator asks to delete several events, say that bulk changes are
    made directly in Google Calendar, and do not call this more than once
    in the same turn."""
    assert _http_client is not None, "calendar_propose_delete invoked before startup"
    try:
        return await handle_calendar_propose_delete(
            _accounts, _http_client, _zone, account=account, calendar_id=calendar_id, event_id=event_id
        )
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def calendar_insert_event(
    account: str,
    calendar_id: str,
    title: str,
    start: str,
    end: str,
    all_day: bool,
    time_zone: str,
) -> dict[str, Any]:
    """Insert one calendar event. Called by the assistant itself right
    after the operator confirms a proposal out loud -- never call this
    directly; propose the event with `calendar_propose_event` and let the
    operator's spoken confirmation drive this call."""
    assert _http_client is not None, "calendar_insert_event invoked before startup"
    try:
        return await handle_calendar_insert_event(
            _accounts,
            _http_client,
            account=account,
            calendar_id=calendar_id,
            title=title,
            start=start,
            end=end,
            all_day=all_day,
            time_zone=time_zone,
        )
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def calendar_delete_event(account: str, calendar_id: str, event_id: str) -> dict[str, Any]:
    """Delete one calendar event. Called by the assistant itself right
    after the operator confirms a proposed deletion out loud -- never
    call this directly; propose the deletion with `calendar_propose_delete`
    and let the operator's spoken confirmation drive this call."""
    assert _http_client is not None, "calendar_delete_event invoked before startup"
    try:
        return await handle_calendar_delete_event(
            _accounts, _http_client, account=account, calendar_id=calendar_id, event_id=event_id
        )
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def gmail_list_unread(account: "str | None" = None) -> dict[str, Any]:
    """Answer "any new email?" -- unread mail in the Primary inbox category
    of every linked Google account, or the one `account` names. Never
    returns any email's own text to you: for one or two unread messages
    you get a short summary each; for more you get the count and the
    senders. Leave `account` unset unless the operator named one."""
    assert _http_client is not None, "gmail_list_unread invoked before startup"
    try:
        return await handle_gmail_list_unread(_accounts, _http_client, account=account)
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def gmail_search(query: str, account: "str | None" = None) -> dict[str, Any]:
    """Search Gmail with a query you build in Gmail's own search grammar
    (for example `from:dana newer_than:7d`), across every linked account
    unless `account` names one. Never returns any email's own text to
    you -- the same summarized-or-counted answer `gmail_list_unread`
    gives."""
    assert _http_client is not None, "gmail_search invoked before startup"
    try:
        return await handle_gmail_search(_accounts, _http_client, query=query, account=account)
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def gmail_read(
    position: "int | None" = None, sender: "str | None" = None, word_for_word: bool = False
) -> dict[str, Any]:
    """Read one message from the list the operator last heard -- name it
    by its position in that list (1 for the first one spoken) or by who
    sent it, not both. Summarized by default; set `word_for_word` true
    only when the operator asked to hear it exactly as written. Never
    returns any email's own text to you."""
    try:
        return await handle_gmail_read(position=position, sender=sender, word_for_word=word_for_word)
    except Denied as exc:
        raise ToolError(exc.reason) from exc


@mcp_server.tool()
async def gmail_fetch_body(account: str, message_id: str) -> dict[str, Any]:
    """Code-only: fetch and clean one message's body. Never call this --
    it is reached only after the operator's own request to read a
    specific message, resolved by code from the last list you spoke."""
    assert _http_client is not None, "gmail_fetch_body invoked before startup"
    try:
        return await handle_gmail_fetch_body(_accounts, _http_client, account=account, message_id=message_id)
    except Denied as exc:
        raise ToolError(exc.reason) from exc


def _resolve_zone() -> ZoneInfo:
    """The house's own time zone, from `TZ` (260924-h2f, issue #1) -- a
    missing or unloadable zone falls back to UTC and says so on stderr,
    the same refuse-to-guess-silently posture every other startup
    validation in this project takes, without refusing to start entirely
    (a wrong time zone is a degraded answer, not a reason to withhold
    calendar reads)."""
    raw = os.environ.get("TZ")
    if not raw:
        return ZoneInfo("UTC")
    try:
        return ZoneInfo(raw)
    except ZoneInfoNotFoundError:
        print(f"atlas_mcp.google: unknown time zone {raw!r} -- falling back to UTC", flush=True)
        return ZoneInfo("UTC")


def _startup() -> None:
    global _accounts, _zone, _http_client
    raw_accounts = os.environ.get(GOOGLE_ACCOUNTS_ENV)
    try:
        _accounts = parse_accounts_env(raw_accounts)
    except ValueError as exc:
        raise SystemExit(f"{GOOGLE_ACCOUNTS_ENV} is invalid: {exc}") from exc
    _zone = _resolve_zone()
    _http_client = httpx.AsyncClient(timeout=10.0)


async def _run() -> None:
    _startup()
    try:
        await mcp_server.run_stdio_async()
    finally:
        if _http_client is not None:
            await _http_client.aclose()


if __name__ == "__main__":
    asyncio.run(_run())
