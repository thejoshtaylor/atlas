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
from datetime import date, datetime, timedelta, timezone as _timezone
from typing import Any
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import httpx

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError

from atlas_mcp.google_api import GoogleApiError, GoogleAuthError, list_events
from atlas_mcp.google_boundary import AccountGrant, parse_accounts_env, resolve_accounts
from atlas_mcp.google_tools import GOOGLE_ACCOUNTS_ENV
from atlas_mcp.safety import Denied

_MAX_RANGE_DAYS = 31


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
