"""Plain async functions over a caller-supplied `httpx.AsyncClient` --
no Google SDK (09-RESEARCH.md's own "Don't-Hand-Roll" verdict: a token
POST and an `events.list` GET are fixed-shape, fully-documented requests,
not the kind of deceptively complex problem that philosophy protects
against). Matches `mcp/atlas_mcp/ha.py`'s and `open_meteo.py`'s own
"one focused async function per concern, client passed in" style, never a
class-based client.

`GoogleApiError` and its two subclasses are this module's one error
vocabulary: every non-2xx response from either endpoint below is raised as
one of these, never returned as a silent empty result (the same
"non-2xx is surfaced as an error, never an empty success" discipline
`ha.py::handle_call_service`'s own docstring states).
"""

from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

TOKEN_URL = "https://oauth2.googleapis.com/token"
CALENDAR_BASE = "https://www.googleapis.com/calendar/v3"


class GoogleApiError(Exception):
    """A non-2xx response from a Google endpoint -- `status` is the HTTP
    status code, `message` is Google's own explanation when the body
    carried one, or a truncated fallback of the raw response text."""

    def __init__(self, status: int, message: str) -> None:
        self.status = status
        self.message = message
        super().__init__(f"google api returned {status}: {message}")


class GoogleAuthError(GoogleApiError):
    """A 401 or 403 -- the access token this call used was rejected or
    lacks the scope the call needed. `google.py`'s own callers map this
    to the account-degrade reason "not authorized" (GOOG-12)."""


class GoogleGrantRevokedError(GoogleApiError):
    """The token endpoint answered `invalid_grant` -- the refresh token
    itself no longer works (revoked by the operator, or Google's own
    7-day Testing-mode expiry, D-01). `GoogleTokenService` (Task 3) maps
    this to account status `needs_relink`."""


def _error_message(response: httpx.Response) -> str:
    """Pull Google's own explanation out of a failed response -- the same
    "the boundary's own words, nothing reworded" pattern `ha.py::_ha_message`
    already establishes, adapted to Google's `{"error": ..., "error_description": ...}`
    (token endpoint) and `{"error": {"message": ...}}` (REST endpoints) shapes."""
    try:
        body = response.json()
    except ValueError:
        return response.text.strip()[:200]
    if isinstance(body, dict):
        error = body.get("error")
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str) and message.strip():
                return message.strip()
        if isinstance(error, str) and error.strip():
            description = body.get("error_description")
            if isinstance(description, str) and description.strip():
                return f"{error.strip()}: {description.strip()}"
            return error.strip()
    return response.text.strip()[:200]


def _is_invalid_grant(response: httpx.Response) -> bool:
    try:
        body = response.json()
    except ValueError:
        return False
    return isinstance(body, dict) and body.get("error") == "invalid_grant"


async def refresh_access_token(
    client: httpx.AsyncClient,
    *,
    client_id: str,
    client_secret: str,
    refresh_token: str,
) -> dict[str, Any]:
    """Exchange a refresh token for a fresh access token, over the token
    endpoint's documented `grant_type=refresh_token` POST. Returns the
    decoded JSON body (`access_token`, `expires_in`, `scope`, `token_type`
    -- no new `refresh_token` unless Google chose to rotate it).

    Raises `GoogleGrantRevokedError` on an `invalid_grant` response
    (the refresh token no longer works), `GoogleAuthError` on any other
    401/403, and `GoogleApiError` on any other non-2xx.
    """
    response = await client.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "refresh_token": refresh_token,
            "grant_type": "refresh_token",
        },
    )
    if response.status_code // 100 != 2:
        message = _error_message(response)
        if _is_invalid_grant(response):
            raise GoogleGrantRevokedError(response.status_code, message)
        if response.status_code in (401, 403):
            raise GoogleAuthError(response.status_code, message)
        raise GoogleApiError(response.status_code, message)
    return response.json()


async def list_events(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    calendar_id: str,
    time_min: str,
    time_max: str,
    time_zone: str,
    query: "str | None" = None,
    max_results: int = 50,
) -> list[dict[str, Any]]:
    """List events on one calendar in `[time_min, time_max)`, expanded
    (`singleEvents=true`) and ordered by start time -- the shape
    09-RESEARCH.md's own Code Example gives, extended with an optional
    text `query` and a `max_results` bound.

    `calendar_id` is URL-encoded as one path segment (`urllib.parse.quote`,
    `safe=""`) because a real calendar id commonly contains `@` and `#`.

    Raises `GoogleAuthError` on 401/403 (an expired or revoked access
    token -- the caller maps this to the account-degrade reason "not
    authorized", GOOG-12) and `GoogleApiError` on any other non-2xx.
    """
    encoded_calendar_id = urllib.parse.quote(calendar_id, safe="")
    params: dict[str, Any] = {
        "timeMin": time_min,
        "timeMax": time_max,
        "timeZone": time_zone,
        "singleEvents": "true",
        "orderBy": "startTime",
        "maxResults": max_results,
    }
    if query:
        params["q"] = query
    response = await client.get(
        f"{CALENDAR_BASE}/calendars/{encoded_calendar_id}/events",
        headers={"Authorization": f"Bearer {access_token}"},
        params=params,
    )
    if response.status_code // 100 != 2:
        message = _error_message(response)
        if response.status_code in (401, 403):
            raise GoogleAuthError(response.status_code, message)
        raise GoogleApiError(response.status_code, message)
    body = response.json()
    return list(body.get("items", []))


async def get_event(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    calendar_id: str,
    event_id: str,
    time_zone: str,
) -> dict[str, Any]:
    """Fetch one event by id -- plan 09-05's own `calendar_propose_delete`
    reads the event once here before building its handoff, and
    `handle_calendar_delete_event`'s executing half re-reads through
    `events.delete` directly rather than this function (delete needs no
    prior read of its own).

    `calendar_id`/`event_id` are each URL-encoded as their own path
    segment (`urllib.parse.quote`, `safe=""`) -- a real calendar or event
    id commonly contains `@` and `#`.

    Raises `GoogleAuthError` on 401/403 and `GoogleApiError` on any other
    non-2xx, including a 404 (deleted or never existed) or 410 (a
    recurring instance already cancelled) -- the caller maps those two
    codes to a spoken "already gone", never treats them as success.
    """
    encoded_calendar_id = urllib.parse.quote(calendar_id, safe="")
    encoded_event_id = urllib.parse.quote(event_id, safe="")
    response = await client.get(
        f"{CALENDAR_BASE}/calendars/{encoded_calendar_id}/events/{encoded_event_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"timeZone": time_zone},
    )
    if response.status_code // 100 != 2:
        message = _error_message(response)
        if response.status_code in (401, 403):
            raise GoogleAuthError(response.status_code, message)
        raise GoogleApiError(response.status_code, message)
    return response.json()


async def insert_event(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    calendar_id: str,
    body: dict[str, Any],
) -> dict[str, Any]:
    """POST one `events.insert` body, returning Google's own created event
    (carrying the assigned `id`). `calendar_id` is URL-encoded the same
    way `list_events`/`get_event` already encode it.

    Raises `GoogleAuthError` on 401/403 and `GoogleApiError` on any other
    non-2xx.
    """
    encoded_calendar_id = urllib.parse.quote(calendar_id, safe="")
    response = await client.post(
        f"{CALENDAR_BASE}/calendars/{encoded_calendar_id}/events",
        headers={"Authorization": f"Bearer {access_token}"},
        json=body,
    )
    if response.status_code // 100 != 2:
        message = _error_message(response)
        if response.status_code in (401, 403):
            raise GoogleAuthError(response.status_code, message)
        raise GoogleApiError(response.status_code, message)
    return response.json()


async def delete_event(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    calendar_id: str,
    event_id: str,
) -> None:
    """DELETE one event id -- deleting a recurring occurrence's own
    instance id cancels only that occurrence, per Google's documented
    recurring-events behavior (assumption A1, confirmed live in plan
    09-06's human check).

    Raises `GoogleAuthError` on 401/403 and `GoogleApiError` on any other
    non-2xx, including 404/410 for an event already gone -- the caller
    maps those two codes to a spoken "already gone", never treats them as
    success.
    """
    encoded_calendar_id = urllib.parse.quote(calendar_id, safe="")
    encoded_event_id = urllib.parse.quote(event_id, safe="")
    response = await client.delete(
        f"{CALENDAR_BASE}/calendars/{encoded_calendar_id}/events/{encoded_event_id}",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    if response.status_code // 100 != 2:
        message = _error_message(response)
        if response.status_code in (401, 403):
            raise GoogleAuthError(response.status_code, message)
        raise GoogleApiError(response.status_code, message)
