"""Plain async functions over a caller-supplied `httpx.AsyncClient` for the
parts of Google's OAuth and Calendar surface `mcp/atlas_mcp/google_api.py`
does not already cover -- no Google SDK, matching that module's own
"one focused async function per concern, client passed in" style.

Reuses `google_api.py`'s own error vocabulary (`GoogleApiError`,
`GoogleAuthError`) rather than inventing a second one: every non-2xx
response from any function below is raised as one of those, never
returned as a silent empty result.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from atlas_mcp.google_api import CALENDAR_BASE, TOKEN_URL, GoogleApiError, GoogleAuthError
from atlas_mcp.google_tools import REQUIRED_SCOPES

AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
REVOKE_URL = "https://oauth2.googleapis.com/revoke"
GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"

# D-05: a calendar Google reports at this access role carries no event
# details, so it is skipped entirely from `list_calendars`'s own result --
# there is nothing this codebase could ever read from one anyway.
_NO_DETAIL_ACCESS_ROLE = "freeBusyReader"
_WRITABLE_ACCESS_ROLES = ("owner", "writer")


def _error_message(response: httpx.Response) -> str:
    """The boundary's own words, nothing reworded -- the identical pattern
    `google_api.py::_error_message` already establishes for the endpoints
    it covers, duplicated here (rather than imported across modules) so
    this module has no dependency beyond the shared error classes."""
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


def _raise_for_status(response: httpx.Response) -> None:
    if response.status_code // 100 == 2:
        return
    message = _error_message(response)
    if response.status_code in (401, 403):
        raise GoogleAuthError(response.status_code, message)
    raise GoogleApiError(response.status_code, message)


def build_consent_url(client_id: str, redirect_uri: str, state: str) -> str:
    """The one consent URL D-03 needs: every `REQUIRED_SCOPES` entry in one
    request, `access_type=offline` so Google issues a refresh token, and
    `prompt=consent` so a re-link gets a fresh one too -- without it,
    Google only issues a refresh token on an account's very first-ever
    consent for this client, and a re-link would silently get none."""
    params = {
        "client_id": client_id,
        "redirect_uri": redirect_uri,
        "response_type": "code",
        "scope": " ".join(REQUIRED_SCOPES),
        "access_type": "offline",
        "prompt": "consent",
        "state": state,
    }
    return f"{AUTH_URL}?{urllib.parse.urlencode(params)}"


async def exchange_code(
    client: httpx.AsyncClient,
    *,
    client_id: str,
    client_secret: str,
    code: str,
    redirect_uri: str,
) -> dict[str, Any]:
    """Exchange an authorization `code` for the token response (`access_token`,
    `refresh_token` when Google issues one, `scope`, and
    `refresh_token_expires_in` when the consent screen is still in Testing
    mode, D-01/Pitfall 1). Raises `GoogleAuthError`/`GoogleApiError` on any
    non-2xx -- never a silent partial result."""
    response = await client.post(
        TOKEN_URL,
        data={
            "client_id": client_id,
            "client_secret": client_secret,
            "code": code,
            "redirect_uri": redirect_uri,
            "grant_type": "authorization_code",
        },
    )
    _raise_for_status(response)
    return response.json()


async def revoke_token(client: httpx.AsyncClient, token: str) -> None:
    """Best effort -- never raises. A revoke that fails (network error, or
    Google already considers the token gone) must never block the caller's
    own write; `complete_link`/the unlink route both rely on this."""
    try:
        await client.post(REVOKE_URL, params={"token": token})
    except httpx.HTTPError:
        pass


async def get_gmail_profile(client: httpx.AsyncClient, access_token: str) -> str:
    """The linked account's own address, read from Gmail's `users/me/profile`
    rather than asking for an extra `openid`/`email` scope this phase does
    not otherwise need."""
    response = await client.get(
        f"{GMAIL_BASE}/users/me/profile",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    _raise_for_status(response)
    body = response.json()
    return body["emailAddress"]


async def list_calendars(client: httpx.AsyncClient, access_token: str) -> list[dict[str, Any]]:
    """Every calendar `calendarList.list` reports for this account, paged
    to completion, each reduced to `{"id", "name", "primary", "can_write"}`
    -- `name` prefers `summaryOverride` (the operator's own rename in
    Google Calendar) over `summary`. A calendar carrying access role
    `freeBusyReader` is skipped: it exposes no event details, so nothing
    in this codebase could ever read from it."""
    calendars: list[dict[str, Any]] = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {}
        if page_token:
            params["pageToken"] = page_token
        response = await client.get(
            f"{CALENDAR_BASE}/users/me/calendarList",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
        _raise_for_status(response)
        body = response.json()
        for item in body.get("items", []):
            access_role = item.get("accessRole")
            if access_role == _NO_DETAIL_ACCESS_ROLE:
                continue
            calendars.append(
                {
                    "id": item["id"],
                    "name": item.get("summaryOverride") or item.get("summary") or item["id"],
                    "primary": bool(item.get("primary", False)),
                    "can_write": access_role in _WRITABLE_ACCESS_ROLES,
                }
            )
        page_token = body.get("nextPageToken")
        if not page_token:
            break
    return calendars
