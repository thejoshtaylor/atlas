"""`FakeGoogle`: an `httpx.MockTransport` handler standing in for both
Google endpoints this plan calls -- the token endpoint (`refresh_access_token`)
and `events.list` (`list_events`), both in `mcp/atlas_mcp/google_api.py`.
The direct sibling of `tests/conftest.py::FakeHomeAssistant`'s own
in-process-transport pattern.

This plan serves the token refresh and `events.list` only; later plans in
this phase add Gmail endpoints to this same fake.

Every token and secret literal anywhere this fake is used in a test is
shorter than eight characters (`"rt-1"`, `"at-1"`, `"cs-1"`) or a dict
key -- `tests/test_repo_hygiene.py::_CREDENTIAL_RE` scans every tracked
file and all of git history for `token|secret|password|api_key` followed
by an eight-plus character value, and a mistake here is permanent.
"""

from __future__ import annotations

import urllib.parse
from typing import Any

import httpx

from atlas_mcp.google_account_api import GMAIL_BASE, REVOKE_URL
from atlas_mcp.google_api import CALENDAR_BASE, TOKEN_URL
from atlas_mcp.google_tools import REQUIRED_SCOPES


class FakeGoogle:
    """In-memory refresh tokens and calendar events, keyed by the access
    token a caller presents -- plus every request this fake ever
    answered, for a test to assert on directly (which bearer token called
    which URL)."""

    def __init__(self) -> None:
        self.refresh_tokens: dict[str, str] = {}
        self.events: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.requests: list[httpx.Request] = []
        # Task 2: per-`events.list` failure injection, keyed by
        # `(access_token, calendar_id)` -- `calendar_id=None` fails every
        # calendar that access token calls. Value is `(status, raise_connect_error)`;
        # `status` alone answers a non-2xx response (401/403 -> auth
        # failure, anything else -> a generic API failure);
        # `raise_connect_error=True` raises `httpx.ConnectError` instead,
        # simulating a transport-level failure with no response at all.
        self._event_failures: dict[tuple[str, "str | None"], tuple["int | None", bool]] = {}
        # Plan 09-03: one authorization `code` exchanges to exactly one
        # scripted token response -- `add_code` below.
        self._codes: dict[str, dict[str, Any]] = {}
        self._profiles: dict[str, str] = {}
        self._calendar_lists: dict[str, list[dict[str, Any]]] = {}
        self.revoked: list[str] = []
        self._transport = httpx.MockTransport(self._handle)

    @property
    def client(self) -> httpx.AsyncClient:
        """A fresh `httpx.AsyncClient` bound to this fake's own in-process
        transport -- never a real socket."""
        return httpx.AsyncClient(transport=self._transport)

    def add_refresh_token(self, refresh_token: str, access_token: str) -> None:
        """Make `refresh_token` exchange to `access_token` at the token
        endpoint -- a refresh token this fake was never told about
        answers `invalid_grant`, the same shape a real revoked token
        does."""
        self.refresh_tokens[refresh_token] = access_token

    def add_events(self, access_token: str, calendar_id: str, events: list[dict[str, Any]]) -> None:
        """Serve `events` from `calendar_id` to any caller presenting
        `access_token` -- a calendar this fake was never told about
        answers an empty list, matching a real, genuinely-empty
        calendar."""
        self.events[(access_token, calendar_id)] = list(events)

    def fail_events(
        self,
        access_token: str,
        calendar_id: "str | None" = None,
        *,
        status: "int | None" = None,
        raise_connect_error: bool = False,
    ) -> None:
        """Task 2: make `events.list` fail for `access_token` (every
        calendar, when `calendar_id` is `None`, or one calendar only) --
        either a non-2xx `status` (401/403 for an auth failure, any other
        code for a generic API failure) or, with `raise_connect_error=True`,
        a raised `httpx.ConnectError` with no response at all (a transport
        failure)."""
        self._event_failures[(access_token, calendar_id)] = (status, raise_connect_error)

    def requests_by_bearer(self, access_token: str) -> list[httpx.Request]:
        """Every request this fake answered carrying `access_token` as its
        bearer token -- so a test can assert exactly which account's own
        token reached which URL."""
        prefix = f"Bearer {access_token}"
        return [r for r in self.requests if r.headers.get("authorization") == prefix]

    def add_code(
        self,
        code: str,
        *,
        access_token: str,
        refresh_token: str | None = None,
        scope: "str | None" = None,
        refresh_token_expires_in: "float | None" = None,
    ) -> None:
        """Make authorization `code` exchange to a token response carrying
        `access_token` -- `refresh_token=None` answers a response with no
        `refresh_token` field at all (matching Google's own shape when it
        does not issue one); `scope` defaults to every `REQUIRED_SCOPES`
        entry, granted (a test narrows it to exercise `scopes_missing`).
        A `code` this fake was never told about answers `invalid_grant`,
        the same shape a real expired/reused code does."""
        self._codes[code] = {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "scope": scope if scope is not None else " ".join(REQUIRED_SCOPES),
            "refresh_token_expires_in": refresh_token_expires_in,
        }

    def add_profile(self, access_token: str, email: str) -> None:
        """Serve `email` from `users/me/profile` to any caller presenting
        `access_token`."""
        self._profiles[access_token] = email

    def add_calendar_list(self, access_token: str, calendars: list[dict[str, Any]]) -> None:
        """Serve `calendars` (each a raw `calendarList.list` item shape --
        `id`, `summary`, optionally `summaryOverride`/`primary`/
        `accessRole`) from `calendarList.list` to any caller presenting
        `access_token`."""
        self._calendar_lists[access_token] = calendars

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.copy_with(query=None) == httpx.URL(TOKEN_URL):
            return self._handle_token(request)
        if request.url.copy_with(query=None) == httpx.URL(REVOKE_URL):
            return self._handle_revoke(request)
        if str(request.url).startswith(GMAIL_BASE) and request.url.path.endswith("/profile"):
            return self._handle_profile(request)
        if str(request.url).startswith(CALENDAR_BASE) and request.url.path.endswith("/calendarList"):
            return self._handle_calendar_list(request)
        if str(request.url).startswith(CALENDAR_BASE) and request.url.path.endswith("/events"):
            return self._handle_events(request)
        return httpx.Response(404, json={"error": "not_found"})

    def _handle_token(self, request: httpx.Request) -> httpx.Response:
        data = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
        grant_type = data.get("grant_type")

        if grant_type == "authorization_code":
            entry = self._codes.get(data.get("code") or "")
            if entry is None:
                return httpx.Response(
                    400, json={"error": "invalid_grant", "error_description": "bad code"}
                )
            body: dict[str, Any] = {
                "access_token": entry["access_token"],
                "expires_in": 3600,
                "token_type": "Bearer",
                "scope": entry["scope"],
            }
            if entry.get("refresh_token"):
                body["refresh_token"] = entry["refresh_token"]
            if entry.get("refresh_token_expires_in") is not None:
                body["refresh_token_expires_in"] = entry["refresh_token_expires_in"]
            return httpx.Response(200, json=body)

        refresh_token = data.get("refresh_token")
        access_token = self.refresh_tokens.get(refresh_token or "")
        if access_token is None:
            return httpx.Response(
                400, json={"error": "invalid_grant", "error_description": "bad refresh token"}
            )
        return httpx.Response(
            200,
            json={"access_token": access_token, "expires_in": 3600, "token_type": "Bearer"},
        )

    def _handle_revoke(self, request: httpx.Request) -> httpx.Response:
        token = request.url.params.get("token")
        if token:
            self.revoked.append(token)
        return httpx.Response(200, json={})

    def _handle_profile(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        email = self._profiles.get(access_token)
        if email is None:
            return httpx.Response(401, json={"error": {"message": "invalid credentials"}})
        return httpx.Response(200, json={"emailAddress": email})

    def _handle_calendar_list(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        items = self._calendar_lists.get(access_token, [])
        return httpx.Response(200, json={"items": items})

    def _handle_events(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        # `.../calendars/{calendar_id}/events` -- the calendar id is the
        # second-to-last path segment, URL-decoded (`google_api.list_events`
        # encodes it with `urllib.parse.quote(calendar_id, safe="")`).
        segments = request.url.path.split("/")
        calendar_id = urllib.parse.unquote(segments[-2])

        failure = self._event_failures.get((access_token, calendar_id)) or self._event_failures.get(
            (access_token, None)
        )
        if failure is not None:
            status, raise_connect_error = failure
            if raise_connect_error:
                raise httpx.ConnectError("connection refused", request=request)
            if status is not None:
                return httpx.Response(status, json={"error": {"message": "forced failure"}})

        events = self.events.get((access_token, calendar_id), [])
        return httpx.Response(200, json={"items": events})
