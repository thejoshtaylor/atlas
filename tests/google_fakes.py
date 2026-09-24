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

from atlas_mcp.google_api import CALENDAR_BASE, TOKEN_URL


class FakeGoogle:
    """In-memory refresh tokens and calendar events, keyed by the access
    token a caller presents -- plus every request this fake ever
    answered, for a test to assert on directly (which bearer token called
    which URL)."""

    def __init__(self) -> None:
        self.refresh_tokens: dict[str, str] = {}
        self.events: dict[tuple[str, str], list[dict[str, Any]]] = {}
        self.requests: list[httpx.Request] = []
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

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.copy_with(query=None) == httpx.URL(TOKEN_URL):
            return self._handle_token(request)
        if str(request.url).startswith(CALENDAR_BASE) and request.url.path.endswith("/events"):
            return self._handle_events(request)
        return httpx.Response(404, json={"error": "not_found"})

    def _handle_token(self, request: httpx.Request) -> httpx.Response:
        data = dict(urllib.parse.parse_qsl(request.content.decode("utf-8")))
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

    def _handle_events(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        # `.../calendars/{calendar_id}/events` -- the calendar id is the
        # second-to-last path segment, URL-decoded (`google_api.list_events`
        # encodes it with `urllib.parse.quote(calendar_id, safe="")`).
        segments = request.url.path.split("/")
        calendar_id = urllib.parse.unquote(segments[-2])
        events = self.events.get((access_token, calendar_id), [])
        return httpx.Response(200, json={"items": events})
