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

import base64
import json
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
        # Plan 09-05, Task 1: `events.get`/`events.insert`/`events.delete` --
        # `inserted` records every insert's own body (for a test to assert
        # what was actually posted), `deleted_event_ids` every id this fake
        # actually removed, and `_write_failures` forces a specific status
        # (404/410 for "already gone", anything else for a generic API
        # failure) for one `(access_token, calendar_id, event_id)` triple's
        # `get`/`delete` call, mirroring `fail_events`'s own shape for reads.
        self.inserted: list[dict[str, Any]] = []
        self.deleted_event_ids: list[str] = []
        self._write_failures: dict[tuple[str, str, str], int] = {}
        # Plan 09-08: Gmail `messages.list`/`messages.get` -- `_gmail_messages`
        # is one access token's own ordered `{"id", "threadId"}` list
        # (`messages.list` pages at `maxResults`, reporting `nextPageToken`
        # for anything left over); `_gmail_metadata`/`_gmail_full` are each
        # keyed `(access_token, message_id)`, seeded independently so a test
        # can give a message metadata headers without ever giving it a body,
        # or the reverse. `_gmail_list_failures` mirrors `_event_failures`'
        # own per-account failure-injection shape for `messages.list`.
        self._gmail_messages: dict[str, list[dict[str, str]]] = {}
        self._gmail_metadata: dict[tuple[str, str], dict[str, Any]] = {}
        self._gmail_full: dict[tuple[str, str], dict[str, Any]] = {}
        self._gmail_list_failures: dict[str, tuple["int | None", bool]] = {}
        # Plan 09-09: `users.settings.sendAs.list` -- each access token's
        # own raw `sendAs` entry list, seeded via `add_send_as`.
        self._send_as: dict[str, list[dict[str, Any]]] = {}
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

    def add_event(self, access_token: str, calendar_id: str, event: dict[str, Any]) -> None:
        """Add one event (carrying its own `id`) to a calendar's list --
        additive, unlike `add_events`' own replace-the-whole-list shape, so
        a test can seed one known event for `events.get`/`events.delete`
        to find by id."""
        self.events.setdefault((access_token, calendar_id), []).append(dict(event))

    def fail_write(
        self, access_token: str, calendar_id: str, event_id: str, *, status: int
    ) -> None:
        """Make `events.get`/`events.delete` for this exact
        `(access_token, calendar_id, event_id)` triple answer `status`
        instead of the real lookup -- a 404/410 simulates an event already
        cancelled or gone; any other status simulates a generic API
        failure."""
        self._write_failures[(access_token, calendar_id, event_id)] = status

    def add_calendar_list(self, access_token: str, calendars: list[dict[str, Any]]) -> None:
        """Serve `calendars` (each a raw `calendarList.list` item shape --
        `id`, `summary`, optionally `summaryOverride`/`primary`/
        `accessRole`) from `calendarList.list` to any caller presenting
        `access_token`."""
        self._calendar_lists[access_token] = calendars

    def add_gmail_messages(self, access_token: str, message_ids: list[str]) -> None:
        """Seed `access_token`'s own `messages.list` result, in the order
        given -- every call (unread list or search alike) returns this
        same page regardless of `q`; a test asserts the query separately
        off `requests_by_bearer`."""
        self._gmail_messages[access_token] = [{"id": mid, "threadId": mid} for mid in message_ids]

    def add_gmail_metadata(
        self,
        access_token: str,
        message_id: str,
        *,
        headers: dict[str, str],
        internal_date: "str | None" = None,
    ) -> None:
        """Seed one message's `format=metadata` response -- `headers` is
        `{name: value}` (already RFC 2047 encoded when a test wants to
        prove decoding); `internal_date` is Gmail's own epoch-millisecond
        string, defaulted to `"0"` (the epoch) so every seeded message
        sorts deterministically even when a test does not care about
        ordering."""
        self._gmail_metadata[(access_token, message_id)] = {
            "id": message_id,
            "threadId": message_id,
            "internalDate": internal_date if internal_date is not None else "0",
            "payload": {"headers": [{"name": name, "value": value} for name, value in headers.items()]},
        }

    def add_gmail_full(
        self,
        access_token: str,
        message_id: str,
        *,
        headers: dict[str, str],
        text: "str | None" = None,
        html: "str | None" = None,
    ) -> None:
        """Seed one message's `format=full` response -- a `text/plain`
        part when `text` is given, else a `text/html` part when `html`
        is given, else a part with no `body.data` at all (an empty
        message). Base64url-encoded the same way Gmail's own API does
        (`=` padding stripped), so `mail_clean.extract_text`'s own padding
        restoration is exercised for real."""
        if text is not None:
            data = base64.urlsafe_b64encode(text.encode("utf-8")).decode("ascii").rstrip("=")
            payload = {
                "mimeType": "text/plain",
                "headers": [{"name": name, "value": value} for name, value in headers.items()],
                "body": {"data": data},
            }
        elif html is not None:
            data = base64.urlsafe_b64encode(html.encode("utf-8")).decode("ascii").rstrip("=")
            payload = {
                "mimeType": "text/html",
                "headers": [{"name": name, "value": value} for name, value in headers.items()],
                "body": {"data": data},
            }
        else:
            payload = {
                "mimeType": "text/plain",
                "headers": [{"name": name, "value": value} for name, value in headers.items()],
                "body": {},
            }
        self._gmail_full[(access_token, message_id)] = {
            "id": message_id,
            "threadId": message_id,
            "payload": payload,
        }

    def fail_gmail_list(
        self, access_token: str, *, status: "int | None" = None, raise_connect_error: bool = False
    ) -> None:
        """Make `messages.list` fail for `access_token` -- the Gmail
        equivalent of `fail_events` above."""
        self._gmail_list_failures[access_token] = (status, raise_connect_error)

    def add_send_as(self, access_token: str, entries: list[dict[str, Any]]) -> None:
        """Seed `access_token`'s own `users.settings.sendAs.list` result --
        each entry a raw send-as shape (`sendAsEmail`, `isDefault`,
        `isPrimary`, `signature`, as Gmail's own API returns them)."""
        self._send_as[access_token] = entries

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.url.copy_with(query=None) == httpx.URL(TOKEN_URL):
            return self._handle_token(request)
        if request.url.copy_with(query=None) == httpx.URL(REVOKE_URL):
            return self._handle_revoke(request)
        if str(request.url).startswith(GMAIL_BASE) and request.url.path.endswith("/profile"):
            return self._handle_profile(request)
        if str(request.url).startswith(GMAIL_BASE) and request.url.path.endswith("/settings/sendAs"):
            return self._handle_send_as(request)
        if str(request.url).startswith(GMAIL_BASE) and request.url.path.endswith("/messages"):
            return self._handle_gmail_list(request)
        if str(request.url).startswith(GMAIL_BASE) and "/messages/" in request.url.path:
            return self._handle_gmail_get(request)
        if str(request.url).startswith(CALENDAR_BASE) and request.url.path.endswith("/calendarList"):
            return self._handle_calendar_list(request)
        if str(request.url).startswith(CALENDAR_BASE) and request.url.path.endswith("/events"):
            if request.method == "POST":
                return self._handle_insert_event(request)
            return self._handle_events(request)
        if str(request.url).startswith(CALENDAR_BASE) and "/events/" in request.url.path:
            if request.method == "GET":
                return self._handle_get_event(request)
            if request.method == "DELETE":
                return self._handle_delete_event(request)
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

    def _handle_gmail_list(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        failure = self._gmail_list_failures.get(access_token)
        if failure is not None:
            status, raise_connect_error = failure
            if raise_connect_error:
                raise httpx.ConnectError("connection refused", request=request)
            if status is not None:
                return httpx.Response(status, json={"error": {"message": "forced failure"}})
        max_results = int(request.url.params.get("maxResults") or 25)
        # Plan 09-09: `pageToken` is this fake's own start offset into
        # `all_messages`, encoded as a plain integer string -- real paging
        # (not the old single-page-only "more" sentinel, which never
        # advanced) so `list_message_ids(max_total=...)` can be proven
        # for real against more than one page.
        page_token = request.url.params.get("pageToken")
        start = int(page_token) if page_token else 0
        all_messages = self._gmail_messages.get(access_token, [])
        page = all_messages[start : start + max_results]
        body: dict[str, Any] = {"messages": page}
        next_start = start + max_results
        if next_start < len(all_messages):
            body["nextPageToken"] = str(next_start)
        return httpx.Response(200, json=body)

    def _handle_send_as(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        entries = self._send_as.get(access_token, [])
        return httpx.Response(200, json={"sendAs": entries})

    def _handle_gmail_get(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        segments = request.url.path.split("/")
        message_id = urllib.parse.unquote(segments[-1])
        store = self._gmail_metadata if request.url.params.get("format") == "metadata" else self._gmail_full
        message = store.get((access_token, message_id))
        if message is None:
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(200, json=message)

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

    def _handle_insert_event(self, request: httpx.Request) -> httpx.Response:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        segments = request.url.path.split("/")
        calendar_id = urllib.parse.unquote(segments[-2])
        body = json.loads(request.content.decode("utf-8"))
        event_id = f"evt-{len(self.inserted) + 1}"
        stored = {"id": event_id, **body}
        self.events.setdefault((access_token, calendar_id), []).append(stored)
        self.inserted.append({"access_token": access_token, "calendar_id": calendar_id, "body": body})
        return httpx.Response(200, json=stored)

    def _event_lookup(
        self, request: httpx.Request
    ) -> tuple[str, str, str, "int | None"]:
        auth = request.headers.get("authorization", "")
        access_token = auth.removeprefix("Bearer ")
        segments = request.url.path.split("/")
        event_id = urllib.parse.unquote(segments[-1])
        calendar_id = urllib.parse.unquote(segments[-3])
        forced_status = self._write_failures.get((access_token, calendar_id, event_id))
        return access_token, calendar_id, event_id, forced_status

    def _handle_get_event(self, request: httpx.Request) -> httpx.Response:
        access_token, calendar_id, event_id, forced_status = self._event_lookup(request)
        if forced_status is not None:
            return httpx.Response(forced_status, json={"error": {"message": "forced failure"}})
        events = self.events.get((access_token, calendar_id), [])
        match = next((e for e in events if e.get("id") == event_id), None)
        if match is None:
            return httpx.Response(404, json={"error": {"message": "not found"}})
        return httpx.Response(200, json=match)

    def _handle_delete_event(self, request: httpx.Request) -> httpx.Response:
        access_token, calendar_id, event_id, forced_status = self._event_lookup(request)
        if forced_status is not None:
            return httpx.Response(forced_status, json={"error": {"message": "forced failure"}})
        events = self.events.get((access_token, calendar_id), [])
        remaining = [e for e in events if e.get("id") != event_id]
        if len(remaining) == len(events):
            return httpx.Response(404, json={"error": {"message": "not found"}})
        self.events[(access_token, calendar_id)] = remaining
        self.deleted_event_ids.append(event_id)
        return httpx.Response(200, json={})
