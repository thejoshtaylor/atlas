"""Plain async functions over a caller-supplied `httpx.AsyncClient` for the
Gmail endpoints plan 09-08 reads -- no Google SDK, matching
`google_api.py`'s own "one focused async function per concern, client
passed in" style. Every function here issues a GET, never anything else
(T-09-47): this module reads a mailbox, it never changes one.

Reuses `google_api.py`'s own error vocabulary (`GoogleApiError`,
`GoogleAuthError`) rather than inventing a second one -- every non-2xx
response below is raised as one of those, never returned as a silent
empty result. `GMAIL_BASE` and the error-message parsing are duplicated
here rather than imported from `google_account_api.py`
(`google_account_api.py`'s own docstring already states why: each of
these small API modules stays independent, sharing only the two error
classes).
"""

from __future__ import annotations

import email.header
import email.utils
import urllib.parse
from typing import Any

import httpx

from atlas_mcp.google_api import GoogleApiError, GoogleAuthError

GMAIL_BASE = "https://gmail.googleapis.com/gmail/v1"


def _error_message(response: httpx.Response) -> str:
    """The boundary's own words, nothing reworded -- the identical pattern
    `google_api.py::_error_message` already establishes."""
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


async def list_message_ids(
    client: httpx.AsyncClient,
    *,
    access_token: str,
    query: str,
    max_results: int = 25,
    max_total: "int | None" = None,
) -> "tuple[list[dict[str, Any]], bool]":
    """`{"id", "threadId"}` for messages matching `query` (Gmail's own
    search grammar, sent exactly as given) -- newest first, Gmail's own
    default order.

    `max_total=None` (the default, and every pre-09-09 caller's own
    behavior, D-14): up to `max_results` messages, one request, never
    paged by this function itself -- the second return value is True when
    Gmail reports more results exist beyond this one page
    (`nextPageToken` present), naming that to the caller instead of
    following it.

    `max_total` given (plan 09-09's own `learn_style`, `SENT_MESSAGES_TO_SCAN`):
    follows `nextPageToken`, `max_results` messages per page, until at
    least `max_total` messages are collected or Gmail reports no further
    page -- returns at most `max_total` messages, and the second value is
    True only when Gmail still reports more beyond that cut."""
    messages: "list[dict[str, Any]]" = []
    page_token: str | None = None
    while True:
        params: dict[str, Any] = {"q": query, "maxResults": max_results}
        if page_token:
            params["pageToken"] = page_token
        response = await client.get(
            f"{GMAIL_BASE}/users/me/messages",
            headers={"Authorization": f"Bearer {access_token}"},
            params=params,
        )
        _raise_for_status(response)
        body = response.json()
        messages.extend(body.get("messages") or [])
        page_token = body.get("nextPageToken")
        if max_total is None:
            return messages, bool(page_token)
        if not page_token or len(messages) >= max_total:
            return messages[:max_total], bool(page_token) and len(messages) > max_total


async def list_send_as(client: httpx.AsyncClient, *, access_token: str) -> "list[dict[str, Any]]":
    """Every send-as identity this account carries (`users.settings.sendAs.list`)
    -- plan 09-09's own source for the default send-as entry's Gmail
    signature (D-22), the one field this module otherwise never reads."""
    response = await client.get(
        f"{GMAIL_BASE}/users/me/settings/sendAs",
        headers={"Authorization": f"Bearer {access_token}"},
    )
    _raise_for_status(response)
    body = response.json()
    return list(body.get("sendAs") or [])


async def get_message_metadata(
    client: httpx.AsyncClient, *, access_token: str, message_id: str
) -> dict[str, Any]:
    """One message's `From`/`Subject`/`Date` headers only (`format=metadata`)
    -- never the body, so listing or searching mail never pulls body text
    over the wire at all."""
    encoded_id = urllib.parse.quote(message_id, safe="")
    response = await client.get(
        f"{GMAIL_BASE}/users/me/messages/{encoded_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"format": "metadata", "metadataHeaders": ["From", "Subject", "Date"]},
    )
    _raise_for_status(response)
    return response.json()


async def get_message_full(
    client: httpx.AsyncClient, *, access_token: str, message_id: str
) -> dict[str, Any]:
    """One message's full payload, headers and body parts alike
    (`format=full`) -- the one call that ever reads body text, made only
    by `handle_gmail_fetch_body` after the operator's own request for one
    specific message (D-18)."""
    encoded_id = urllib.parse.quote(message_id, safe="")
    response = await client.get(
        f"{GMAIL_BASE}/users/me/messages/{encoded_id}",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"format": "full"},
    )
    _raise_for_status(response)
    return response.json()


def header_value(message: dict[str, Any], name: str) -> str:
    """One header's own value off a `format=metadata`/`format=full`
    message, RFC 2047 decoded (`email.header.make_header`) -- an
    undecodable value falls back to the raw string rather than raising.
    Empty string when `message` carries no header by that name."""
    headers = (message.get("payload") or {}).get("headers") or []
    for header in headers:
        if str(header.get("name", "")).lower() == name.lower():
            raw = str(header.get("value", ""))
            try:
                return str(email.header.make_header(email.header.decode_header(raw)))
            except (UnicodeDecodeError, LookupError, ValueError):
                return raw
    return ""


def parse_from(value: str) -> "tuple[str, str]":
    """`(display_name, address)` from a decoded `From` header value
    (`email.utils.parseaddr`) -- a header with no display name returns an
    empty `display_name` and keeps the address."""
    name, address = email.utils.parseaddr(value)
    return name, address
