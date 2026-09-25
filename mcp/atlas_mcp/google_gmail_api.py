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
    client: httpx.AsyncClient, *, access_token: str, query: str, max_results: int = 25
) -> "tuple[list[dict[str, Any]], bool]":
    """`{"id", "threadId"}` for up to `max_results` messages matching
    `query` (Gmail's own search grammar, sent exactly as given) -- newest
    first, Gmail's own default order. The second return value is True
    when Gmail reports more results exist beyond this page
    (`nextPageToken` present) -- this function never fetches a second
    page itself (D-14, `has_more` names it to the caller instead)."""
    response = await client.get(
        f"{GMAIL_BASE}/users/me/messages",
        headers={"Authorization": f"Bearer {access_token}"},
        params={"q": query, "maxResults": max_results},
    )
    _raise_for_status(response)
    body = response.json()
    messages = list(body.get("messages") or [])
    has_more = bool(body.get("nextPageToken"))
    return messages, has_more


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
