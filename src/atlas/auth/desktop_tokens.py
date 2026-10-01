"""Mac device token issue, hash, and the `/ws/desktop` authentication
dependency (Phase 14, D-01, D-02, T-14-01, T-14-03).

This is a deliberate copy of `auth/edge_tokens.py`, not an import of it. A
Mac token and an edge device token are separate bearer credentials with
separate tables, and a function shared between them would blur that. It is
also what makes PAIR-06 true by construction: a Mac token hash is never in
the edge table, so `/ws/edge` cannot find it.

`require_desktop_device` reads the token only from the `Authorization`
header. Missing header, missing repository, unknown token and revoked token
all raise the same `WebSocketException` before the socket is accepted. Nothing
here logs the token or its hash.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import WebSocketException
from starlette import status
from starlette.websockets import WebSocket

from atlas.db.desktop_repository import DesktopDevice


def issue_desktop_token() -> str:
    """One opaque, unguessable Mac token (256 bits)."""
    return secrets.token_urlsafe(32)


def hash_desktop_token(token: str) -> str:
    """The storage form of a Mac token: SHA-256 hex. The repository is only
    ever handed this form."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def bearer_token_from_header(value: str | None) -> str | None:
    """The bearer token from an `Authorization` header value, or `None`
    when it is absent or not a `Bearer` credential. Never reads a query
    parameter: proxies log URLs and shells keep them in history."""
    if value is None:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _refused() -> WebSocketException:
    """The one refusal for every failure case. The caller never learns
    which case it hit."""
    return WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


async def require_desktop_device(websocket: WebSocket) -> DesktopDevice:
    """Refuse the connection before accept unless the `Authorization`
    header carries a bearer token that hashes to an active Mac."""
    token = bearer_token_from_header(websocket.headers.get("authorization"))
    if token is None:
        raise _refused()

    desktop_device_repo = getattr(websocket.app.state, "desktop_device_repo", None)
    if desktop_device_repo is None:
        raise _refused()

    device = await desktop_device_repo.get_active_by_token_hash(hash_desktop_token(token))
    if device is None:
        raise _refused()
    return device
