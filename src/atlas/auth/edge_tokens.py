"""Device token issue, hash, and the `/ws/edge` authentication dependency
(D-03, T-10-01, T-10-02, T-10-03).

`issue_edge_token`/`hash_edge_token` are local to this module, kept apart
from `auth/tokens.py`'s `issue_refresh_token`/`hash_refresh_token` for the
same reason `routes/accounts.py`'s `_hash_invite_token` is local rather
than imported: an edge device token, a refresh token, and an invite token
are three different bearer credentials with three different tables and
three different lifetimes, and sharing one function name across them would
blur that they are not interchangeable.

`require_edge_device` reads the bearer token only from the `Authorization`
header (T-10-03) and never logs the token or its hash (T-10-02). A
missing header, a missing repository, or an unknown/revoked token all
raise the identical `WebSocketException(code=status.WS_1008_POLICY_
VIOLATION)` before the socket is ever accepted -- the generic refusal
never says which case happened, the same `_unauthenticated_error`
discipline `auth/dependencies.py` already holds to for the cookie-session
path.
"""

from __future__ import annotations

import hashlib
import secrets

from fastapi import WebSocketException
from starlette import status
from starlette.websockets import WebSocket

from atlas.db.edge_repository import EdgeDevice


def issue_edge_token() -> str:
    """Generate one opaque, unguessable device token -- the same shape
    `auth/tokens.py::issue_refresh_token` uses, kept as a separate
    function in this module (see the module docstring) rather than
    imported."""
    return secrets.token_urlsafe(32)


def hash_edge_token(token: str) -> str:
    """The storage form of a device token -- SHA-256 hex, the one form
    `EdgeDeviceRepository.get_active_by_token_hash` is ever handed."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def bearer_token_from_header(value: str | None) -> str | None:
    """Extract the bearer token from an `Authorization` header value, or
    `None` when `value` is absent or not a `Bearer` credential. Never
    reads a query parameter (T-10-03) -- a URL is logged by proxies and
    kept in browser/shell history, and a bearer credential does not belong
    there."""
    if value is None:
        return None
    scheme, _, token = value.partition(" ")
    if scheme.lower() != "bearer" or not token:
        return None
    return token


def _refused() -> WebSocketException:
    """One generic refusal for every case `require_edge_device` can hit --
    a missing header, a missing repository, an unknown token, or a
    revoked one. The caller never learns which (T-10-01), the same
    discipline `auth/dependencies.py::_unauthenticated_error` already
    holds to for the cookie-session path."""
    return WebSocketException(code=status.WS_1008_POLICY_VIOLATION)


async def require_edge_device(websocket: WebSocket) -> EdgeDevice:
    """Refuse the connection before accept unless the `Authorization`
    header carries a bearer token that hashes to an active, unrevoked
    device (T-10-01). Never logs the token or its hash (T-10-02)."""
    token = bearer_token_from_header(websocket.headers.get("authorization"))
    if token is None:
        raise _refused()

    edge_device_repo = getattr(websocket.app.state, "edge_device_repo", None)
    if edge_device_repo is None:
        raise _refused()

    device = await edge_device_repo.get_active_by_token_hash(hash_edge_token(token))
    if device is None:
        raise _refused()
    return device
