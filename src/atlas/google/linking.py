"""`complete_link`: turns an OAuth `code` into a linked, encrypted Google
account (T-09-01, T-09-15) -- the one place a code exchange, the granted-
scope check, and the profile/calendar reads all happen together, called
from `routes/google_accounts.py`'s own `GET /api/google/oauth/callback`.

Every failure here is a `LinkError` carrying one code from the closed
`LINK_ERROR_CODES` set (T-09-16) -- the route turns that into
`RedirectResponse("/google?link_error=<code>", 303)`, never the raw
exception text.
"""

from __future__ import annotations

import logging
from datetime import datetime, timedelta

import httpx

from atlas_mcp.google_account_api import exchange_code, get_gmail_profile, list_calendars, revoke_token
from atlas_mcp.google_api import GoogleApiError
from atlas_mcp.google_tools import REQUIRED_SCOPES

from atlas.config import SecurityConfig
from atlas.crypto.credentials import decrypt_credential, encrypt_credential
from atlas.db.google_repository import GoogleAccount, GoogleAccountRepository, GoogleOAuthState
from atlas.google.token_service import GoogleTokenService

logger = logging.getLogger("atlas.google.linking")

LINK_ERROR_CODES: tuple[str, ...] = (
    "denied",
    "state_invalid",
    "client_missing",
    "exchange_failed",
    "no_refresh_token",
    "scopes_missing",
    "profile_failed",
    "label_taken",
)


class LinkError(Exception):
    """One failure from `LINK_ERROR_CODES` -- the route's own
    `link_error=<code>` redirect reads `.code` directly, never `str(exc)`
    (T-09-16: no raw exception text ever reaches the browser's URL bar)."""

    def __init__(self, code: str) -> None:
        assert code in LINK_ERROR_CODES, f"unknown link error code: {code!r}"
        self.code = code
        super().__init__(code)


async def complete_link(
    *,
    repo: GoogleAccountRepository,
    security: SecurityConfig,
    http_client: httpx.AsyncClient,
    token_service: GoogleTokenService,
    state: GoogleOAuthState,
    code: str,
    now: datetime,
) -> GoogleAccount:
    """Exchange `code` (already bound to `state`'s own `redirect_uri`),
    require every `REQUIRED_SCOPES` entry in the granted scope, read the
    linked address, and store the result -- re-linking an existing address
    (`find_account_by_email`) rather than creating a second row for it.
    A calendar-discovery failure after a successful link is logged and the
    link stands (calendars can be refreshed from the account page later).
    """
    oauth_client = await repo.get_oauth_client()
    if oauth_client is None:
        raise LinkError("client_missing")

    client_secret = decrypt_credential(
        oauth_client.client_secret_ciphertext, oauth_client.key_version, security
    )

    try:
        token_body = await exchange_code(
            http_client,
            client_id=oauth_client.client_id,
            client_secret=client_secret,
            code=code,
            redirect_uri=state.redirect_uri,
        )
    except GoogleApiError as exc:
        logger.warning("google code exchange failed: %s", type(exc).__name__)
        raise LinkError("exchange_failed") from exc

    refresh_token = token_body.get("refresh_token")
    if not refresh_token:
        raise LinkError("no_refresh_token")

    granted_scope = token_body.get("scope") or ""
    granted = set(granted_scope.split())
    if not set(REQUIRED_SCOPES).issubset(granted):
        await revoke_token(http_client, refresh_token)
        raise LinkError("scopes_missing")

    access_token = token_body["access_token"]
    try:
        email = await get_gmail_profile(http_client, access_token)
    except GoogleApiError as exc:
        logger.warning("google profile read failed after linking: %s", type(exc).__name__)
        raise LinkError("profile_failed") from exc

    refresh_token_expires_at: "datetime | None" = None
    if "refresh_token_expires_in" in token_body:
        refresh_token_expires_at = now + timedelta(
            seconds=float(token_body["refresh_token_expires_in"])
        )

    ciphertext, key_version = encrypt_credential(refresh_token, security)

    existing = await repo.find_account_by_email(email)
    if existing is not None:
        account = await repo.update_account_link(
            existing.id, ciphertext, key_version, granted_scope, refresh_token_expires_at, now
        )
        token_service.forget(existing.id)
    else:
        accounts = await repo.list_accounts()
        if any(a.label == state.label for a in accounts):
            # The label was free when `POST /oauth/start` checked it, but
            # another admin's link (or a re-link) claimed it before this
            # callback ran -- a real, if narrow, race window between the
            # two requests.
            raise LinkError("label_taken")
        account = await repo.insert_account(
            label=state.label,
            email=email,
            refresh_token_ciphertext=ciphertext,
            key_version=key_version,
            granted_scopes=granted_scope,
            refresh_token_expires_at=refresh_token_expires_at,
            linked_by_user_id=state.created_by_user_id,
            linked_at=now,
        )

    try:
        calendars = await list_calendars(http_client, access_token)
    except GoogleApiError as exc:
        logger.warning("calendar discovery failed after linking %r: %s", email, type(exc).__name__)
        return account

    await repo.add_calendars(
        account.id,
        [(c["id"], c["name"], c["primary"], c["can_write"]) for c in calendars],
        discovered_at=now,
    )
    return account
