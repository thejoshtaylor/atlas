"""`GoogleTokenService`: the one place a refresh token or the OAuth client
secret is ever decrypted (T-09-01, D-03) -- decrypts, posts to the token
endpoint, caches the resulting access token by account id, and never logs
either decrypted value.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

import httpx

from atlas_mcp.google_api import GoogleApiError, refresh_access_token
from atlas.config import SecurityConfig
from atlas.crypto.credentials import decrypt_credential
from atlas.db.google_repository import GoogleAccount, GoogleAccountRepository


@dataclass(frozen=True)
class AccessToken:
    """One resolved access token -- `token`/`expires_at` are both `None`
    exactly when `unreachable_reason` names why nothing could be issued."""

    token: str | None
    expires_at: float | None
    unreachable_reason: str | None


class GoogleTokenService:
    """Turns a linked account's encrypted refresh token into a short-lived
    access token, caching by account id so a call within
    `min_validity_s` of a still-fresh cached token posts nothing at all.

    `clock` is injectable (matching `WorkflowScheduler`/`PluginManager`'s
    own `sleep=`/`clock=` precedent) so a test drives expiry with no
    wall-clock wait.
    """

    def __init__(
        self,
        repo: GoogleAccountRepository,
        security: SecurityConfig,
        http_client: httpx.AsyncClient,
        *,
        clock: Callable[[], float] = time.time,
    ) -> None:
        self._repo = repo
        self._security = security
        self._http_client = http_client
        self._clock = clock
        self._cache: dict[int, AccessToken] = {}

    async def access_token_for(
        self, account: GoogleAccount, *, min_validity_s: float = 1800.0
    ) -> AccessToken:
        """The account's current access token, refreshing it first when
        no cached token has at least `min_validity_s` left."""
        cached = self._cache.get(account.id)
        now = self._clock()
        if cached is not None and cached.token is not None and cached.expires_at is not None:
            if cached.expires_at - now >= min_validity_s:
                return cached

        oauth_client = await self._repo.get_oauth_client()
        if oauth_client is None:
            result = AccessToken(
                token=None,
                expires_at=None,
                unreachable_reason="needs_relink",
            )
            self._cache[account.id] = result
            return result

        client_secret = decrypt_credential(
            oauth_client.client_secret_ciphertext, oauth_client.key_version, self._security
        )
        refresh_token = decrypt_credential(
            account.refresh_token_ciphertext, account.key_version, self._security
        )
        try:
            body = await refresh_access_token(
                self._http_client,
                client_id=oauth_client.client_id,
                client_secret=client_secret,
                refresh_token=refresh_token,
            )
        except (GoogleApiError, httpx.HTTPError) as exc:
            result = AccessToken(
                token=None,
                expires_at=None,
                unreachable_reason=_reason_for(exc),
            )
            self._cache[account.id] = result
            return result

        expires_at = now + float(body.get("expires_in", 3600))
        result = AccessToken(token=body["access_token"], expires_at=expires_at, unreachable_reason=None)
        self._cache[account.id] = result
        return result

    def forget(self, account_id: int) -> None:
        """Drop `account_id`'s cached access token, if any -- so the next
        call refreshes rather than serving a token this caller has reason
        to believe is stale."""
        self._cache.pop(account_id, None)


def _reason_for(exc: Exception) -> str:
    """`GoogleGrantRevokedError` (a `GoogleApiError` subclass) names
    "needs_relink"; every other failure (a non-`invalid_grant`
    `GoogleApiError`, or a transport-level `httpx.HTTPError`) names
    "unreachable". Kept as a free function rather than importing
    `GoogleGrantRevokedError` at call sites more than once."""
    from atlas_mcp.google_api import GoogleGrantRevokedError

    if isinstance(exc, GoogleGrantRevokedError):
        return "needs_relink"
    return "unreachable"
