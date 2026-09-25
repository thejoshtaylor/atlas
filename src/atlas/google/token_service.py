"""`GoogleTokenService`: the one place a refresh token or the OAuth client
secret is ever decrypted (T-09-01, D-03) -- decrypts, posts to the token
endpoint, caches the resulting access token by account id, and never logs
either decrypted value.

Task 3 (GOOG-12): every outcome also writes the account's own
degraded-reachability status through `GoogleAccountRepository.
set_account_status` -- `needs_relink` for a revoked grant (or no OAuth
client configured at all), `unreachable` for any other failure, and `ok`
again the moment a call after either succeeds. Every log line here names
the account's own label and the exception's own type -- never a token or
a secret value (T-09-03).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import httpx

from atlas_mcp.google_api import GoogleApiError, GoogleGrantRevokedError, refresh_access_token
from atlas.config import SecurityConfig
from atlas.crypto.credentials import decrypt_credential
from atlas.db.google_repository import GoogleAccount, GoogleAccountRepository

logger = logging.getLogger("atlas.google.token_service")

_NO_OAUTH_CLIENT_DETAIL = "no google oauth client is configured"


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

    `clock` (a `time.time`-shaped float clock, for cache expiry math) and
    `now` (a `datetime`-shaped wall clock, for the status rows this
    writes) are both injectable -- matching `WorkflowScheduler`/
    `PluginManager`'s own `sleep=`/`clock=` precedent -- so a test drives
    both expiry and status timestamps with no wall-clock wait.
    """

    def __init__(
        self,
        repo: GoogleAccountRepository,
        security: SecurityConfig,
        http_client: httpx.AsyncClient,
        *,
        clock: Callable[[], float] = time.time,
        now: Callable[[], datetime] = lambda: datetime.now(timezone.utc),
    ) -> None:
        self._repo = repo
        self._security = security
        self._http_client = http_client
        self._clock = clock
        self._now = now
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
            return await self._fail(
                account, "needs_relink", _NO_OAUTH_CLIENT_DETAIL, "GoogleOAuthClientMissing"
            )

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
        except GoogleGrantRevokedError as exc:
            return await self._fail(account, "needs_relink", str(exc), type(exc).__name__)
        except (GoogleApiError, httpx.HTTPError) as exc:
            return await self._fail(account, "unreachable", str(exc), type(exc).__name__)

        expires_at = now + float(body.get("expires_in", 3600))
        result = AccessToken(token=body["access_token"], expires_at=expires_at, unreachable_reason=None)
        self._cache[account.id] = result
        await self._repo.set_account_status(account.id, "ok", None, self._now())
        return result

    async def _fail(
        self, account: GoogleAccount, reason: str, detail: str, exc_type_name: str
    ) -> AccessToken:
        """The shared failure body every non-success branch above uses:
        log the account's own label and the exception's own type name
        only -- never its message or `detail`, either of which could echo
        back something Google's own error body quoted -- cache and return
        the degraded `AccessToken`, and persist `reason` as this
        account's own status.
        """
        logger.warning(
            "google account %r could not be refreshed (%s): %s",
            account.label,
            exc_type_name,
            reason,
        )
        result = AccessToken(token=None, expires_at=None, unreachable_reason=reason)
        self._cache[account.id] = result
        await self._repo.set_account_status(account.id, reason, detail, self._now())
        return result

    def forget(self, account_id: int) -> None:
        """Drop `account_id`'s cached access token, if any -- so the next
        call refreshes rather than serving a token this caller has reason
        to believe is stale."""
        self._cache.pop(account_id, None)
