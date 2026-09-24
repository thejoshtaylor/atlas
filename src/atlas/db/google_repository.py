"""Phase 9's own repository module -- the plain value objects, the closed
sets, and the `GoogleAccountRepository` Protocol this plan's plugin,
token service, and env builder all read through. Kept beside (not inside)
`db/repository.py`, the same reasoning `db/google_models.py` gives for
staying out of `db/models.py`: this phase owns its own repository surface
without every other phase's `db/repository.py` edit needing to route
around it.

Every method here follows `PluginRepository`'s own "the caller validates,
this layer only writes" convention (`db/repository.py::PluginRepository`'s
docstring) -- `set_calendar_access` is the one exception, since D-05's
closed set (`CALENDAR_ACCESS`) is this repository's own invariant to
enforce, not a route's.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, Sequence

# D-05: a newly discovered calendar is off until the operator enables it.
# `set_calendar_access` refuses any value outside this tuple.
CALENDAR_ACCESS: tuple[str, ...] = ("off", "read_only", "read_write")

# GOOG-12: an account's own degraded-reachability status. `"ok"` is every
# account's status until a refresh fails -- `GoogleTokenService` (plan
# 09-01 Task 3) is what ever writes `"needs_relink"`/`"unreachable"`.
ACCOUNT_STATUSES: tuple[str, ...] = ("ok", "needs_relink", "unreachable")


@dataclass(frozen=True)
class GoogleOAuthClient:
    """The one deployer-supplied OAuth client (D-01) -- a plain value
    object, matching `Plugin`/`Credential`'s own "no caller outside
    `src/atlas/db/` holds a SQLAlchemy-mapped instance" convention.
    `client_secret_ciphertext` is exactly what `encrypt_credential`
    returns; nothing in this dataclass, nor any repository built on it,
    ever holds the decrypted secret.
    """

    client_id: str
    client_secret_ciphertext: bytes
    key_version: int
    updated_at: datetime


@dataclass(frozen=True)
class GoogleCalendar:
    """One calendar a linked account's `calendarList` reported -- off
    (D-05) until the operator enables it, with `access` one of
    `CALENDAR_ACCESS`."""

    id: int
    account_id: int
    google_calendar_id: str
    name: str
    is_primary: bool
    can_write: bool
    access: str


@dataclass(frozen=True)
class GoogleAccount:
    """One linked Google account (D-01, D-03) with its own calendars,
    ordered by `google_calendar_id` the same way `list_accounts` orders
    accounts by `label` -- deterministic, never database-cursor order.
    `refresh_token_ciphertext` is exactly what `encrypt_credential`
    returns; nothing in this dataclass, nor any repository built on it,
    ever holds the decrypted token.
    """

    id: int
    label: str
    email: str
    refresh_token_ciphertext: bytes
    key_version: int
    granted_scopes: str
    is_default: bool
    status: str
    status_detail: str | None
    refresh_token_expires_at: datetime | None
    linked_at: datetime
    calendars: tuple[GoogleCalendar, ...]


class GoogleAccountRepository(Protocol):
    """What Google account storage must answer for this plan: the OAuth
    client, linked accounts and their calendars, and the account status
    transitions `GoogleTokenService` (Task 3) writes.

    Structurally satisfied by `PostgresGoogleAccountRepository`
    (`db/google_postgres.py`, real) and `FakeGoogleAccountRepository`
    (`tests/google_repo_fakes.py`), the same dependency-injection-over-
    subclassing convention every other repository in this project uses.
    """

    async def get_oauth_client(self) -> "GoogleOAuthClient | None":
        """The one stored OAuth client, or `None` if a deployer has not
        yet pasted one in (D-01)."""
        ...

    async def set_oauth_client(
        self,
        *,
        client_id: str,
        client_secret_ciphertext: bytes,
        key_version: int,
        updated_by_user_id: int | None,
        updated_at: datetime,
    ) -> GoogleOAuthClient:
        """Write the singleton OAuth client row, inserting it the first
        time and overwriting it on every later save -- the same singleton
        convention `PostgresPolicyRepository.set_mode` already uses for
        `safety_policy`."""
        ...

    async def list_accounts(self) -> Sequence[GoogleAccount]:
        """Every linked account, each with its own calendars, ordered by
        label -- what a request naming no account reads every one of
        (D-04)."""
        ...

    async def get_account(self, account_id: int) -> "GoogleAccount | None":
        """One linked account by id, with its calendars, or `None`."""
        ...

    async def insert_account(
        self,
        *,
        label: str,
        email: str,
        refresh_token_ciphertext: bytes,
        key_version: int,
        granted_scopes: str,
        refresh_token_expires_at: "datetime | None",
        linked_by_user_id: int | None,
        linked_at: datetime,
    ) -> GoogleAccount:
        """Link a new account -- the same repository method the OAuth
        callback (plan 09-03) will call, so nothing in this plan is a
        throwaway path (09-01-PLAN.md's own objective). Starts with no
        calendars and `status="ok"`."""
        ...

    async def add_calendars(
        self,
        account_id: int,
        calendars: Sequence[tuple[str, str, bool, bool]],
        *,
        discovered_at: datetime,
    ) -> None:
        """Record calendars `calendarList.list` discovered for
        `account_id` -- each a `(google_calendar_id, name, is_primary,
        can_write)` tuple. Every new row starts `access="off"` (D-05); a
        `google_calendar_id` already present for this account is left
        untouched, never reset back to `off` or duplicated
        (`uq_google_calendars_account_calendar`)."""
        ...

    async def set_calendar_access(
        self, calendar_id: int, access: str, *, updated_at: datetime
    ) -> None:
        """Set one calendar's own access (D-05). Refuses `access` outside
        `CALENDAR_ACCESS` with `ValueError` -- this repository's own
        invariant, not a route's, since a value outside the closed set
        must never reach the row at all."""
        ...

    async def set_account_status(
        self, account_id: int, status: str, detail: "str | None", at: datetime
    ) -> None:
        """Record `account_id`'s own degraded-reachability status
        (GOOG-12) -- `GoogleTokenService` (Task 3) is the one caller that
        writes `"needs_relink"`/`"unreachable"`; a success after either
        writes `"ok"` again."""
        ...
