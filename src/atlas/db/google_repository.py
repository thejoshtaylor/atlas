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
from typing import Mapping, Protocol, Sequence

# D-05: a newly discovered calendar is off until the operator enables it.
# `set_calendar_access` refuses any value outside this tuple.
CALENDAR_ACCESS: tuple[str, ...] = ("off", "read_only", "read_write")

# GOOG-12: an account's own degraded-reachability status. `"ok"` is every
# account's status until a refresh fails -- `GoogleTokenService` (plan
# 09-01 Task 3) is what ever writes `"needs_relink"`/`"unreachable"`.
ACCOUNT_STATUSES: tuple[str, ...] = ("ok", "needs_relink", "unreachable")

# Plan 09-09 (D-19, D-20): one account's own learned-style status. A row
# with no style learned yet answers `"not_learned"` (never a missing row --
# `get_style`/`get_style_by_label` synthesize this value object rather
# than returning `None` for an account that simply has not been learned
# yet); `"learning"` is set synchronously by the route that schedules
# `learn_style`, before the background task itself ever runs.
STYLE_STATUSES: tuple[str, ...] = ("not_learned", "learning", "ready", "failed")


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
class GoogleOAuthState:
    """One in-flight OAuth `state` value (D-02, T-09-10) -- stored only as
    a SHA-256 hash (`state_hash`), never the raw value a browser carries.
    `used_at` is `None` until `consume_oauth_state` claims it; every
    consumer of this dataclass treats a state with `used_at` already set
    as spent."""

    id: int
    state_hash: str
    label: str
    redirect_uri: str
    created_by_user_id: int | None
    created_at: datetime
    expires_at: datetime
    used_at: datetime | None


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


@dataclass(frozen=True)
class GoogleAccountStyle:
    """One account's own learned writing style (D-19, D-20) and Gmail
    signature (D-22) -- a plain-text value object, never Fernet
    ciphertext: the profile, samples, and signature are not credentials,
    the admin must be able to read and edit the profile (D-19), and
    Postgres is already their trust boundary (`google_account_style`'s own
    docstring in `db/google_models.py`).

    `get_style`/`get_style_by_label` synthesize this with `status="not_learned"`
    for an account with no style row yet -- never `None` for a known
    account, so a caller never has to special-case "no row" separately
    from "learned, but empty"."""

    account_id: int
    profile: str
    samples: tuple[str, ...]
    signature_html: "str | None"
    signature_text: "str | None"
    status: str
    status_detail: "str | None"
    messages_scanned: int
    learned_at: "datetime | None"


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

    # --- Plan 09-03: linking, editing, and unlinking -----------------

    async def create_oauth_state(
        self,
        state_hash: str,
        label: str,
        redirect_uri: str,
        created_by_user_id: "int | None",
        created_at: datetime,
        expires_at: datetime,
    ) -> GoogleOAuthState:
        """Record one in-flight OAuth `state` (T-09-10) -- `state_hash` is
        already the SHA-256 hex digest of the raw value the browser
        carries; this repository never sees the raw value."""
        ...

    async def consume_oauth_state(self, state_hash: str, now: datetime) -> "GoogleOAuthState | None":
        """Claim `state_hash` exactly once: one conditional
        `UPDATE ... WHERE used_at IS NULL AND expires_at > now RETURNING`,
        so a state can never be consumed twice and an expired state is
        never consumed at all. `None` for an unknown, already-used, or
        expired state -- the caller cannot tell those three apart, and
        must not (T-09-10: no information about *why* a state failed is
        ever useful to whoever is presenting it)."""
        ...

    async def purge_expired_oauth_states(self, now: datetime) -> None:
        """Delete every OAuth state whose `expires_at` has passed --
        called at the start of every `POST /api/google/oauth/start`
        (09-03-PLAN.md's own action text), so this table never grows
        without bound from abandoned flows."""
        ...

    async def find_account_by_email(self, email: str) -> "GoogleAccount | None":
        """The linked account whose Google address is `email`, with its
        calendars, or `None` -- how the OAuth callback recognizes a
        re-link (D-03's own "the callback recognises an existing account
        by its Google address")."""
        ...

    async def update_account_link(
        self,
        account_id: int,
        refresh_token_ciphertext: bytes,
        key_version: int,
        granted_scopes: str,
        refresh_token_expires_at: "datetime | None",
        at: datetime,
    ) -> GoogleAccount:
        """Re-link an existing account: replace its ciphertext, granted
        scopes, and refresh-token expiry, and set `status` back to
        `"ok"` -- the account's own label and calendars are left
        untouched."""
        ...

    async def update_account(
        self,
        account_id: int,
        *,
        label: "str | None" = None,
        is_default: "bool | None" = None,
        at: datetime,
    ) -> "GoogleAccount | None":
        """Update `account_id`'s own `label` and/or `is_default` (each
        `None` means "leave alone") and return the updated account with
        its calendars, or `None` if `account_id` does not exist.

        Setting `is_default=True` clears every *other* account's own
        `is_default` flag in the same transaction (D-04) -- the caller
        never has to make a second call to un-default the previous
        account, and there is never a moment where two rows are both
        default at once."""
        ...

    async def delete_account(self, account_id: int) -> None:
        """Delete `account_id` and, through `ondelete="CASCADE"`, its
        calendars and style row -- a no-op if the account does not
        exist. The caller is responsible for revoking the refresh token
        at Google first; this method only removes the row."""
        ...

    async def get_calendar(self, calendar_id: int) -> "GoogleCalendar | None":
        """One calendar by its own id, or `None` -- the 404 the calendar
        routes read to decide "unknown calendar," and the read
        `set_calendar_access`'s own caller uses to check `can_write`
        before allowing `read_write` (D-05)."""
        ...

    async def update_calendar_names(
        self, account_id: int, names: Mapping[str, str], at: datetime
    ) -> None:
        """Refresh the stored `name` for every one of `account_id`'s own
        calendars that `names` (keyed by `google_calendar_id`) names --
        never touches `access`, and never touches a calendar `names` does
        not mention (`POST .../calendars/refresh`'s own "leaves every
        stored access value untouched")."""
        ...

    # --- Plan 09-09: style learning and drafting ----------------------

    async def get_style(self, account_id: int) -> GoogleAccountStyle:
        """`account_id`'s own learned style -- `status="not_learned"` and
        an empty profile/samples/signature when no style row exists yet
        (an account never learned, or one this plan's own migration
        created no row for at link time), never `None`."""
        ...

    async def get_style_by_label(self, label: str) -> "GoogleAccountStyle | None":
        """The same value `get_style` would return, found by the
        account's own label rather than its id -- `None` only when no
        account carries `label` at all (an unlearned account with that
        label still answers `status="not_learned"`, exactly like
        `get_style`)."""
        ...

    async def set_style_status(
        self, account_id: int, status: str, detail: "str | None", at: datetime
    ) -> None:
        """Record `account_id`'s own style `status` (one of
        `STYLE_STATUSES`) and `detail` -- the route that schedules
        `learn_style` calls this with `"learning"` synchronously, before
        the background task itself ever runs; `learn_style` calls this
        with `"failed"` on any failure, never raising instead."""
        ...

    async def save_learned_style(
        self,
        account_id: int,
        *,
        profile: str,
        samples: "Sequence[str]",
        signature_html: "str | None",
        signature_text: "str | None",
        messages_scanned: int,
        learned_at: datetime,
    ) -> GoogleAccountStyle:
        """Store a successful `learn_style` result -- sets `status="ready"`
        and clears `status_detail`, upserting the one row this account's
        style ever has."""
        ...

    async def update_style_profile(self, account_id: int, profile: str, at: datetime) -> None:
        """The admin's own edit of `account_id`'s profile (D-19) -- leaves
        `status`, `samples`, and the signature untouched; the next draft
        reads this profile back."""
        ...
