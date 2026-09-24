"""`FakeGoogleAccountRepository`: an in-memory `GoogleAccountRepository`
(`atlas.db.google_repository`) -- the Postgres-free implementation this
phase's own tests use for anything that does not specifically need to
prove a real migration or a real Postgres round-trip, the direct sibling
of `tests/conftest.py::FakePluginRepository`.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from atlas.db.google_repository import CALENDAR_ACCESS, GoogleAccount, GoogleCalendar, GoogleOAuthClient


class FakeGoogleAccountRepository:
    """A plain-Python mirror of `PostgresGoogleAccountRepository`'s own
    contract -- constructed empty; every write method mutates this
    instance's own dicts directly, the same "test builds state through
    the repository's own methods" convention `FakePluginRepository`
    already uses."""

    def __init__(self) -> None:
        self._oauth_client: GoogleOAuthClient | None = None
        self._accounts: dict[int, GoogleAccount] = {}
        self._calendars: dict[int, GoogleCalendar] = {}
        self._next_account_id = 1
        self._next_calendar_id = 1

    async def get_oauth_client(self) -> "GoogleOAuthClient | None":
        return self._oauth_client

    async def set_oauth_client(
        self,
        *,
        client_id: str,
        client_secret_ciphertext: bytes,
        key_version: int,
        updated_by_user_id: "int | None",
        updated_at: datetime,
    ) -> GoogleOAuthClient:
        self._oauth_client = GoogleOAuthClient(
            client_id=client_id,
            client_secret_ciphertext=client_secret_ciphertext,
            key_version=key_version,
            updated_at=updated_at,
        )
        return self._oauth_client

    def _calendars_for(self, account_id: int) -> tuple[GoogleCalendar, ...]:
        return tuple(
            sorted(
                (c for c in self._calendars.values() if c.account_id == account_id),
                key=lambda c: c.google_calendar_id,
            )
        )

    def _with_calendars(self, account: GoogleAccount) -> GoogleAccount:
        return GoogleAccount(
            id=account.id,
            label=account.label,
            email=account.email,
            refresh_token_ciphertext=account.refresh_token_ciphertext,
            key_version=account.key_version,
            granted_scopes=account.granted_scopes,
            is_default=account.is_default,
            status=account.status,
            status_detail=account.status_detail,
            refresh_token_expires_at=account.refresh_token_expires_at,
            linked_at=account.linked_at,
            calendars=self._calendars_for(account.id),
        )

    async def list_accounts(self) -> list[GoogleAccount]:
        return [
            self._with_calendars(account)
            for account in sorted(self._accounts.values(), key=lambda a: a.label)
        ]

    async def get_account(self, account_id: int) -> "GoogleAccount | None":
        account = self._accounts.get(account_id)
        return self._with_calendars(account) if account is not None else None

    async def insert_account(
        self,
        *,
        label: str,
        email: str,
        refresh_token_ciphertext: bytes,
        key_version: int,
        granted_scopes: str,
        refresh_token_expires_at: "datetime | None",
        linked_by_user_id: "int | None",
        linked_at: datetime,
    ) -> GoogleAccount:
        account = GoogleAccount(
            id=self._next_account_id,
            label=label,
            email=email,
            refresh_token_ciphertext=refresh_token_ciphertext,
            key_version=key_version,
            granted_scopes=granted_scopes,
            is_default=False,
            status="ok",
            status_detail=None,
            refresh_token_expires_at=refresh_token_expires_at,
            linked_at=linked_at,
            calendars=(),
        )
        self._accounts[account.id] = account
        self._next_account_id += 1
        return account

    async def add_calendars(
        self,
        account_id: int,
        calendars: Sequence[tuple[str, str, bool, bool]],
        *,
        discovered_at: datetime,
    ) -> None:
        existing_ids = {c.google_calendar_id for c in self._calendars_for(account_id)}
        for google_calendar_id, name, is_primary, can_write in calendars:
            if google_calendar_id in existing_ids:
                continue
            calendar = GoogleCalendar(
                id=self._next_calendar_id,
                account_id=account_id,
                google_calendar_id=google_calendar_id,
                name=name,
                is_primary=is_primary,
                can_write=can_write,
                access="off",
            )
            self._calendars[calendar.id] = calendar
            self._next_calendar_id += 1

    async def set_calendar_access(
        self, calendar_id: int, access: str, *, updated_at: datetime
    ) -> None:
        if access not in CALENDAR_ACCESS:
            raise ValueError(
                f"calendar access must be one of {CALENDAR_ACCESS!r}, got {access!r}"
            )
        calendar = self._calendars.get(calendar_id)
        if calendar is None:
            return
        self._calendars[calendar_id] = GoogleCalendar(
            id=calendar.id,
            account_id=calendar.account_id,
            google_calendar_id=calendar.google_calendar_id,
            name=calendar.name,
            is_primary=calendar.is_primary,
            can_write=calendar.can_write,
            access=access,
        )

    async def set_account_status(
        self, account_id: int, status: str, detail: "str | None", at: datetime
    ) -> None:
        account = self._accounts.get(account_id)
        if account is None:
            return
        self._accounts[account_id] = GoogleAccount(
            id=account.id,
            label=account.label,
            email=account.email,
            refresh_token_ciphertext=account.refresh_token_ciphertext,
            key_version=account.key_version,
            granted_scopes=account.granted_scopes,
            is_default=account.is_default,
            status=status,
            status_detail=detail,
            refresh_token_expires_at=account.refresh_token_expires_at,
            linked_at=account.linked_at,
            calendars=account.calendars,
        )
