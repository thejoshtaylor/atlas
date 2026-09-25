"""`FakeGoogleAccountRepository`: an in-memory `GoogleAccountRepository`
(`atlas.db.google_repository`) -- the Postgres-free implementation this
phase's own tests use for anything that does not specifically need to
prove a real migration or a real Postgres round-trip, the direct sibling
of `tests/conftest.py::FakePluginRepository`.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Mapping, Sequence

from atlas.db.google_repository import (
    CALENDAR_ACCESS,
    GoogleAccount,
    GoogleAccountStyle,
    GoogleCalendar,
    GoogleOAuthClient,
    GoogleOAuthState,
)


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
        self._oauth_states: dict[int, GoogleOAuthState] = {}
        self._next_oauth_state_id = 1
        # Plan 09-09: one style value object per account id -- absent
        # entirely until the first `set_style_status`/`save_learned_style`/
        # `update_style_profile` call, matching
        # `PostgresGoogleAccountRepository`'s own "no row yet" case.
        self._styles: dict[int, GoogleAccountStyle] = {}

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
        state = GoogleOAuthState(
            id=self._next_oauth_state_id,
            state_hash=state_hash,
            label=label,
            redirect_uri=redirect_uri,
            created_by_user_id=created_by_user_id,
            created_at=created_at,
            expires_at=expires_at,
            used_at=None,
        )
        self._oauth_states[state.id] = state
        self._next_oauth_state_id += 1
        return state

    async def consume_oauth_state(self, state_hash: str, now: datetime) -> "GoogleOAuthState | None":
        for state in self._oauth_states.values():
            if state.state_hash == state_hash and state.used_at is None and state.expires_at > now:
                consumed = replace(state, used_at=now)
                self._oauth_states[state.id] = consumed
                return consumed
        return None

    async def purge_expired_oauth_states(self, now: datetime) -> None:
        expired_ids = [sid for sid, state in self._oauth_states.items() if state.expires_at < now]
        for sid in expired_ids:
            del self._oauth_states[sid]

    async def find_account_by_email(self, email: str) -> "GoogleAccount | None":
        for account in self._accounts.values():
            if account.email == email:
                return self._with_calendars(account)
        return None

    async def update_account_link(
        self,
        account_id: int,
        refresh_token_ciphertext: bytes,
        key_version: int,
        granted_scopes: str,
        refresh_token_expires_at: "datetime | None",
        at: datetime,
    ) -> GoogleAccount:
        account = self._accounts[account_id]
        updated = replace(
            account,
            refresh_token_ciphertext=refresh_token_ciphertext,
            key_version=key_version,
            granted_scopes=granted_scopes,
            refresh_token_expires_at=refresh_token_expires_at,
            status="ok",
            status_detail=None,
        )
        self._accounts[account_id] = updated
        return self._with_calendars(updated)

    async def update_account(
        self,
        account_id: int,
        *,
        label: "str | None" = None,
        is_default: "bool | None" = None,
        at: datetime,
    ) -> "GoogleAccount | None":
        account = self._accounts.get(account_id)
        if account is None:
            return None
        if is_default is True:
            for other_id, other in list(self._accounts.items()):
                if other_id != account_id and other.is_default:
                    self._accounts[other_id] = replace(other, is_default=False)
            account = self._accounts[account_id]
        updated = replace(
            account,
            label=account.label if label is None else label,
            is_default=account.is_default if is_default is None else is_default,
        )
        self._accounts[account_id] = updated
        return self._with_calendars(updated)

    async def delete_account(self, account_id: int) -> None:
        self._accounts.pop(account_id, None)
        stale_calendar_ids = [cid for cid, c in self._calendars.items() if c.account_id == account_id]
        for cid in stale_calendar_ids:
            del self._calendars[cid]

    async def get_calendar(self, calendar_id: int) -> "GoogleCalendar | None":
        return self._calendars.get(calendar_id)

    async def update_calendar_names(
        self, account_id: int, names: Mapping[str, str], at: datetime
    ) -> None:
        for cid, calendar in list(self._calendars.items()):
            if calendar.account_id != account_id:
                continue
            new_name = names.get(calendar.google_calendar_id)
            if new_name is not None and new_name != calendar.name:
                self._calendars[cid] = replace(calendar, name=new_name)

    # --- Plan 09-09: style learning and drafting -----------------------

    def _not_learned_style(self, account_id: int) -> GoogleAccountStyle:
        return GoogleAccountStyle(
            account_id=account_id,
            profile="",
            samples=(),
            signature_html=None,
            signature_text=None,
            status="not_learned",
            status_detail=None,
            messages_scanned=0,
            learned_at=None,
        )

    async def get_style(self, account_id: int) -> GoogleAccountStyle:
        return self._styles.get(account_id) or self._not_learned_style(account_id)

    async def get_style_by_label(self, label: str) -> "GoogleAccountStyle | None":
        account = next((a for a in self._accounts.values() if a.label == label), None)
        if account is None:
            return None
        return await self.get_style(account.id)

    async def set_style_status(
        self, account_id: int, status: str, detail: "str | None", at: datetime
    ) -> None:
        current = await self.get_style(account_id)
        self._styles[account_id] = replace(current, status=status, status_detail=detail)

    async def save_learned_style(
        self,
        account_id: int,
        *,
        profile: str,
        samples: Sequence[str],
        signature_html: "str | None",
        signature_text: "str | None",
        messages_scanned: int,
        learned_at: datetime,
    ) -> GoogleAccountStyle:
        style = GoogleAccountStyle(
            account_id=account_id,
            profile=profile,
            samples=tuple(samples),
            signature_html=signature_html,
            signature_text=signature_text,
            status="ready",
            status_detail=None,
            messages_scanned=messages_scanned,
            learned_at=learned_at,
        )
        self._styles[account_id] = style
        return style

    async def update_style_profile(self, account_id: int, profile: str, at: datetime) -> None:
        current = await self.get_style(account_id)
        self._styles[account_id] = replace(current, profile=profile)
