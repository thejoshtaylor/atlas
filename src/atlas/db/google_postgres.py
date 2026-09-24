"""`GoogleAccountRepository`, implemented against a real Postgres --
following `PostgresWakeEventRepository`'s own shape (`db/postgres.py`) and
reusing that module's `_to_naive_utc`/`_to_aware_utc` naive-UTC boundary
convention rather than re-deriving it (D-01, `db/postgres.py`'s own
module-wide convention comment).
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from sqlalchemy import select
from sqlalchemy.ext.asyncio import async_sessionmaker

from atlas.db.google_models import (
    GoogleAccountRow,
    GoogleCalendarRow,
    GoogleOAuthClientRow,
)
from atlas.db.google_repository import (
    CALENDAR_ACCESS,
    GoogleAccount,
    GoogleCalendar,
    GoogleOAuthClient,
)
from atlas.db.postgres import _to_aware_utc, _to_naive_utc

# `google_oauth_client` is a single-row table -- `id` is always this value,
# the same singleton convention `PostgresPolicyRepository`'s own
# `_SINGLETON_POLICY_ID` establishes.
_SINGLETON_OAUTH_CLIENT_ID = 1


def _calendar_from_row(row: GoogleCalendarRow) -> GoogleCalendar:
    return GoogleCalendar(
        id=row.id,
        account_id=row.account_id,
        google_calendar_id=row.google_calendar_id,
        name=row.name,
        is_primary=row.is_primary,
        can_write=row.can_write,
        access=row.access,
    )


def _account_from_row(row: GoogleAccountRow, calendar_rows: Sequence[GoogleCalendarRow]) -> GoogleAccount:
    return GoogleAccount(
        id=row.id,
        label=row.label,
        email=row.email,
        refresh_token_ciphertext=row.refresh_token_ciphertext,
        key_version=row.key_version,
        granted_scopes=row.granted_scopes,
        is_default=row.is_default,
        status=row.status,
        status_detail=row.status_detail,
        refresh_token_expires_at=_to_aware_utc(row.refresh_token_expires_at),
        linked_at=_to_aware_utc(row.linked_at),
        calendars=tuple(
            _calendar_from_row(c)
            for c in sorted(calendar_rows, key=lambda c: c.google_calendar_id)
        ),
    )


def _oauth_client_from_row(row: GoogleOAuthClientRow) -> GoogleOAuthClient:
    return GoogleOAuthClient(
        client_id=row.client_id,
        client_secret_ciphertext=row.client_secret_ciphertext,
        key_version=row.key_version,
        updated_at=_to_aware_utc(row.updated_at),
    )


class PostgresGoogleAccountRepository:
    """`GoogleAccountRepository`, implemented against a real Postgres.

    Structurally satisfies `atlas.db.google_repository.GoogleAccountRepository`
    (a `typing.Protocol`) -- there is no base class to inherit from,
    matching every `Postgres*Repository` class in `db/postgres.py`.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def get_oauth_client(self) -> "GoogleOAuthClient | None":
        async with self._sessionmaker() as session:
            row = await session.get(GoogleOAuthClientRow, _SINGLETON_OAUTH_CLIENT_ID)
            return _oauth_client_from_row(row) if row is not None else None

    async def set_oauth_client(
        self,
        *,
        client_id: str,
        client_secret_ciphertext: bytes,
        key_version: int,
        updated_by_user_id: "int | None",
        updated_at: datetime,
    ) -> GoogleOAuthClient:
        naive_updated_at = _to_naive_utc(updated_at)
        async with self._sessionmaker() as session:
            row = await session.get(GoogleOAuthClientRow, _SINGLETON_OAUTH_CLIENT_ID)
            if row is None:
                row = GoogleOAuthClientRow(
                    id=_SINGLETON_OAUTH_CLIENT_ID,
                    client_id=client_id,
                    client_secret_ciphertext=client_secret_ciphertext,
                    key_version=key_version,
                    updated_at=naive_updated_at,
                    updated_by_user_id=updated_by_user_id,
                )
                session.add(row)
            else:
                row.client_id = client_id
                row.client_secret_ciphertext = client_secret_ciphertext
                row.key_version = key_version
                row.updated_at = naive_updated_at
                row.updated_by_user_id = updated_by_user_id
            await session.commit()
            await session.refresh(row)
            return _oauth_client_from_row(row)

    async def _calendars_by_account(
        self, session, account_ids: Sequence[int]
    ) -> dict[int, list[GoogleCalendarRow]]:
        if not account_ids:
            return {}
        rows = (
            await session.execute(
                select(GoogleCalendarRow).where(GoogleCalendarRow.account_id.in_(account_ids))
            )
        ).scalars().all()
        by_account: dict[int, list[GoogleCalendarRow]] = {aid: [] for aid in account_ids}
        for row in rows:
            by_account.setdefault(row.account_id, []).append(row)
        return by_account

    async def list_accounts(self) -> list[GoogleAccount]:
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(select(GoogleAccountRow).order_by(GoogleAccountRow.label))
            ).scalars().all()
            by_account = await self._calendars_by_account(session, [r.id for r in rows])
            return [_account_from_row(row, by_account.get(row.id, [])) for row in rows]

    async def get_account(self, account_id: int) -> "GoogleAccount | None":
        async with self._sessionmaker() as session:
            row = await session.get(GoogleAccountRow, account_id)
            if row is None:
                return None
            calendar_rows = (
                await session.execute(
                    select(GoogleCalendarRow).where(GoogleCalendarRow.account_id == account_id)
                )
            ).scalars().all()
            return _account_from_row(row, calendar_rows)

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
        naive_linked_at = _to_naive_utc(linked_at)
        async with self._sessionmaker() as session:
            row = GoogleAccountRow(
                label=label,
                email=email,
                refresh_token_ciphertext=refresh_token_ciphertext,
                key_version=key_version,
                granted_scopes=granted_scopes,
                is_default=False,
                status="ok",
                status_detail=None,
                status_at=None,
                refresh_token_expires_at=(
                    _to_naive_utc(refresh_token_expires_at)
                    if refresh_token_expires_at is not None
                    else None
                ),
                linked_at=naive_linked_at,
                updated_at=naive_linked_at,
                linked_by_user_id=linked_by_user_id,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _account_from_row(row, [])

    async def add_calendars(
        self,
        account_id: int,
        calendars: Sequence[tuple[str, str, bool, bool]],
        *,
        discovered_at: datetime,
    ) -> None:
        naive_discovered_at = _to_naive_utc(discovered_at)
        async with self._sessionmaker() as session:
            existing_rows = (
                await session.execute(
                    select(GoogleCalendarRow).where(GoogleCalendarRow.account_id == account_id)
                )
            ).scalars().all()
            existing_ids = {row.google_calendar_id for row in existing_rows}
            for google_calendar_id, name, is_primary, can_write in calendars:
                if google_calendar_id in existing_ids:
                    # D-05: a calendar already discovered is left untouched
                    # -- its `access` (possibly enabled by the operator) is
                    # never reset back to `off` by a later discovery run.
                    continue
                session.add(
                    GoogleCalendarRow(
                        account_id=account_id,
                        google_calendar_id=google_calendar_id,
                        name=name,
                        is_primary=is_primary,
                        can_write=can_write,
                        access="off",
                        discovered_at=naive_discovered_at,
                        updated_at=naive_discovered_at,
                    )
                )
            await session.commit()

    async def set_calendar_access(
        self, calendar_id: int, access: str, *, updated_at: datetime
    ) -> None:
        if access not in CALENDAR_ACCESS:
            raise ValueError(
                f"calendar access must be one of {CALENDAR_ACCESS!r}, got {access!r}"
            )
        async with self._sessionmaker() as session:
            row = await session.get(GoogleCalendarRow, calendar_id)
            if row is None:
                return
            row.access = access
            row.updated_at = _to_naive_utc(updated_at)
            await session.commit()

    async def set_account_status(
        self, account_id: int, status: str, detail: "str | None", at: datetime
    ) -> None:
        async with self._sessionmaker() as session:
            row = await session.get(GoogleAccountRow, account_id)
            if row is None:
                return
            naive_at = _to_naive_utc(at)
            row.status = status
            row.status_detail = detail
            row.status_at = naive_at
            row.updated_at = naive_at
            await session.commit()
