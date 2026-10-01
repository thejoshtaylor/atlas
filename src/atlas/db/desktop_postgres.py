"""`DesktopDeviceRepository`, implemented against a real Postgres (Phase 14,
plan 14-04; D-01, D-06, D-14, D-28).

Structurally satisfies `atlas.db.desktop_repository.DesktopDeviceRepository`.
Built from `db/edge_postgres.py`, reusing the naive-UTC boundary helpers in
`db/postgres.py`. The database holds the invariants: a partial unique index
refuses a second default Mac, and another refuses a second active Mac whose
name differs only in case. This class turns those refusals into the
exceptions the Protocol names.

This module never logs token material.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.exc import DBAPIError, IntegrityError
from sqlalchemy.ext.asyncio import async_sessionmaker

from atlas.db.desktop_models import DesktopDeviceRow
from atlas.db.desktop_repository import (
    DesktopDefaultConflict,
    DesktopDevice,
    DesktopDeviceChanges,
    DesktopDeviceNameTaken,
)
from atlas.db.postgres import _to_aware_utc, _to_naive_utc

_NAME_INDEX = "uq_desktop_devices_active_name"
_DEFAULT_INDEX = "uq_desktop_devices_single_default"


def _desktop_device_from_row(row: DesktopDeviceRow) -> DesktopDevice:
    return DesktopDevice(
        id=row.id,
        name=row.name,
        created_at=_to_aware_utc(row.created_at),
        created_by_user_id=row.created_by_user_id,
        revoked_at=_to_aware_utc(row.revoked_at),
        last_seen_at=_to_aware_utc(row.last_seen_at),
        edge_device_id=row.edge_device_id,
        is_default=row.is_default,
    )


class PostgresDesktopDeviceRepository:
    """`DesktopDeviceRepository`, implemented against a real Postgres.

    No base class, matching every `Postgres*Repository` in `db/`.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def get_active_by_token_hash(self, token_hash: str) -> "DesktopDevice | None":
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(DesktopDeviceRow).where(DesktopDeviceRow.token_hash == token_hash)
                )
            ).scalar_one_or_none()
            if row is None or row.revoked_at is not None:
                return None
            return _desktop_device_from_row(row)

    async def get_device(self, device_id: int) -> "DesktopDevice | None":
        async with self._sessionmaker() as session:
            row = await session.get(DesktopDeviceRow, device_id)
            return _desktop_device_from_row(row) if row is not None else None

    async def list_devices(self) -> "list[DesktopDevice]":
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(
                    select(DesktopDeviceRow).order_by(DesktopDeviceRow.created_at, DesktopDeviceRow.id)
                )
            ).scalars().all()
            return [_desktop_device_from_row(row) for row in rows]

    async def create_device(
        self, *, name: str, token_hash: str, created_by_user_id: int, created_at: datetime
    ) -> DesktopDevice:
        async with self._sessionmaker() as session:
            row = DesktopDeviceRow(
                name=name,
                token_hash=token_hash,
                created_by_user_id=created_by_user_id,
                created_at=_to_naive_utc(created_at),
                revoked_at=None,
                last_seen_at=None,
                edge_device_id=None,
                is_default=False,
            )
            session.add(row)
            try:
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                if _NAME_INDEX in str(exc):
                    raise DesktopDeviceNameTaken(name) from exc
                raise
            await session.refresh(row)
            return _desktop_device_from_row(row)

    async def update_device(
        self, device_id: int, changes: DesktopDeviceChanges, *, at: datetime
    ) -> "DesktopDevice | None":
        """One session, one transaction. A "set default" write first locks
        every active row in id order, so two concurrent default writers
        queue behind each other instead of deadlocking. Then it clears the
        other rows' flag and sets this one's (D-14)."""
        async with self._sessionmaker() as session:
            try:
                if changes.is_default is True:
                    await session.execute(
                        select(DesktopDeviceRow.id)
                        .where(DesktopDeviceRow.revoked_at.is_(None))
                        .order_by(DesktopDeviceRow.id)
                        .with_for_update()
                    )
                row = (
                    await session.execute(
                        select(DesktopDeviceRow)
                        .where(DesktopDeviceRow.id == device_id)
                        .with_for_update()
                        .execution_options(populate_existing=True)
                    )
                ).scalar_one_or_none()
                if row is None or row.revoked_at is not None:
                    await session.rollback()
                    return None
                if changes.is_default is True:
                    await session.execute(
                        update(DesktopDeviceRow)
                        .where(DesktopDeviceRow.id != device_id, DesktopDeviceRow.is_default.is_(True))
                        .values(is_default=False)
                    )
                    row.is_default = True
                elif changes.is_default is False:
                    row.is_default = False
                if changes.name is not None:
                    row.name = changes.name
                if changes.set_edge_device:
                    row.edge_device_id = changes.edge_device_id
                await session.commit()
            except IntegrityError as exc:
                await session.rollback()
                text = str(exc)
                if _NAME_INDEX in text:
                    raise DesktopDeviceNameTaken(changes.name or "") from exc
                if _DEFAULT_INDEX in text:
                    raise DesktopDefaultConflict() from exc
                raise
            except DBAPIError as exc:
                await session.rollback()
                if "deadlock detected" in str(exc):
                    raise DesktopDefaultConflict() from exc
                raise
            await session.refresh(row)
            return _desktop_device_from_row(row)

    async def revoke_device(self, device_id: int, *, revoked_at: datetime) -> bool:
        """One `UPDATE ... WHERE id = :id AND revoked_at IS NULL`. Its row
        count decides the return value, so two concurrent revokes of the
        same Mac cannot both report success. The same statement clears
        `is_default`, so a revoked Mac is never the default."""
        async with self._sessionmaker() as session:
            result = await session.execute(
                update(DesktopDeviceRow)
                .where(DesktopDeviceRow.id == device_id, DesktopDeviceRow.revoked_at.is_(None))
                .values(revoked_at=_to_naive_utc(revoked_at), is_default=False)
            )
            await session.commit()
            return bool(result.rowcount == 1)

    async def mark_seen(self, device_id: int, *, at: datetime) -> None:
        async with self._sessionmaker() as session:
            await session.execute(
                update(DesktopDeviceRow)
                .where(DesktopDeviceRow.id == device_id)
                .values(last_seen_at=_to_naive_utc(at))
            )
            await session.commit()
