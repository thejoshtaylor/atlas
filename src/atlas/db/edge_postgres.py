"""`EdgeDeviceRepository`, implemented against a real Postgres (D-03,
Phase 10, plan 10-04).

Structurally satisfies `atlas.db.edge_repository.EdgeDeviceRepository` (a
`typing.Protocol`) -- built from `PostgresWakeEventRepository`'s own
shape (`db/postgres.py`), reusing that module's `_to_naive_utc`/
`_to_aware_utc` naive-UTC boundary convention rather than re-deriving it,
the same way `db/google_postgres.py` already does.
"""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import async_sessionmaker

from atlas.db.edge_models import EdgeDeviceRow
from atlas.db.edge_repository import EdgeDevice
from atlas.db.postgres import _to_aware_utc, _to_naive_utc


def _edge_device_from_row(row: EdgeDeviceRow) -> EdgeDevice:
    return EdgeDevice(
        id=row.id,
        name=row.name,
        created_at=_to_aware_utc(row.created_at),
        created_by_user_id=row.created_by_user_id,
        revoked_at=_to_aware_utc(row.revoked_at),
        last_connected_at=_to_aware_utc(row.last_connected_at),
    )


class PostgresEdgeDeviceRepository:
    """`EdgeDeviceRepository`, implemented against a real Postgres.

    There is no base class to inherit from, matching every
    `Postgres*Repository` class in `db/postgres.py`/`db/google_postgres.py`.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def get_active_by_token_hash(self, token_hash: str) -> "EdgeDevice | None":
        async with self._sessionmaker() as session:
            row = (
                await session.execute(
                    select(EdgeDeviceRow).where(EdgeDeviceRow.token_hash == token_hash)
                )
            ).scalar_one_or_none()
            if row is None or row.revoked_at is not None:
                return None
            return _edge_device_from_row(row)

    async def create_device(
        self, *, name: str, token_hash: str, created_by_user_id: int, created_at: datetime
    ) -> EdgeDevice:
        async with self._sessionmaker() as session:
            row = EdgeDeviceRow(
                name=name,
                token_hash=token_hash,
                created_by_user_id=created_by_user_id,
                created_at=_to_naive_utc(created_at),
                revoked_at=None,
                last_connected_at=None,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _edge_device_from_row(row)

    async def list_devices(self) -> "list[EdgeDevice]":
        async with self._sessionmaker() as session:
            rows = (
                await session.execute(
                    select(EdgeDeviceRow).order_by(EdgeDeviceRow.created_at, EdgeDeviceRow.id)
                )
            ).scalars().all()
            return [_edge_device_from_row(row) for row in rows]

    async def revoke_device(self, device_id: int, *, revoked_at: datetime) -> bool:
        """One `UPDATE ... WHERE id = :id AND revoked_at IS NULL` --
        its own row count decides the return value, so two concurrent
        revokes of the same device cannot both report success (matching
        `PostgresGoogleAccountRepository.consume_oauth_state`'s own
        rowcount-decides-truth shape)."""
        async with self._sessionmaker() as session:
            statement = (
                update(EdgeDeviceRow)
                .where(EdgeDeviceRow.id == device_id, EdgeDeviceRow.revoked_at.is_(None))
                .values(revoked_at=_to_naive_utc(revoked_at))
            )
            result = await session.execute(statement)
            await session.commit()
            return bool(result.rowcount == 1)

    async def mark_connected(self, device_id: int, *, at: datetime) -> None:
        async with self._sessionmaker() as session:
            await session.execute(
                update(EdgeDeviceRow)
                .where(EdgeDeviceRow.id == device_id)
                .values(last_connected_at=_to_naive_utc(at))
            )
            await session.commit()
