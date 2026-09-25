"""The pending-action row (`db/google_models.py::PendingActionRow`) and the
`PendingActionRepository` Protocol code stores a proposal's exact
executing call through (D-08, D-09).

`PostgresPendingActionRepository` lands in plan 09-04 Task 2. Task 1 builds
against `tests/pending_action_fakes.py::FakePendingActionRepository` only,
matching the "protocol first, fake first, real implementation next task"
shape this project's other repositories already established (compare
`db/repository.py`'s protocols against `db/postgres.py`'s implementations).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Protocol

from sqlalchemy import update
from sqlalchemy.ext.asyncio import async_sessionmaker

from atlas.db.google_models import PendingActionRow

# The closed set `status` may hold -- code enforces this set, the column
# itself is `Text` (`db/google_models.py::PendingActionRow`'s own doctrine,
# matching every other status-shaped column in this project).
PENDING_ACTION_STATUSES: tuple[str, ...] = (
    "awaiting",
    "confirmed",
    "executed",
    "failed",
    "cancelled",
    "expired",
    "superseded",
)


@dataclass(frozen=True)
class PendingAction:
    """One code-held pending action -- the exact executing tool name and
    arguments a later confirm re-runs verbatim (D-08), never re-derived
    from model output at confirm time."""

    id: int
    source: str
    action: str
    tool_name: str
    arguments: dict[str, Any]
    readback: str
    status: str
    created_at: datetime
    expires_at: datetime
    resolved_at: "datetime | None"
    result_detail: "str | None"


class PendingActionRepository(Protocol):
    """Structural contract every implementation (fake or Postgres)
    satisfies -- there is no base class to inherit from, matching every
    other repository Protocol in this project (`db/repository.py`).
    """

    async def create(
        self,
        *,
        source: str,
        action: str,
        tool_name: str,
        arguments: dict[str, Any],
        readback: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> PendingAction:
        """Insert one new `awaiting` row, superseding this same source's
        own earlier `awaiting` row (if any) in the same transaction (T-09-25:
        a stale ask can never be confirmed once a newer one exists for the
        same source)."""
        ...

    async def get(self, action_id: int) -> "PendingAction | None": ...

    async def claim_for_confirmation(self, action_id: int, now: datetime) -> "PendingAction | None":
        """One conditional update, `awaiting` -> `confirmed`, only while
        `expires_at > now` -- succeeds at most once per row (T-09-25's
        single-use claim), returning `None` on a second attempt or after
        expiry."""
        ...

    async def resolve(
        self, action_id: int, status: str, detail: "str | None", at: datetime
    ) -> None:
        """Record a row's terminal status, its own outcome detail, and
        when -- `status` must be one of `PENDING_ACTION_STATUSES`."""
        ...


# `db/postgres.py`'s own convention (its module docstring's "the convention
# going forward"): every datetime this module hands to the database goes
# through `_to_naive_utc` first (the column is `TIMESTAMP WITHOUT TIME
# ZONE`); every datetime it reads back comes out through `_to_aware_utc`.
# Not imported from `db/postgres.py` -- this module has no other dependency
# on that one, and duplicating two five-line pure functions is cheaper than
# adding one.


def _to_naive_utc(dt: datetime) -> datetime:
    if dt.tzinfo is not None:
        dt = dt.astimezone(timezone.utc)
    return dt.replace(tzinfo=None)


def _to_aware_utc(dt: "datetime | None") -> "datetime | None":
    if dt is None or dt.tzinfo is not None:
        return dt
    return dt.replace(tzinfo=timezone.utc)


def _pending_action_from_row(row: PendingActionRow) -> PendingAction:
    return PendingAction(
        id=row.id,
        source=row.source,
        action=row.action,
        tool_name=row.tool_name,
        arguments=row.arguments,
        readback=row.readback,
        status=row.status,
        created_at=_to_aware_utc(row.created_at),
        expires_at=_to_aware_utc(row.expires_at),
        resolved_at=_to_aware_utc(row.resolved_at),
        result_detail=row.result_detail,
    )


class PostgresPendingActionRepository:
    """`PendingActionRepository`, implemented against a real Postgres.

    Structurally satisfies `PendingActionRepository` (a `typing.Protocol`)
    -- there is no base class to inherit from, matching every
    `Postgres*Repository` class in `db/postgres.py`.
    """

    def __init__(self, sessionmaker: async_sessionmaker) -> None:
        self._sessionmaker = sessionmaker

    async def create(
        self,
        *,
        source: str,
        action: str,
        tool_name: str,
        arguments: dict[str, Any],
        readback: str,
        created_at: datetime,
        expires_at: datetime,
    ) -> PendingAction:
        naive_created = _to_naive_utc(created_at)
        naive_expires = _to_naive_utc(expires_at)
        async with self._sessionmaker() as session:
            # T-09-25: supersede this source's own earlier `awaiting` row
            # and the insert below in the one transaction this `async with`
            # block commits together -- a stale ask can never be confirmed
            # once a newer one exists for the same source.
            await session.execute(
                update(PendingActionRow)
                .where(PendingActionRow.source == source, PendingActionRow.status == "awaiting")
                .values(status="superseded", resolved_at=naive_created)
            )
            row = PendingActionRow(
                source=source,
                action=action,
                tool_name=tool_name,
                arguments=arguments,
                readback=readback,
                status="awaiting",
                created_at=naive_created,
                expires_at=naive_expires,
                resolved_at=None,
                result_detail=None,
            )
            session.add(row)
            await session.commit()
            await session.refresh(row)
            return _pending_action_from_row(row)

    async def get(self, action_id: int) -> "PendingAction | None":
        async with self._sessionmaker() as session:
            row = await session.get(PendingActionRow, action_id)
            return _pending_action_from_row(row) if row is not None else None

    async def claim_for_confirmation(self, action_id: int, now: datetime) -> "PendingAction | None":
        naive_now = _to_naive_utc(now)
        async with self._sessionmaker() as session:
            # T-09-25: one conditional `UPDATE ... WHERE ... RETURNING` --
            # Postgres evaluates the `WHERE` and the write as one atomic
            # operation, so this succeeds at most once per row, and never
            # once `expires_at` has passed. The same
            # `PostgresAccountRepository.claim_invite`-shaped single-UPDATE
            # compare-and-swap `db/postgres.py` already establishes.
            result = await session.execute(
                update(PendingActionRow)
                .where(
                    PendingActionRow.id == action_id,
                    PendingActionRow.status == "awaiting",
                    PendingActionRow.expires_at > naive_now,
                )
                .values(status="confirmed")
                .returning(PendingActionRow)
            )
            row = result.scalar_one_or_none()
            await session.commit()
            return _pending_action_from_row(row) if row is not None else None

    async def resolve(self, action_id: int, status: str, detail: "str | None", at: datetime) -> None:
        naive_at = _to_naive_utc(at)
        async with self._sessionmaker() as session:
            row = await session.get(PendingActionRow, action_id)
            if row is not None:
                row.status = status
                row.result_detail = detail
                row.resolved_at = naive_at
                await session.commit()
