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
from datetime import datetime
from typing import Any, Protocol

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
