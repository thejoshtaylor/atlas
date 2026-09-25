"""`FakePendingActionRepository` -- an in-memory double for
`atlas.db.pending_action_repository.PendingActionRepository`, following
`tests/google_repo_fakes.py`'s own in-process, no-network convention.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from typing import Any

from atlas.db.pending_action_repository import PendingAction


class FakePendingActionRepository:
    def __init__(self) -> None:
        self._rows: dict[int, PendingAction] = {}
        self._next_id = 1

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
        for existing in list(self._rows.values()):
            if existing.source == source and existing.status == "awaiting":
                self._rows[existing.id] = replace(
                    existing, status="superseded", resolved_at=created_at
                )
        row = PendingAction(
            id=self._next_id,
            source=source,
            action=action,
            tool_name=tool_name,
            arguments=arguments,
            readback=readback,
            status="awaiting",
            created_at=created_at,
            expires_at=expires_at,
            resolved_at=None,
            result_detail=None,
        )
        self._rows[row.id] = row
        self._next_id += 1
        return row

    async def get(self, action_id: int) -> "PendingAction | None":
        return self._rows.get(action_id)

    async def claim_for_confirmation(self, action_id: int, now: datetime) -> "PendingAction | None":
        row = self._rows.get(action_id)
        if row is None or row.status != "awaiting" or row.expires_at <= now:
            return None
        claimed = replace(row, status="confirmed")
        self._rows[action_id] = claimed
        return claimed

    async def resolve(self, action_id: int, status: str, detail: "str | None", at: datetime) -> None:
        row = self._rows.get(action_id)
        if row is None:
            return
        self._rows[action_id] = replace(row, status=status, result_detail=detail, resolved_at=at)
