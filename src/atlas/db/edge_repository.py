"""`EdgeDevice` and the `EdgeDeviceRepository` Protocol (D-03, Phase 10).

Plan 10-02 built the shape only: the dataclass a token lookup returns, and
the one method `require_edge_device` (`auth/edge_tokens.py`) needs at
connect time. Plan 10-04 (this revision) adds the Postgres implementation
(`db/edge_postgres.py`, a new `edge_devices` table mirroring `InviteRow`)
and the admin create/list/revoke methods this Protocol gains below --
the same "this plan builds the shape, a later plan builds the storage"
split `WakeEventRepository` (`db/repository.py`) already models for a
different resource.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class EdgeDevice:
    """One row an `EdgeDeviceRepository` would return -- never the token
    itself, only what `require_edge_device` needs to admit or refuse a
    connection and what a later admin screen (plan 10-04) needs to list.
    """

    id: int
    name: str
    created_at: datetime
    created_by_user_id: int | None
    revoked_at: datetime | None
    last_connected_at: datetime | None


class EdgeDeviceRepository(Protocol):
    """What `require_edge_device` needs at connect time
    (`get_active_by_token_hash`), plus the admin create/list/revoke/
    connect-tracking methods plan 10-04's routes (`routes/edge_devices.py`)
    and `/ws/edge` need. `None` from `get_active_by_token_hash` covers
    both an unknown token hash and a revoked device: the caller
    (`require_edge_device`) never learns which, the same generic-refusal
    discipline `auth/dependencies.py`'s `_unauthenticated_error` already
    holds to (T-10-01)."""

    async def get_active_by_token_hash(self, token_hash: str) -> EdgeDevice | None: ...

    async def create_device(
        self, *, name: str, token_hash: str, created_by_user_id: int, created_at: datetime
    ) -> EdgeDevice: ...

    async def list_devices(self) -> "list[EdgeDevice]": ...

    async def revoke_device(self, device_id: int, *, revoked_at: datetime) -> bool: ...

    async def mark_connected(self, device_id: int, *, at: datetime) -> None: ...
