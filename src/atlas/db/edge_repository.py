"""`EdgeDevice` and the `EdgeDeviceRepository` Protocol (D-03, Phase 10).

This plan builds the shape only: the dataclass a token lookup returns, and
the one method `require_edge_device` (`auth/edge_tokens.py`) needs at
connect time. Plan 10-04 adds the Postgres implementation (a new
`edge_devices` table, mirroring `InviteRow`) and the admin create/list/
revoke methods -- this Protocol is deliberately narrow until then, the
same "this plan builds the shape, a later plan builds the storage" split
`WakeEventRepository` (`db/repository.py`) already models for a different
resource.
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
    """What `require_edge_device` needs at connect time -- the one method
    this plan requires. `None` covers both an unknown token hash and a
    revoked device: the caller (`require_edge_device`) never learns which,
    the same generic-refusal discipline `auth/dependencies.py`'s
    `_unauthenticated_error` already holds to (T-10-01)."""

    async def get_active_by_token_hash(self, token_hash: str) -> EdgeDevice | None: ...
