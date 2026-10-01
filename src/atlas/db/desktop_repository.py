"""`DesktopDevice` and the `DesktopDeviceRepository` Protocol (Phase 14,
D-01, D-06, D-14, D-28).

The interface is complete here so the in-memory fake (`tests/desktop_
fakes.py`) and the Postgres class (plan 14-04) are built against one
contract. Both must behave the same.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol


@dataclass(frozen=True)
class DesktopDevice:
    """One paired Mac. Never carries the token or its hash."""

    id: int
    name: str
    created_at: datetime
    created_by_user_id: int | None
    revoked_at: datetime | None
    last_seen_at: datetime | None
    edge_device_id: int | None
    is_default: bool


@dataclass(frozen=True)
class DesktopDeviceChanges:
    """A partial update. `None` means "leave unchanged" for `name` and
    `is_default`. `edge_device_id` can be set to `None` on purpose, so
    `set_edge_device` says whether to write it at all."""

    name: str | None = None
    set_edge_device: bool = False
    edge_device_id: int | None = None
    is_default: bool | None = None


class DesktopDeviceNameTaken(Exception):
    """Another active Mac already uses this name. Names compare
    case-insensitively, and a revoked Mac frees its name (D-28)."""


class DesktopDefaultConflict(Exception):
    """A concurrent write made a different Mac the default (D-14)."""


class DesktopDeviceRepository(Protocol):
    """Rules every implementation must follow:

    - `get_active_by_token_hash` returns `None` for an unknown hash and for
      a revoked Mac. The caller never learns which (T-14-01).
    - `create_device` raises `DesktopDeviceNameTaken` when an active Mac has
      the name, compared with `casefold()` (D-28).
    - `update_device` returns `None` when the row does not exist or is
      revoked. It raises `DesktopDeviceNameTaken` or `DesktopDefaultConflict`.
      `is_default=True` clears every other row's flag in the same
      transaction (D-14).
    - `revoke_device` is one UPDATE where `revoked_at` is null. The row count
      decides the returned bool. The same statement sets `is_default` false.
    - `mark_seen` records `last_seen_at` and never raises for a missing row.
    """

    async def get_active_by_token_hash(self, token_hash: str) -> DesktopDevice | None: ...

    async def get_device(self, device_id: int) -> DesktopDevice | None: ...

    async def list_devices(self) -> "list[DesktopDevice]": ...

    async def create_device(
        self, *, name: str, token_hash: str, created_by_user_id: int, created_at: datetime
    ) -> DesktopDevice: ...

    async def update_device(
        self, device_id: int, changes: DesktopDeviceChanges, *, at: datetime
    ) -> DesktopDevice | None: ...

    async def revoke_device(self, device_id: int, *, revoked_at: datetime) -> bool: ...

    async def mark_seen(self, device_id: int, *, at: datetime) -> None: ...
