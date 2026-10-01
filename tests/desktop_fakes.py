"""Shared Mac-device test doubles (Phase 14).

`FakeDesktopDeviceRepository` implements the whole `DesktopDeviceRepository`
Protocol (`atlas.db.desktop_repository`) in memory, with exactly the rules
that Protocol's docstring states, so the Postgres class (plan 14-04) and
this fake behave the same. Every token literal anywhere in this project's
tests stays under 8 characters (for example `"t-1"`), because
`tests/test_repo_hygiene.py` flags longer quoted token literals.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone

from atlas.db.desktop_repository import (
    DesktopDevice,
    DesktopDeviceChanges,
    DesktopDeviceNameTaken,
)


class FakeDesktopDeviceRepository:
    def __init__(self) -> None:
        self._by_hash: dict[str, int] = {}
        self._by_id: dict[int, DesktopDevice] = {}
        self._next_id = 1
        # Every `mark_seen` call, in order, so a test can prove the hub wrote one.
        self.seen_calls: list[tuple[int, datetime]] = []

    def add(self, token_hash: str, device: DesktopDevice) -> None:
        self._by_hash[token_hash] = device.id
        self._by_id[device.id] = device
        self._next_id = max(self._next_id, device.id + 1)

    def _name_taken(self, name: str, *, except_id: int | None = None) -> bool:
        folded = name.casefold()
        return any(
            device.name.casefold() == folded
            for device in self._by_id.values()
            if device.revoked_at is None and device.id != except_id
        )

    async def get_active_by_token_hash(self, token_hash: str) -> DesktopDevice | None:
        device_id = self._by_hash.get(token_hash)
        if device_id is None:
            return None
        device = self._by_id[device_id]
        return None if device.revoked_at is not None else device

    async def get_device(self, device_id: int) -> DesktopDevice | None:
        return self._by_id.get(device_id)

    async def list_devices(self) -> "list[DesktopDevice]":
        return sorted(self._by_id.values(), key=lambda d: (d.created_at, d.id))

    async def create_device(
        self, *, name: str, token_hash: str, created_by_user_id: "int | None", created_at: datetime
    ) -> DesktopDevice:
        if self._name_taken(name):
            raise DesktopDeviceNameTaken(name)
        device = DesktopDevice(
            id=self._next_id,
            name=name,
            created_at=created_at,
            created_by_user_id=created_by_user_id,
            revoked_at=None,
            last_seen_at=None,
            edge_device_id=None,
            is_default=False,
        )
        self.add(token_hash, device)
        return device

    async def update_device(
        self, device_id: int, changes: DesktopDeviceChanges, *, at: datetime
    ) -> DesktopDevice | None:
        device = self._by_id.get(device_id)
        if device is None or device.revoked_at is not None:
            return None
        updated = device
        if changes.name is not None:
            if self._name_taken(changes.name, except_id=device_id):
                raise DesktopDeviceNameTaken(changes.name)
            updated = replace(updated, name=changes.name)
        if changes.set_edge_device:
            updated = replace(updated, edge_device_id=changes.edge_device_id)
        if changes.is_default is not None:
            if changes.is_default:
                for other_id, other in list(self._by_id.items()):
                    if other_id != device_id and other.is_default:
                        self._by_id[other_id] = replace(other, is_default=False)
            updated = replace(updated, is_default=changes.is_default)
        self._by_id[device_id] = updated
        return updated

    async def revoke_device(self, device_id: int, *, revoked_at: datetime) -> bool:
        device = self._by_id.get(device_id)
        if device is None or device.revoked_at is not None:
            return False
        self._by_id[device_id] = replace(device, revoked_at=revoked_at, is_default=False)
        return True

    async def mark_seen(self, device_id: int, *, at: datetime) -> None:
        self.seen_calls.append((device_id, at))
        device = self._by_id.get(device_id)
        if device is not None:
            self._by_id[device_id] = replace(device, last_seen_at=at)


def fake_desktop_device(**overrides) -> DesktopDevice:
    """One `DesktopDevice` shaped for a test. Override any field by name."""
    fields = dict(
        id=1,
        name="Test Mac",
        created_at=datetime.now(timezone.utc),
        created_by_user_id=None,
        revoked_at=None,
        last_seen_at=None,
        edge_device_id=None,
        is_default=False,
    )
    fields.update(overrides)
    return DesktopDevice(**fields)
