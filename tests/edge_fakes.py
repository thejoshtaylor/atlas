"""Shared edge-protocol test doubles (Phase 10).

`FakeEdgeDeviceRepository` and `interleave` back `tests/test_edge_tracer.py`
(Task 1). `FakeEdgeSocket` backs `tests/test_edge_source.py` (Task 2) --
a scripted double for the subset of `starlette.websockets.WebSocket`
`EdgeAudioSource.serve` actually calls, driving the real class under test
end to end with no mock of it anywhere. Every token literal anywhere in
this project's tests stays under 8 characters (for example `"t-1"`),
because `tests/test_repo_hygiene.py`'s `_CREDENTIAL_RE` flags any quoted
`token = "..."`-shaped literal of 8 or more characters as a possible
leaked credential.
"""

from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import datetime, timezone
from typing import Any

from atlas.db.edge_repository import EdgeDevice


class FakeEdgeDeviceRepository:
    """An in-memory `EdgeDeviceRepository` (`atlas.db.edge_repository`) --
    the Postgres-free double this plan's tests drive, matching every other
    `Fake*Repository` in `tests/conftest.py`. Keyed by token hash;
    `get_active_by_token_hash` returns `None` for both an unknown hash and
    a revoked device (T-10-01) -- the caller never learns which.

    Plan 10-04 extends this to the full Protocol: `create_device`/
    `list_devices`/`revoke_device`/`mark_connected`, keyed additionally by
    id (an admin route or `/ws/edge`'s `mark_connected` call has an id, not
    a token hash). `add` stays for `tests/test_edge_tracer.py`'s own
    pre-existing call shape.
    """

    def __init__(self) -> None:
        self._by_hash: dict[str, EdgeDevice] = {}
        self._by_id: dict[int, EdgeDevice] = {}
        self._next_id = 1

    def add(self, token_hash: str, device: EdgeDevice) -> None:
        self._by_hash[token_hash] = device
        self._by_id[device.id] = device

    async def get_active_by_token_hash(self, token_hash: str) -> "EdgeDevice | None":
        device = self._by_hash.get(token_hash)
        if device is None or device.revoked_at is not None:
            return None
        return device

    async def create_device(
        self, *, name: str, token_hash: str, created_by_user_id: "int | None", created_at: datetime
    ) -> EdgeDevice:
        if token_hash in self._by_hash:
            raise ValueError(f"duplicate token_hash {token_hash!r} -- the real Postgres table's own unique constraint")
        device = EdgeDevice(
            id=self._next_id,
            name=name,
            created_at=created_at,
            created_by_user_id=created_by_user_id,
            revoked_at=None,
            last_connected_at=None,
        )
        self._next_id += 1
        self.add(token_hash, device)
        return device

    async def list_devices(self) -> "list[EdgeDevice]":
        return sorted(self._by_id.values(), key=lambda d: (d.created_at, d.id))

    async def revoke_device(self, device_id: int, *, revoked_at: datetime) -> bool:
        device = self._by_id.get(device_id)
        if device is None or device.revoked_at is not None:
            return False
        self._replace(device_id, replace(device, revoked_at=revoked_at))
        return True

    async def mark_connected(self, device_id: int, *, at: datetime) -> None:
        device = self._by_id.get(device_id)
        if device is None:
            return
        self._replace(device_id, replace(device, last_connected_at=at))

    def _replace(self, device_id: int, updated: EdgeDevice) -> None:
        self._by_id[device_id] = updated
        for token_hash, existing in list(self._by_hash.items()):
            if existing.id == device_id:
                self._by_hash[token_hash] = updated


def fake_edge_device(
    *, device_id: int = 1, name: str = "test-device", revoked_at: "datetime | None" = None
) -> EdgeDevice:
    """One `EdgeDevice` shaped for a test -- every timestamp field a real
    boot would set, defaulted to now/never-revoked."""
    return EdgeDevice(
        id=device_id,
        name=name,
        created_at=datetime.now(timezone.utc),
        created_by_user_id=None,
        revoked_at=revoked_at,
        last_connected_at=None,
    )


class FakeEdgeSocket:
    """A scripted double for the one-way-in-two-ways-out shape
    `EdgeAudioSource.serve` drives: `receive()` returns whatever this test
    pushed, in order; `send_text`/`send_bytes`/`close` are recorded rather
    than sent anywhere. Never mocks `EdgeAudioSource` itself -- the class
    under test runs unchanged against this socket.

    A test drives inbound traffic with `push_bytes`/`push_text`/
    `push_disconnect`; `receive()` awaits the next one, matching
    Starlette's own ASGI message shape (`{"type": "websocket.receive",
    "bytes": ...}` or `{"type": "websocket.disconnect", "code": ...}`).
    """

    def __init__(self) -> None:
        self._inbound: "asyncio.Queue[dict[str, Any]]" = asyncio.Queue()
        self.sent_text: list[str] = []
        self.sent_bytes: list[bytes] = []
        self.close_calls: list[tuple[int, str | None]] = []

    def push_bytes(self, data: bytes) -> None:
        self._inbound.put_nowait({"type": "websocket.receive", "bytes": data})

    def push_text(self, text: str) -> None:
        self._inbound.put_nowait({"type": "websocket.receive", "text": text})

    def push_disconnect(self, code: int = 1000) -> None:
        self._inbound.put_nowait({"type": "websocket.disconnect", "code": code})

    async def receive(self) -> "dict[str, Any]":
        return await self._inbound.get()

    async def send_text(self, data: str) -> None:
        self.sent_text.append(data)

    async def send_bytes(self, data: bytes) -> None:
        self.sent_bytes.append(data)

    async def close(self, code: int = 1000, reason: "str | None" = None) -> None:
        self.close_calls.append((code, reason))


def interleave(ch0: int, ch1: int, samples: int) -> bytes:
    """`samples` frames of 2-channel little-endian PCM16, channel 0 fixed
    at `ch0` and channel 1 fixed at `ch1` -- the fixture
    `tests/test_edge_tracer.py` streams so a real `select_channel` call
    can prove it picked channel 1, never channel 0."""
    import struct

    return struct.pack(f"<{samples * 2}h", *([ch0, ch1] * samples))
