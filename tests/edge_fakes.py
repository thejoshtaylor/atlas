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
from datetime import datetime, timezone
from typing import Any

from atlas.db.edge_repository import EdgeDevice


class FakeEdgeDeviceRepository:
    """An in-memory `EdgeDeviceRepository` (`atlas.db.edge_repository`) --
    the Postgres-free double this plan's tests drive, matching every other
    `Fake*Repository` in `tests/conftest.py`. Keyed by token hash;
    `get_active_by_token_hash` returns `None` for both an unknown hash and
    a revoked device (T-10-01) -- the caller never learns which.
    """

    def __init__(self) -> None:
        self._by_hash: dict[str, EdgeDevice] = {}

    def add(self, token_hash: str, device: EdgeDevice) -> None:
        self._by_hash[token_hash] = device

    async def get_active_by_token_hash(self, token_hash: str) -> "EdgeDevice | None":
        device = self._by_hash.get(token_hash)
        if device is None or device.revoked_at is not None:
            return None
        return device


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
