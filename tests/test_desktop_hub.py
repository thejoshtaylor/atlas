"""`DesktopHub` lifecycle against `FakeDesktopSocket` (Phase 14, D-02, D-21).

`DesktopHub.serve` runs unchanged against a scripted socket and the in-memory
repository. Timeouts are small, so each test ends well under a second.
Token literals stay under 8 characters.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from datetime import datetime, timezone

import pytest
from starlette.websockets import WebSocketDisconnect

from atlas.desktop.hub import DesktopHub, DesktopNotConnected
from atlas.desktop.protocol import (
    CLOSE_POLICY_VIOLATION,
    CLOSE_PROTOCOL_MISMATCH,
    CLOSE_REVOKED,
    CLOSE_SUPERSEDED,
    MAX_INVALID_MESSAGES,
    MAX_TEXT_FRAME_BYTES,
)
from tests.desktop_fakes import (
    FakeDesktopDeviceRepository,
    FakeDesktopSocket,
    fake_desktop_device,
)

HELLO = json.dumps(
    {
        "type": "hello",
        "protocol": 1,
        "app_version": "0.1.0",
        "os_version": "26.0",
        "capabilities": [],
    }
)


class _DyingSocket(FakeDesktopSocket):
    """A socket whose peer vanished: every send raises, like Starlette does
    for a Mac that slept or lost its network."""

    def __init__(self, exc: Exception) -> None:
        super().__init__()
        self.exc = exc
        self.dead = False

    async def send_text(self, data: str) -> None:
        if self.dead:
            raise self.exc
        await super().send_text(data)


def _hub(repo=None, **overrides) -> DesktopHub:
    settings = dict(hello_timeout_s=0.5, ping_timeout_s=0.1, idle_timeout_s=0.5)
    settings.update(overrides)
    return DesktopHub(device_repo=repo, **settings)


async def _until(predicate, *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true within the timeout")


def _start(hub: DesktopHub, socket: FakeDesktopSocket, device) -> "asyncio.Task[None]":
    socket.push_text(HELLO)
    return asyncio.create_task(hub.serve(socket, device))


async def _connected(hub: DesktopHub, socket: FakeDesktopSocket, device) -> "asyncio.Task[None]":
    task = _start(hub, socket, device)
    await _until(lambda: "hello.ack" in socket.sent_types() if socket.sent else False)
    return task


async def test_a_reconnect_supersedes_the_old_socket_with_4000() -> None:
    hub = _hub()
    device = fake_desktop_device(id=1)
    first, second = FakeDesktopSocket(), FakeDesktopSocket()
    first_task = await _connected(hub, first, device)

    second_task = await _connected(hub, second, device)

    assert first.close_code == CLOSE_SUPERSEDED
    await _until(first_task.done)
    assert first_task.result() is None  # ended quietly, not by a CancelledError
    assert list(hub.snapshot()) == [1]
    # The first call's cleanup ran, and it did not remove the new entry.
    assert hub.is_connected(1)
    assert second.close_code is None
    second.push_disconnect()
    await _until(second_task.done)
    assert not hub.is_connected(1)


async def test_a_previous_socket_that_fails_to_close_does_not_block_the_new_one() -> None:
    hub = _hub()
    device = fake_desktop_device(id=1)
    dead, fresh = FakeDesktopSocket(close_raises=True), FakeDesktopSocket()
    dead_task = await _connected(hub, dead, device)

    fresh_task = await _connected(hub, fresh, device)

    assert dead.close_code == CLOSE_SUPERSEDED
    await _until(dead_task.done)
    assert hub.is_connected(1)
    fresh.push_disconnect()
    await _until(fresh_task.done)


async def test_disconnect_device_closes_with_4001_and_cancels_the_task() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    await hub.disconnect_device(1, code=CLOSE_REVOKED, reason="revoked")

    assert socket.close_code == CLOSE_REVOKED
    assert socket.close_reason == "revoked"
    await _until(task.done)
    assert task.result() is None  # cancelled by the hub, ended quietly
    assert not hub.is_connected(1)


async def test_disconnect_device_still_cancels_when_the_close_raises() -> None:
    hub = _hub()
    socket = FakeDesktopSocket(close_raises=True)
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    await hub.disconnect_device(1, code=CLOSE_REVOKED, reason="revoked")

    await _until(task.done)
    assert task.result() is None  # cancelled by the hub, ended quietly
    assert not hub.is_connected(1)


async def test_disconnect_device_for_an_unconnected_id_does_nothing() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    await hub.disconnect_device(2, code=CLOSE_REVOKED, reason="revoked")

    assert socket.close_code is None
    assert hub.is_connected(1)
    assert not task.done()
    socket.push_disconnect()
    await _until(task.done)


async def test_a_mac_revoked_before_its_hello_is_closed_with_4001_and_never_registered() -> None:
    repo = FakeDesktopDeviceRepository()
    device = fake_desktop_device(id=1)
    repo.add("h-1", replace(device, revoked_at=datetime.now(timezone.utc)))
    hub = _hub(repo)
    socket = FakeDesktopSocket()
    socket.push_text(HELLO)

    await hub.serve(socket, device)

    assert socket.close_code == CLOSE_REVOKED
    assert hub.snapshot() == {}
    assert socket.sent == []
    assert repo.seen_calls == []


async def test_a_silent_socket_is_closed_with_1008_after_the_idle_timeout() -> None:
    hub = _hub(idle_timeout_s=0.1)
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    await _until(task.done)

    assert socket.close_code == CLOSE_POLICY_VIOLATION
    assert socket.close_reason == "idle"
    assert not hub.is_connected(1)


@pytest.mark.parametrize(
    "push",
    [
        lambda s: s.push_text("x" * (MAX_TEXT_FRAME_BYTES + 1)),
        lambda s: s.push_bytes(b"\x00\x01"),
        lambda s: s.push_text("{not json"),
        lambda s: s.push_text(json.dumps({"type": "pong", "id": 77})),
    ],
    ids=["oversize", "binary", "invalid-json", "unmatched-pong"],
)
async def test_each_kind_of_bad_frame_counts_and_the_twentieth_closes_1008(push) -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    for _ in range(MAX_INVALID_MESSAGES - 1):
        push(socket)
    await asyncio.sleep(0.05)
    assert socket.close_code is None
    assert hub.is_connected(1)

    push(socket)
    await _until(task.done)
    assert socket.close_code == CLOSE_POLICY_VIOLATION
    assert socket.close_reason == "too_many_invalid"


async def test_an_unknown_message_type_is_ignored_and_not_counted() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    for _ in range(MAX_INVALID_MESSAGES * 2):
        socket.push_text(json.dumps({"type": "from.the.future", "x": 1}))
    await asyncio.sleep(0.05)

    assert socket.close_code is None
    assert hub.is_connected(1)
    socket.push_disconnect()
    await _until(task.done)


async def test_a_client_ping_is_answered_with_a_pong_of_the_same_id() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    socket.push_text(json.dumps({"type": "ping", "id": 41}))
    await _until(lambda: "pong" in socket.sent_types())

    assert socket.sent_json()[-1] == {"type": "pong", "id": 41}
    socket.push_disconnect()
    await _until(task.done)


async def test_a_future_protocol_hello_that_fails_v1_validation_still_gets_4002() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    # The v2 hello dropped `capabilities` and `os_version`: invalid under v1.
    socket.push_text(json.dumps({"type": "hello", "protocol": 2, "app_version": "9.0.0"}))

    await hub.serve(socket, fake_desktop_device(id=1))

    assert socket.sent_json()[0]["code"] == "protocol_mismatch"
    assert socket.close_code == CLOSE_PROTOCOL_MISMATCH
    assert not hub.is_connected(1)


async def test_an_oversized_first_frame_is_refused_as_a_bad_hello() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    big = json.dumps(
        {
            "type": "hello",
            "protocol": 1,
            "app_version": "0.1.0",
            "os_version": "26.0",
            "capabilities": [],
            "padding": "x" * MAX_TEXT_FRAME_BYTES,
        }
    )
    socket.push_text(big)

    await hub.serve(socket, fake_desktop_device(id=1))

    assert socket.sent_json()[0]["code"] == "bad_hello"
    assert socket.close_code == CLOSE_POLICY_VIOLATION
    assert not hub.is_connected(1)


async def test_ping_returns_none_after_its_timeout() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    assert await hub.ping(1, timeout_s=0.05) is None

    assert "ping" in socket.sent_types()
    socket.push_disconnect()
    await _until(task.done)


async def test_ping_returns_a_round_trip_when_the_mac_answers() -> None:
    hub = _hub()
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    waiting = asyncio.create_task(hub.ping(1, timeout_s=1.0))
    await _until(lambda: "ping" in socket.sent_types())
    ping_id = next(f["id"] for f in socket.sent_json() if f["type"] == "ping")
    socket.push_text(json.dumps({"type": "pong", "id": ping_id}))

    rtt = await waiting
    assert rtt is not None and rtt >= 0
    socket.push_disconnect()
    await _until(task.done)


async def test_a_disconnect_while_a_ping_waits_returns_none_at_once() -> None:
    hub = _hub(ping_timeout_s=5.0)
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    waiting = asyncio.create_task(hub.ping(1))
    await _until(lambda: "ping" in socket.sent_types())
    socket.push_disconnect()

    assert await asyncio.wait_for(waiting, 1.0) is None
    await _until(task.done)


@pytest.mark.parametrize("exc", [WebSocketDisconnect(1006), RuntimeError("closed"), BrokenPipeError()])
async def test_a_ping_to_a_socket_that_just_died_returns_none(exc) -> None:
    hub = _hub()
    socket = _DyingSocket(exc)
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    socket.dead = True
    assert await hub.ping(1, timeout_s=1.0) is None

    socket.push_disconnect()
    await _until(task.done)


async def test_a_pong_send_on_a_dead_socket_ends_serve_quietly() -> None:
    hub = _hub()
    socket = _DyingSocket(WebSocketDisconnect(1006))
    task = await _connected(hub, socket, fake_desktop_device(id=1))

    socket.dead = True
    socket.push_text(json.dumps({"type": "ping", "id": 3}))
    await _until(task.done)

    assert task.exception() is None
    assert not hub.is_connected(1)


async def test_a_hello_ack_send_on_a_dead_socket_ends_serve_quietly() -> None:
    hub = _hub()
    socket = _DyingSocket(RuntimeError("closed"))
    socket.dead = True
    task = _start(hub, socket, fake_desktop_device(id=1))
    await _until(task.done)

    assert task.exception() is None
    assert not hub.is_connected(1)


async def test_ping_for_an_unconnected_mac_raises() -> None:
    hub = _hub()
    with pytest.raises(DesktopNotConnected):
        await hub.ping(9)


async def test_last_seen_is_written_on_hello_and_again_when_the_connection_ends() -> None:
    repo = FakeDesktopDeviceRepository()
    device = fake_desktop_device(id=1)
    repo.add("h-1", device)
    hub = _hub(repo)
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, device)

    assert [seen_id for seen_id, _ in repo.seen_calls] == [1]
    socket.push_disconnect()
    await _until(task.done)

    assert [seen_id for seen_id, _ in repo.seen_calls] == [1, 1]
    assert (await repo.get_device(1)).last_seen_at is not None


async def test_a_failing_mark_seen_is_logged_and_the_connection_keeps_running(caplog) -> None:
    repo = FakeDesktopDeviceRepository()
    device = fake_desktop_device(id=1)
    repo.add("h-1", device)

    async def broken(device_id, *, at):
        raise RuntimeError("store down")

    repo.mark_seen = broken
    hub = _hub(repo)
    socket = FakeDesktopSocket()
    task = await _connected(hub, socket, device)

    assert hub.is_connected(1)
    assert not task.done()
    socket.push_disconnect()
    await _until(task.done)
    assert any("desktop 1: mark_seen failed" in record.getMessage() for record in caplog.records)
