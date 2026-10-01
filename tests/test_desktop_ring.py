"""Phase 15 (D-14, D-15, D-17): a ringing timer reaches every Mac with no wake
word and no turn, a Mac that connects during the ring still sees it, and a
Stop click from any Mac stops exactly that ring.

The unit tests run a real `DesktopHub` against `FakeDesktopSocket`s. The last
test boots the real application through `lifespan` with a due timer. Labels
and tokens are fake, and every token literal stays under 8 characters.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from datetime import datetime, timedelta, timezone

import test_auth_setup

from atlas.db.timer_repository import Timer
from atlas.desktop.hub import DesktopHub
from atlas.desktop.protocol import MAX_INVALID_MESSAGES, build_timer_ringing
from atlas.desktop.ring import RING_STICKY_KEY, DesktopRingRelay
from tests.desktop_fakes import (
    FakeDesktopDeviceRepository,
    FakeDesktopSocket,
    fake_desktop_device,
)
from tests.timer_fakes import FakeTimerRepository

T0 = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)

HELLO = json.dumps(
    {
        "type": "hello",
        "protocol": 1,
        "app_version": "0.1.0",
        "os_version": "26.0",
        "capabilities": [],
    }
)


def _timer(timer_id: int = 12, kind: str = "timer", label: str = "pasta") -> Timer:
    return Timer(
        id=timer_id,
        kind=kind,
        label=label,
        due_at=T0,
        remaining_s=None,
        duration_s=60,
        time_of_day=None,
        repeat_days=0,
        enabled=True,
        created_at=T0,
    )


def _hub(**overrides) -> DesktopHub:
    settings = dict(hello_timeout_s=0.5, ping_timeout_s=0.1, idle_timeout_s=0.5)
    settings.update(overrides)
    return DesktopHub(**settings)


async def _until(predicate, *, timeout: float = 1.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.005)
    raise AssertionError("condition never became true within the timeout")


async def _connected(hub: DesktopHub, socket: FakeDesktopSocket, device_id: int):
    socket.push_text(HELLO)
    task = asyncio.create_task(hub.serve(socket, fake_desktop_device(id=device_id)))
    await _until(lambda: "hello.ack" in socket.sent_types() if socket.sent else False)
    return task


def _types(socket: FakeDesktopSocket) -> list[str]:
    return socket.sent_types()


# --- the relay ----------------------------------------------------------------


async def test_a_ring_start_reaches_every_connected_mac_with_its_label():
    hub = _hub()
    first, second = FakeDesktopSocket(), FakeDesktopSocket()
    await _connected(hub, first, 1)
    await _connected(hub, second, 2)
    relay = DesktopRingRelay(hub)

    await relay.on_ring_event(True, _timer(12, "alarm", "wake up"))

    for socket in (first, second):
        await _until(lambda s=socket: "timer.ringing" in _types(s))
        [ringing] = [f for f in socket.sent_json() if f["type"] == "timer.ringing"]
        assert ringing["timer_id"] == 12
        assert ringing["kind"] == "alarm"
        assert ringing["label"] == "wake up"


async def test_a_ring_end_clears_the_sticky_ring_and_sends_stopped_to_every_mac():
    hub = _hub()
    first, second = FakeDesktopSocket(), FakeDesktopSocket()
    await _connected(hub, first, 1)
    await _connected(hub, second, 2)
    relay = DesktopRingRelay(hub)
    await relay.on_ring_event(True, _timer(12))

    await relay.on_ring_event(False, _timer(12))

    for socket in (first, second):
        await _until(lambda s=socket: "timer.stopped" in _types(s))
        [stopped] = [f for f in socket.sent_json() if f["type"] == "timer.stopped"]
        assert stopped["timer_id"] == 12
    late = FakeDesktopSocket()
    await _connected(hub, late, 3)
    await asyncio.sleep(0.05)
    assert _types(late) == ["hello.ack"]


async def test_a_mac_that_connects_during_the_ring_gets_it_right_after_its_hello_ack():
    hub = _hub()
    relay = DesktopRingRelay(hub)
    await relay.on_ring_event(True, _timer(12, "timer", "tea"))

    late = FakeDesktopSocket()
    await _connected(hub, late, 5)
    await _until(lambda: "timer.ringing" in _types(late))

    assert _types(late) == ["hello.ack", "timer.ringing"]
    assert late.sent_json()[1]["timer_id"] == 12
    assert late.sent_json()[1]["label"] == "tea"


async def test_a_mac_that_connects_after_the_ring_ended_gets_no_ring_frame():
    hub = _hub()
    relay = DesktopRingRelay(hub)
    await relay.on_ring_event(True, _timer(12))
    await relay.on_ring_event(False, _timer(12))

    late = FakeDesktopSocket()
    await _connected(hub, late, 5)
    await asyncio.sleep(0.05)

    assert _types(late) == ["hello.ack"]


async def test_the_sticky_key_is_cleared_through_the_hub():
    hub = _hub()
    hub.broadcast(build_timer_ringing(1, "timer", "x"), frame_type="timer.ringing", sticky_key="k")
    hub.clear_sticky("k")
    hub.clear_sticky("never-set")  # a missing key is not an error

    late = FakeDesktopSocket()
    await _connected(hub, late, 5)
    await asyncio.sleep(0.05)

    assert _types(late) == ["hello.ack"]
    assert RING_STICKY_KEY == "ring"


async def test_a_timer_stop_frame_calls_on_timer_stop_once_with_the_device_and_timer_id():
    calls: list[tuple[int, int]] = []
    hub = _hub(on_timer_stop=lambda device_id, timer_id: calls.append((device_id, timer_id)))
    socket = FakeDesktopSocket()
    await _connected(hub, socket, 7)

    socket.push_text(json.dumps({"type": "timer.stop", "timer_id": 12}))
    await _until(lambda: calls == [(7, 12)])
    await asyncio.sleep(0.05)

    assert calls == [(7, 12)]
    assert socket.close_code is None  # a valid stop is never an invalid frame


async def test_a_malformed_timer_stop_counts_as_invalid_and_is_not_dispatched():
    calls: list[tuple[int, int]] = []
    hub = _hub(
        idle_timeout_s=2.0,
        on_timer_stop=lambda device_id, timer_id: calls.append((device_id, timer_id)),
    )
    socket = FakeDesktopSocket()
    await _connected(hub, socket, 7)

    for _ in range(MAX_INVALID_MESSAGES):
        socket.push_text(json.dumps({"type": "timer.stop", "timer_id": "12"}))
    await _until(lambda: socket.close_code is not None)

    assert calls == []
    assert socket.close_reason == "too_many_invalid"


async def test_a_raising_on_timer_stop_never_ends_the_connection(caplog):
    def boom(device_id: int, timer_id: int) -> None:
        raise RuntimeError("a failing stop never ends a connection")

    hub = _hub(on_timer_stop=boom)
    socket = FakeDesktopSocket()
    await _connected(hub, socket, 7)

    with caplog.at_level(logging.WARNING):
        socket.push_text(json.dumps({"type": "timer.stop", "timer_id": 12}))
        await asyncio.sleep(0.05)
        socket.push_text(json.dumps({"type": "ping", "id": 1}))
        await _until(lambda: "pong" in _types(socket))

    assert socket.close_code is None
    assert [r for r in caplog.records if "timer.stop" in r.getMessage()]


def test_on_timer_stop_with_a_stale_id_stops_nothing():
    stopped: list[int] = []

    def stop_ring_for(timer_id: int) -> bool:
        stopped.append(timer_id)
        return False  # the scheduler says that timer does not ring

    relay = DesktopRingRelay(_hub(), stop_ring_for=stop_ring_for)

    relay.on_timer_stop(7, 99)

    assert stopped == [99]


def test_on_timer_stop_with_the_ringing_id_calls_stop_ring_for():
    stopped: list[int] = []
    relay = DesktopRingRelay(_hub(), stop_ring_for=lambda timer_id: stopped.append(timer_id) or True)

    relay.on_timer_stop(7, 12)

    assert stopped == [12]


def test_on_timer_stop_with_no_scheduler_does_nothing_and_does_not_raise():
    DesktopRingRelay(_hub()).on_timer_stop(7, 12)
    DesktopRingRelay(_hub(), stop_ring_for=lambda timer_id: (_ for _ in ()).throw(ValueError("x"))).on_timer_stop(
        7, 12
    )


async def test_on_ring_event_never_raises_and_never_logs_the_label(caplog):
    class _BrokenHub:
        def broadcast(self, *args, **kwargs):
            raise RuntimeError("hub failed")

        def clear_sticky(self, key):
            raise RuntimeError("hub failed")

    relay = DesktopRingRelay(_BrokenHub())  # type: ignore[arg-type]

    with caplog.at_level(logging.DEBUG):
        await relay.on_ring_event(True, _timer(12, "timer", "secret-label"))
        await relay.on_ring_event(False, _timer(12, "timer", "secret-label"))
        await DesktopRingRelay(_hub()).on_ring_event(True, _timer(12, "mystery", "secret-label"))

    assert "secret-label" not in caplog.text
    assert "12" in caplog.text  # the timer id is named


async def test_an_unknown_timer_kind_is_skipped_and_nothing_is_sent():
    hub = _hub()
    socket = FakeDesktopSocket()
    await _connected(hub, socket, 1)

    await DesktopRingRelay(hub).on_ring_event(True, _timer(12, "mystery", "x"))
    await asyncio.sleep(0.05)

    assert _types(socket) == ["hello.ack"]


# --- through the real application ----------------------------------------------


def _boot(tmp_path, monkeypatch, timer_repo, desktop_repo):
    import atlas.app as app_module

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)
    original = app_module._build_repositories

    def _with_repos(config, engine):
        repositories = original(config, engine)
        repositories["timer_repo"] = timer_repo
        repositories["desktop_device_repo"] = desktop_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_repos)
    return client


def _read_until(socket, wanted: str, *, bound_s: float = 8.0) -> dict:
    """Read frames until one has type `wanted`. A bounded wait, because the
    test client's `receive_text` blocks."""
    deadline = time.monotonic() + bound_s
    with ThreadPoolExecutor(max_workers=1) as pool:
        while True:
            remaining = deadline - time.monotonic()
            assert remaining > 0, f"no {wanted} frame within {bound_s} s"
            future = pool.submit(socket.receive_text)
            try:
                frame = json.loads(future.result(timeout=remaining))
            except FutureTimeout:
                raise AssertionError(f"no {wanted} frame within {bound_s} s") from None
            if frame["type"] == wanted:
                return frame


def test_a_due_timer_rings_a_paired_mac_and_its_stop_ends_the_ring(tmp_path, monkeypatch):
    import atlas.app as app_module
    import test_startup_smoke as smoke
    from atlas.providers.tts_xai import SinkFormat

    async def fake_send_audio(self, chunk):
        await asyncio.sleep(0)

    monkeypatch.setattr(
        smoke._FakeCameraSource, "sink_format", lambda self: SinkFormat("alaw", 8000), raising=False
    )
    monkeypatch.setattr(smoke._FakeCameraSource, "send_audio", fake_send_audio, raising=False)
    timer_repo = FakeTimerRepository()
    desktop_repo = FakeDesktopDeviceRepository()
    client = _boot(tmp_path, monkeypatch, timer_repo, desktop_repo)

    with client:
        test_auth_setup_admin = {
            "email": "ring-admin@example.invalid",
            "display_name": "Ring Admin",
            "password": "a-plainly-fictional-test-password",
        }
        assert client.post("/api/auth/create-admin", json=test_auth_setup_admin).status_code == 201
        created = client.post("/api/desktop-devices", json={"name": "Ring Mac"})
        assert created.status_code == 201, created.text
        token = created.json()["token"]

        with client.websocket_connect(
            "/ws/desktop", headers={"Authorization": f"Bearer {token}"}
        ) as socket:
            socket.send_text(HELLO)
            assert json.loads(socket.receive_text())["type"] == "hello.ack"

            now = datetime.now(timezone.utc)
            timer = client.portal.call(
                lambda: timer_repo.create_timer(
                    kind="timer",
                    label="pasta",
                    due_at=now,
                    remaining_s=None,
                    duration_s=1,
                    time_of_day=None,
                    repeat_days=0,
                    enabled=True,
                    created_at=now - timedelta(seconds=1),
                )
            )

            ringing = _read_until(socket, "timer.ringing")
            assert ringing["timer_id"] == timer.id
            assert ringing["kind"] == "timer"
            assert ringing["label"] == "pasta"
            scheduler = app_module.app.state.timer_scheduler
            assert client.portal.call(lambda: scheduler.ringing) is True

            socket.send_text(json.dumps({"type": "timer.stop", "timer_id": timer.id}))
            stopped = _read_until(socket, "timer.stopped")
            assert stopped["timer_id"] == timer.id

            deadline = time.monotonic() + 3
            while client.portal.call(lambda: scheduler.ringing) and time.monotonic() < deadline:
                time.sleep(0.02)
            assert client.portal.call(lambda: scheduler.ringing) is False
