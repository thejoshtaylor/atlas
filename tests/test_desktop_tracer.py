"""Phase 14 tracer: an admin creates a Mac, its token connects to
`/ws/desktop`, the shared hello fixture is acknowledged, the list shows it
online from the hub, and the Test route gets an answered round trip.

Boots the real application through `lifespan` (`tests/test_auth_setup.py`'s
`_boot_with_empty_accounts`), with a `FakeDesktopDeviceRepository` layered
onto its repository dict. The far end is a plain client that sends the exact
bytes of `desktop/protocol/v1/messages/hello.json`, the message the Swift app
sends.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import test_auth_setup

from tests.desktop_fakes import FakeDesktopDeviceRepository

PROTOCOL_DIR = Path(__file__).resolve().parents[1] / "desktop" / "protocol" / "v1"


def _fixture_message(name: str) -> dict:
    return json.loads((PROTOCOL_DIR / "messages" / name).read_text())["message"]


def _boot_with_desktop_repo(tmp_path, monkeypatch):
    """`_boot_with_empty_accounts`'s app, plus a `desktop_device_repo` its
    own fake repository dict does not carry. Enter the returned client as a
    context manager to run `lifespan`."""
    import atlas.app as app_module

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)
    desktop_device_repo = FakeDesktopDeviceRepository()
    original_build_repositories = app_module._build_repositories

    def _with_desktop_repo(config, engine):
        repositories = original_build_repositories(config, engine)
        repositories["desktop_device_repo"] = desktop_device_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_desktop_repo)
    return client, desktop_device_repo


def _create_admin(client) -> None:
    response = client.post(
        "/api/auth/create-admin",
        json={
            "email": "desktop-admin@example.invalid",
            "display_name": "Desktop Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert response.status_code == 201, response.text


def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError("condition never became true within the timeout")


def test_admin_creates_a_mac_that_connects_shows_online_and_answers_test(tmp_path, monkeypatch):
    client, repo = _boot_with_desktop_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)

        created = client.post("/api/desktop-devices", json={"name": "Test Mac"})
        assert created.status_code == 201, created.text
        assert created.headers["cache-control"] == "no-store"
        body = created.json()
        token = body["token"]
        device_id = body["id"]
        assert len(token) > 16

        [listed] = client.get("/api/desktop-devices").json()
        assert listed["connected"] is False
        assert "token" not in listed and "token_hash" not in listed

        ack_fixture = _fixture_message("hello_ack.json")
        with client.websocket_connect(
            "/ws/desktop", headers={"Authorization": f"Bearer {token}"}
        ) as socket:
            socket.send_text(json.dumps(_fixture_message("hello.json")))
            ack = json.loads(socket.receive_text())
            assert set(ack) == set(ack_fixture)
            assert ack["type"] == "hello.ack"
            assert ack["protocol"] == 1
            assert ack["ping_interval_s"] == 15
            assert ack["device_id"] == device_id

            [online] = client.get("/api/desktop-devices").json()
            assert online["connected"] is True

            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(client.post, f"/api/desktop-devices/{device_id}/test")
                ping = json.loads(socket.receive_text())
                assert ping["type"] == "ping"
                socket.send_text(json.dumps({"type": "pong", "id": ping["id"]}))
                result = pending.result(timeout=5)
            assert result.status_code == 200, result.text
            answer = result.json()
            assert answer["answered"] is True
            assert isinstance(answer["rtt_ms"], int) and answer["rtt_ms"] >= 0

        _wait_until(lambda: client.get("/api/desktop-devices").json()[0]["connected"] is False)
        assert any(seen_id == device_id for seen_id, _at in repo.seen_calls)
