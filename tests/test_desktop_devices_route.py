"""Admin routes for paired Macs through the real app (Phase 14, D-01, D-13,
D-15, D-21).

Boots the real application with a Mac repository and an edge repository. The
edge repository holds one active and one revoked edge device, so a room
mapping has something real to accept or refuse. Token literals stay under 8
characters because `tests/test_repo_hygiene.py` flags longer ones.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
from starlette.testclient import WebSocketDisconnect

from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig
from atlas.desktop.protocol import CLOSE_POLICY_VIOLATION, CLOSE_REVOKED
from test_desktop_tracer import _boot_with_desktop_repo, _create_admin, _fixture_message
from tests.edge_fakes import FakeEdgeDeviceRepository, fake_edge_device

ACTIVE_EDGE_ID = 1
REVOKED_EDGE_ID = 2


def _boot(tmp_path, monkeypatch):
    """The tracer's app, plus an edge repository with an active and a revoked
    device. Returns the un-entered client and the Mac repository."""
    import atlas.app as app_module

    client, desktop_repo = _boot_with_desktop_repo(tmp_path, monkeypatch)
    edge_repo = FakeEdgeDeviceRepository()
    edge_repo.add("e-1", fake_edge_device(device_id=ACTIVE_EDGE_ID, name="kitchen-pi"))
    edge_repo.add(
        "e-2",
        fake_edge_device(
            device_id=REVOKED_EDGE_ID, name="old-pi", revoked_at=datetime.now(timezone.utc)
        ),
    )
    wrapped = app_module._build_repositories

    def _with_edge_repo(config, engine):
        repositories = wrapped(config, engine)
        repositories["edge_device_repo"] = edge_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_edge_repo)
    return client, desktop_repo


def _operator_cookie() -> str:
    """A non-admin identity, created through the booted app's own account
    repository, as `tests/test_edge_devices_route.py` does."""
    import atlas.app as app_module

    operator = asyncio.run(
        app_module.app.state.account_repo.create_user(
            email="desktop-operator@example.invalid",
            display_name="Desktop Operator",
            password_hash="not-checked-by-this-test",
            role="operator",
        )
    )
    return issue_access_token(user_id=operator.id, role="operator", security=SecurityConfig())


def _create_mac(client, name: str = "Desk Mac") -> dict:
    response = client.post("/api/desktop-devices", json={"name": name})
    assert response.status_code == 201, response.text
    return response.json()


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def _hub():
    import atlas.app as app_module

    return app_module.app.state.desktop_hub


def test_revoking_a_connected_mac_closes_its_socket_with_4001_and_refuses_the_next_dial(
    tmp_path, monkeypatch
):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        with client.websocket_connect("/ws/desktop", headers=_bearer(mac["token"])) as socket:
            socket.send_text(json.dumps(_fixture_message("hello.json")))
            assert json.loads(socket.receive_text())["type"] == "hello.ack"
            assert _hub().is_connected(mac["id"])

            assert client.delete(f"/api/desktop-devices/{mac['id']}").status_code == 204

            with pytest.raises(WebSocketDisconnect) as exc_info:
                socket.receive_text()
            assert exc_info.value.code == CLOSE_REVOKED

        assert _hub().snapshot() == {}
        with pytest.raises(WebSocketDisconnect) as refused:
            with client.websocket_connect("/ws/desktop", headers=_bearer(mac["token"])):
                pass
        assert refused.value.code == CLOSE_POLICY_VIOLATION


def test_delete_on_an_unknown_id_returns_404(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        assert client.delete("/api/desktop-devices/999999").status_code == 404


def test_a_second_delete_is_a_204_no_op_and_the_row_stays_revoked_and_not_default(
    tmp_path, monkeypatch
):
    client, repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        from atlas.db.desktop_repository import DesktopDeviceChanges

        asyncio.run(
            repo.update_device(
                mac["id"], DesktopDeviceChanges(is_default=True), at=datetime.now(timezone.utc)
            )
        )
        assert client.get("/api/desktop-devices").json()[0]["is_default"] is True

        assert client.delete(f"/api/desktop-devices/{mac['id']}").status_code == 204
        first_revoked_at = asyncio.run(repo.get_device(mac["id"])).revoked_at
        assert client.delete(f"/api/desktop-devices/{mac['id']}").status_code == 204

        [listed] = client.get("/api/desktop-devices").json()
        assert listed["revoked"] is True
        assert listed["is_default"] is False
        assert asyncio.run(repo.get_device(mac["id"])).revoked_at == first_revoked_at


def test_test_with_no_pong_returns_answered_false_and_a_null_rtt(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        _hub().ping_timeout_s = 0.2
        with client.websocket_connect("/ws/desktop", headers=_bearer(mac["token"])) as socket:
            socket.send_text(json.dumps(_fixture_message("hello.json")))
            assert json.loads(socket.receive_text())["type"] == "hello.ack"

            result = client.post(f"/api/desktop-devices/{mac['id']}/test")

            assert result.status_code == 200, result.text
            assert result.json() == {"answered": False, "rtt_ms": None}
            assert json.loads(socket.receive_text())["type"] == "ping"


def test_test_on_a_revoked_mac_returns_409(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        assert client.delete(f"/api/desktop-devices/{mac['id']}").status_code == 204

        assert client.post(f"/api/desktop-devices/{mac['id']}/test").status_code == 409
