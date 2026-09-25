"""Admin routes to create, list, and revoke edge devices, and revoke
taking effect on a live connection at once (D-03, T-10-02, T-10-10,
T-10-11, T-10-12, Phase 10, plan 10-04).

Boots the real application through `lifespan`, reusing `tests/
test_auth_setup.py`'s `_boot_with_empty_accounts` helper (the same shape
`tests/test_invites.py` already uses), with an in-memory
`FakeEdgeDeviceRepository` (`tests/edge_fakes.py`) layered onto its own
empty-repository dict -- the failure mode this file guards against is
specifically about the wired-together routes, not a handler called in
isolation.
"""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient as StarletteTestClient
from starlette.testclient import WebSocketDisconnect

import test_auth_setup
from atlas.auth.tokens import issue_access_token
from atlas.config import EdgeSourceConfig, SecurityConfig
from atlas.transports.edge import CLOSE_POLICY_VIOLATION, EdgeAudioSource

from tests.edge_fakes import FakeEdgeDeviceRepository, FakeEdgeSocket, fake_edge_device


def _boot_with_edge_repo(tmp_path, monkeypatch):
    """`_boot_with_empty_accounts`'s app, plus an `edge_device_repo` its
    own fake repository dict does not carry -- layered on top rather than
    duplicating that helper's whole monkeypatch sequence. Returns the
    (un-entered) `TestClient` and the fake repository the routes below
    read/write through; enter the client as its own context manager
    (`with client:`) to run `lifespan`, the same requirement `_boot_
    with_empty_accounts` already carries."""
    import atlas.app as app_module

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)
    edge_device_repo = FakeEdgeDeviceRepository()
    original_build_repositories = app_module._build_repositories

    def _with_edge_repo(config, engine):
        repositories = original_build_repositories(config, engine)
        repositories["edge_device_repo"] = edge_device_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_edge_repo)
    return client, edge_device_repo


def _create_admin(client) -> None:
    response = client.post(
        "/api/auth/create-admin",
        json={
            "email": "edge-admin@example.invalid",
            "display_name": "Edge Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert response.status_code == 201, response.text


def _operator_cookie() -> str:
    """A second, non-admin identity: created directly through the
    already-booted app's own `account_repo`, the same shape
    `test_auth_roles.py`'s own role tests use, rather than an invite
    round trip this file has no interest in testing."""
    import atlas.app as app_module

    security = SecurityConfig()
    operator = asyncio.run(
        app_module.app.state.account_repo.create_user(
            email="edge-operator@example.invalid",
            display_name="Edge Operator",
            password_hash="not-checked-by-this-test",
            role="operator",
        )
    )
    return issue_access_token(user_id=operator.id, role="operator", security=security)


def _measured_edge_config(**overrides) -> EdgeSourceConfig:
    base = dict(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300)
    base.update(overrides)
    return EdgeSourceConfig(**base)


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition never became true within the timeout")


def test_create_returns_the_token_once(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)

        create_response = client.post("/api/edge-devices", json={"name": "kitchen-pi"})
        assert create_response.status_code == 201, create_response.text
        token = create_response.json()["token"]
        assert len(token) > 16, "the plaintext token must be a real, non-trivial bearer credential"

        list_response = client.get("/api/edge-devices")
        assert list_response.status_code == 200, list_response.text
        assert token not in list_response.text, (
            "the plaintext device token must never reappear in the device-listing response"
        )
        for device in list_response.json():
            assert "token" not in device, f"device listing entry carries a token field: {device!r}"
            assert "token_hash" not in device, f"device listing entry carries a token_hash field: {device!r}"


@pytest.mark.parametrize("name", ["", "   ", "x" * 65, "bad/name", "bad;name"])
def test_a_disallowed_name_is_refused(tmp_path, monkeypatch, name):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        response = client.post("/api/edge-devices", json={"name": name})
        assert response.status_code == 422, response.text


def test_an_operator_session_gets_403_on_all_three_routes(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        create_response = client.post("/api/edge-devices", json={"name": "kitchen-pi"})
        device_id = create_response.json()["id"]

        client.cookies.set(SecurityConfig().cookie_name, _operator_cookie())

        assert client.get("/api/edge-devices").status_code == 403
        assert client.post("/api/edge-devices", json={"name": "another-pi"}).status_code == 403
        assert client.delete(f"/api/edge-devices/{device_id}").status_code == 403


def test_no_session_gets_401(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)

        import atlas.app as app_module

        # A fresh, un-entered `TestClient` around the already-booted app --
        # no session cookie of its own, the same "second client, no cookie
        # jar inherited" shape `test_wizard_flow.py`'s own `anonymous`
        # client uses.
        anonymous = StarletteTestClient(app_module.app)
        assert anonymous.get("/api/edge-devices").status_code == 401


def test_delete_on_an_unknown_id_returns_404(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        response = client.delete("/api/edge-devices/999999")
        assert response.status_code == 404, response.text


def test_delete_on_an_already_revoked_id_returns_204_and_changes_nothing(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        create_response = client.post("/api/edge-devices", json={"name": "kitchen-pi"})
        device_id = create_response.json()["id"]

        first_delete = client.delete(f"/api/edge-devices/{device_id}")
        assert first_delete.status_code == 204, first_delete.text

        second_delete = client.delete(f"/api/edge-devices/{device_id}")
        assert second_delete.status_code == 204, second_delete.text

        [listed] = client.get("/api/edge-devices").json()
        assert listed["revoked"] is True


def test_revoked_token_is_refused_on_next_connect(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        create_response = client.post("/api/edge-devices", json={"name": "kitchen-pi"})
        token = create_response.json()["token"]
        device_id = create_response.json()["id"]

        assert client.delete(f"/api/edge-devices/{device_id}").status_code == 204

        with pytest.raises(WebSocketDisconnect) as exc_info:
            with client.websocket_connect("/ws/edge", headers={"Authorization": f"Bearer {token}"}):
                pass
        assert exc_info.value.code == CLOSE_POLICY_VIOLATION


async def test_revoke_closes_a_live_connection(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        create_response = client.post("/api/edge-devices", json={"name": "kitchen-pi"})
        device_id = create_response.json()["id"]

        import atlas.app as app_module

        edge_source = EdgeAudioSource(_measured_edge_config())
        app_module.app.state.edge_source = edge_source
        try:
            socket = FakeEdgeSocket()
            device = fake_edge_device(device_id=device_id)
            serve_task = asyncio.create_task(edge_source.serve(socket, device))
            await _wait_until(lambda: socket.sent_text != [])
            assert edge_source.connected_device_id == device_id

            delete_response = client.delete(f"/api/edge-devices/{device_id}")
            assert delete_response.status_code == 204, delete_response.text

            await _wait_until(lambda: socket.close_calls != [])
            assert socket.close_calls == [(1008, "revoked")]

            await _wait_until(serve_task.done)
            assert serve_task.cancelled()
        finally:
            app_module.app.state.edge_source = None


def test_get_reports_connected_only_for_the_edge_sources_own_device(tmp_path, monkeypatch):
    client, _repo = _boot_with_edge_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        first = client.post("/api/edge-devices", json={"name": "kitchen-pi"}).json()
        second = client.post("/api/edge-devices", json={"name": "hallway-pi"}).json()

        import atlas.app as app_module

        class _StubEdgeSource:
            connected_device_id = first["id"]

        app_module.app.state.edge_source = _StubEdgeSource()
        try:
            listed = {d["id"]: d for d in client.get("/api/edge-devices").json()}
            assert listed[first["id"]]["connected"] is True
            assert listed[second["id"]]["connected"] is False
        finally:
            app_module.app.state.edge_source = None
