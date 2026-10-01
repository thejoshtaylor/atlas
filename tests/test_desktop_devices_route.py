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
from atlas.desktop.protocol import (
    CLOSE_REVOKED,
    REFUSAL_HEADER,
    REFUSAL_STATUS,
    REFUSAL_TOKEN,
)
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
        # A revoked token is the marked HTTP 403 denial (D-30).
        assert refused.value.status_code == REFUSAL_STATUS
        assert refused.value.headers.get(REFUSAL_HEADER) == REFUSAL_TOKEN


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


def _patch(client, mac: dict, body: dict):
    return client.patch(f"/api/desktop-devices/{mac['id']}", json=body)


def _listed(client) -> dict:
    return {row["id"]: row for row in client.get("/api/desktop-devices").json()}


def test_patch_renames_a_mac_and_the_list_shows_the_new_name(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)

        response = _patch(client, mac, {"name": "  Studio Mac  "})

        assert response.status_code == 200, response.text
        assert response.json()["name"] == "Studio Mac"
        assert _listed(client)[mac["id"]]["name"] == "Studio Mac"


def test_a_rename_that_matches_another_active_name_ignoring_case_is_409_until_it_is_revoked(
    tmp_path, monkeypatch
):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        first = _create_mac(client, "Desk Mac")
        second = _create_mac(client, "Other Mac")

        clash = _patch(client, second, {"name": "desk mac"})
        assert clash.status_code == 409
        assert clash.json()["detail"] == "another active Mac already uses this name"

        assert client.delete(f"/api/desktop-devices/{first['id']}").status_code == 204
        assert _patch(client, second, {"name": "desk mac"}).status_code == 200


def test_create_with_a_name_that_matches_an_active_mac_ignoring_case_is_409(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        _create_mac(client, "Desk Mac")

        clash = client.post("/api/desktop-devices", json={"name": "DESK MAC"})

        assert clash.status_code == 409
        assert clash.json()["detail"] == "another active Mac already uses this name"


def test_a_revoked_mac_frees_its_name_for_a_new_one(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        old = _create_mac(client, "Desk Mac")
        assert client.delete(f"/api/desktop-devices/{old['id']}").status_code == 204

        assert client.post("/api/desktop-devices", json={"name": "desk mac"}).status_code == 201


@pytest.mark.parametrize("name", ["bad/name", "", "   ", "x" * 65, None])
def test_patch_with_an_invalid_name_is_422(tmp_path, monkeypatch, name):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)

        assert _patch(client, mac, {"name": name}).status_code == 422


def test_patch_maps_a_mac_to_an_active_edge_device_and_null_clears_it(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)

        mapped = _patch(client, mac, {"edge_device_id": ACTIVE_EDGE_ID})
        assert mapped.status_code == 200, mapped.text
        assert mapped.json()["edge_device_id"] == ACTIVE_EDGE_ID

        cleared = _patch(client, mac, {"edge_device_id": None})
        assert cleared.status_code == 200, cleared.text
        assert cleared.json()["edge_device_id"] is None


def test_a_body_without_edge_device_id_leaves_the_mapping_unchanged(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        assert _patch(client, mac, {"edge_device_id": ACTIVE_EDGE_ID}).status_code == 200

        renamed = _patch(client, mac, {"name": "Studio Mac"})

        assert renamed.json()["edge_device_id"] == ACTIVE_EDGE_ID


def test_several_macs_may_map_to_the_same_edge_device(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        first = _create_mac(client, "Mac One")
        second = _create_mac(client, "Mac Two")

        assert _patch(client, first, {"edge_device_id": ACTIVE_EDGE_ID}).status_code == 200
        assert _patch(client, second, {"edge_device_id": ACTIVE_EDGE_ID}).status_code == 200
        rows = _listed(client)
        assert rows[first["id"]]["edge_device_id"] == ACTIVE_EDGE_ID
        assert rows[second["id"]]["edge_device_id"] == ACTIVE_EDGE_ID


@pytest.mark.parametrize("edge_id", [999, REVOKED_EDGE_ID])
def test_mapping_to_an_unknown_or_revoked_edge_device_is_422(tmp_path, monkeypatch, edge_id):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)

        response = _patch(client, mac, {"edge_device_id": edge_id})

        assert response.status_code == 422
        assert response.json()["detail"] == "no such active edge device"
        assert _listed(client)[mac["id"]]["edge_device_id"] is None


def test_mapping_is_422_when_the_server_has_no_edge_repository(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        import atlas.app as app_module

        monkeypatch.setattr(app_module.app.state, "edge_device_repo", None)

        assert _patch(client, mac, {"edge_device_id": ACTIVE_EDGE_ID}).status_code == 422


def test_setting_a_default_clears_the_previous_one_and_false_leaves_none(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac_a = _create_mac(client, "Mac A")
        mac_b = _create_mac(client, "Mac B")

        assert _patch(client, mac_a, {"is_default": True}).json()["is_default"] is True
        assert _patch(client, mac_b, {"is_default": True}).json()["is_default"] is True
        rows = _listed(client)
        assert [rows[mac_a["id"]]["is_default"], rows[mac_b["id"]]["is_default"]] == [False, True]

        assert _patch(client, mac_b, {"is_default": False}).json()["is_default"] is False
        assert not any(row["is_default"] for row in _listed(client).values())


def test_an_empty_patch_is_422_nothing_to_change(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)

        response = _patch(client, mac, {})

        assert response.status_code == 422
        assert response.json()["detail"] == "nothing to change"


def test_patch_on_an_unknown_id_is_404(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)

        assert client.patch("/api/desktop-devices/999999", json={"name": "Nobody"}).status_code == 404


def test_patch_on_a_revoked_mac_is_409(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        assert client.delete(f"/api/desktop-devices/{mac['id']}").status_code == 204

        response = _patch(client, mac, {"name": "Studio Mac"})

        assert response.status_code == 409
        assert response.json()["detail"] == "this Mac is revoked"


def test_a_default_race_in_the_repository_is_409_try_again(tmp_path, monkeypatch):
    from atlas.db.desktop_repository import DesktopDefaultConflict

    client, repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)

        async def lost_the_race(device_id, changes, *, at):
            raise DesktopDefaultConflict()

        monkeypatch.setattr(repo, "update_device", lost_the_race)

        response = _patch(client, mac, {"is_default": True})

        assert response.status_code == 409
        assert (
            response.json()["detail"]
            == "another Mac became the default at the same time. Try again."
        )


def test_a_mac_revoked_between_the_lookup_and_the_update_is_409(tmp_path, monkeypatch):
    client, repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)

        async def revoked_in_between(device_id, changes, *, at):
            return None

        monkeypatch.setattr(repo, "update_device", revoked_in_between)

        response = _patch(client, mac, {"name": "Studio Mac"})

        assert response.status_code == 409
        assert response.json()["detail"] == "this Mac is revoked"


def test_the_create_response_is_never_cached_and_the_list_has_exactly_the_documented_keys(
    tmp_path, monkeypatch
):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        created = client.post("/api/desktop-devices", json={"name": "Desk Mac"})
        assert created.headers["cache-control"] == "no-store"

        [row] = client.get("/api/desktop-devices").json()
        assert set(row) == {
            "id",
            "name",
            "created_at",
            "revoked",
            "last_seen_at",
            "connected",
            "edge_device_id",
            "is_default",
        }


def test_every_route_is_403_for_an_operator(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        client.cookies.set(SecurityConfig().cookie_name, _operator_cookie())
        base = "/api/desktop-devices"

        assert client.get(base).status_code == 403
        assert client.post(base, json={"name": "Other Mac"}).status_code == 403
        assert client.patch(f"{base}/{mac['id']}", json={"name": "Other Mac"}).status_code == 403
        assert client.delete(f"{base}/{mac['id']}").status_code == 403
        assert client.post(f"{base}/{mac['id']}/test").status_code == 403


def test_every_route_is_401_with_no_session(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        mac = _create_mac(client)
        import atlas.app as app_module
        from starlette.testclient import TestClient

        anonymous = TestClient(app_module.app)
        base = "/api/desktop-devices"

        assert anonymous.get(base).status_code == 401
        assert anonymous.post(base, json={"name": "Other Mac"}).status_code == 401
        assert anonymous.patch(f"{base}/{mac['id']}", json={"name": "Other Mac"}).status_code == 401
        assert anonymous.delete(f"{base}/{mac['id']}").status_code == 401
        assert anonymous.post(f"{base}/{mac['id']}/test").status_code == 401
