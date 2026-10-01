"""`/ws/desktop` refusals, the protocol-mismatch close, and PAIR-06 in both
directions (Phase 14, D-02, D-09, T-14-01, T-14-02, T-14-05).

Boots the real application with a Mac repository and an edge repository,
both seeded with one device. Token literals stay under 8 characters
(`"t-1"`, `"e-1"`) because `tests/test_repo_hygiene.py` flags longer ones.
"""

from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone

import pytest
from starlette.testclient import WebSocketDisconnect
from starlette.websockets import WebSocket

from atlas.auth.desktop_tokens import handle_desktop_token_refused, hash_desktop_token
from atlas.auth.edge_tokens import hash_edge_token
from atlas.desktop.protocol import (
    CLOSE_POLICY_VIOLATION,
    CLOSE_PROTOCOL_MISMATCH,
    REFUSAL_HEADER,
    REFUSAL_STATUS,
    REFUSAL_TOKEN,
)
from test_desktop_tracer import _boot_with_desktop_repo, _create_admin, _fixture_message
from tests.desktop_fakes import fake_desktop_device
from tests.edge_fakes import FakeEdgeDeviceRepository, fake_edge_device

DESKTOP_TOKEN = "t-1"
EDGE_TOKEN = "e-1"
REVOKED_TOKEN = "t-2"


def _boot(tmp_path, monkeypatch):
    """The tracer's app, plus an edge repository, so a refusal on `/ws/edge`
    is a real lookup miss and not a missing repository."""
    import atlas.app as app_module

    client, desktop_repo = _boot_with_desktop_repo(tmp_path, monkeypatch)
    edge_repo = FakeEdgeDeviceRepository()
    desktop_wrapped = app_module._build_repositories

    def _with_edge_repo(config, engine):
        repositories = desktop_wrapped(config, engine)
        repositories["edge_device_repo"] = edge_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_edge_repo)
    desktop_repo.add(hash_desktop_token(DESKTOP_TOKEN), fake_desktop_device(id=1))
    desktop_repo.add(
        hash_desktop_token(REVOKED_TOKEN),
        fake_desktop_device(id=2, name="Old Mac", revoked_at=datetime.now(timezone.utc)),
    )
    edge_repo.add(hash_edge_token(EDGE_TOKEN), fake_edge_device(device_id=1))
    return client


def _hub_snapshot() -> dict:
    import atlas.app as app_module

    return app_module.app.state.desktop_hub.snapshot()


def _refusal(client, path, headers) -> tuple[int, str]:
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(path, headers=headers):
            pass
    return exc_info.value.code, exc_info.value.reason


def _desktop_refusal(client, path, headers) -> tuple[int, str | None]:
    """A refused Mac token is an HTTP 403 denial. Returns its status and the
    value of the refusal header (D-30)."""
    with pytest.raises(WebSocketDisconnect) as exc_info:
        with client.websocket_connect(path, headers=headers):
            pass
    denial = exc_info.value
    assert hasattr(denial, "status_code"), "expected a denial response, not a close frame"
    return denial.status_code, denial.headers.get(REFUSAL_HEADER)


def _bearer(token: str) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_a_hello_with_the_wrong_protocol_gets_a_named_error_then_4002(tmp_path, monkeypatch):
    client = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        hello = {**_fixture_message("hello.json"), "protocol": 2}
        with client.websocket_connect("/ws/desktop", headers=_bearer(DESKTOP_TOKEN)) as socket:
            socket.send_text(json.dumps(hello))
            error = json.loads(socket.receive_text())
            assert error["type"] == "error"
            assert error["code"] == "protocol_mismatch"
            with pytest.raises(WebSocketDisconnect) as exc_info:
                socket.receive_text()
            assert exc_info.value.code == CLOSE_PROTOCOL_MISMATCH
        assert _hub_snapshot() == {}


def test_a_first_frame_that_is_not_a_hello_gets_bad_hello_then_1008(tmp_path, monkeypatch):
    client = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        with client.websocket_connect("/ws/desktop", headers=_bearer(DESKTOP_TOKEN)) as socket:
            socket.send_text(json.dumps(_fixture_message("ping_client.json")))
            error = json.loads(socket.receive_text())
            assert error["code"] == "bad_hello"
            with pytest.raises(WebSocketDisconnect) as exc_info:
                socket.receive_text()
            assert exc_info.value.code == CLOSE_POLICY_VIOLATION
        assert _hub_snapshot() == {}


def test_a_desktop_token_on_ws_edge_is_refused_and_registers_nothing(tmp_path, monkeypatch):
    client = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        code, _reason = _refusal(client, "/ws/edge", _bearer(DESKTOP_TOKEN))
        assert code == CLOSE_POLICY_VIOLATION
        assert _hub_snapshot() == {}


def test_an_edge_token_on_ws_desktop_is_refused_and_registers_nothing(tmp_path, monkeypatch):
    client = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        refusal = _desktop_refusal(client, "/ws/desktop", _bearer(EDGE_TOKEN))
        assert refusal == (REFUSAL_STATUS, REFUSAL_TOKEN)
        assert _hub_snapshot() == {}


def test_missing_header_query_token_unknown_and_revoked_tokens_get_the_same_refusal(
    tmp_path, monkeypatch
):
    client = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        refusals = {
            "missing header": _desktop_refusal(client, "/ws/desktop", {}),
            "query parameter only": _desktop_refusal(
                client, f"/ws/desktop?token={DESKTOP_TOKEN}", {}
            ),
            "unknown token": _desktop_refusal(client, "/ws/desktop", _bearer("t-9")),
            "revoked token": _desktop_refusal(client, "/ws/desktop", _bearer(REVOKED_TOKEN)),
        }
        # Every case is an HTTP 403 that carries the ATLAS refusal marker (D-30).
        assert set(refusals.values()) == {(REFUSAL_STATUS, REFUSAL_TOKEN)}, refusals
        assert _hub_snapshot() == {}


def test_a_server_without_the_denial_extension_falls_back_to_a_plain_1008_close():
    sent: list[dict] = []

    async def receive():
        return {"type": "websocket.connect"}

    async def send(message):
        sent.append(message)

    scope = {"type": "websocket", "headers": [], "extensions": {}}
    asyncio.run(handle_desktop_token_refused(WebSocket(scope, receive, send), Exception()))
    assert sent == [
        {"type": "websocket.close", "code": CLOSE_POLICY_VIOLATION, "reason": ""},
    ]


def test_an_unknown_message_type_is_ignored_and_the_socket_stays_open(tmp_path, monkeypatch):
    client = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        with client.websocket_connect("/ws/desktop", headers=_bearer(DESKTOP_TOKEN)) as socket:
            socket.send_text(json.dumps(_fixture_message("hello.json")))
            assert json.loads(socket.receive_text())["type"] == "hello.ack"
            socket.send_text(json.dumps(_fixture_message("unknown_client.json")))
            socket.send_text(json.dumps(_fixture_message("ping_client.json")))
            pong = json.loads(socket.receive_text())
            assert pong == _fixture_message("pong_server.json")


def test_a_hello_that_is_not_valid_json_is_refused_with_bad_hello(tmp_path, monkeypatch):
    client = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        with client.websocket_connect("/ws/desktop", headers=_bearer(DESKTOP_TOKEN)) as socket:
            socket.send_text("not json")
            assert json.loads(socket.receive_text())["code"] == "bad_hello"
            with pytest.raises(WebSocketDisconnect) as exc_info:
                socket.receive_text()
            assert exc_info.value.code == CLOSE_POLICY_VIOLATION
