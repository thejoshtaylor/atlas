"""`/api/timers` CRUD (quick task 260930-06x, Q-09).

Boots the real application through `lifespan`, with `FakeTimerRepository`
layered onto the empty-account boot helper, the same shape as
`tests/test_edge_devices_route.py`.
"""

from __future__ import annotations

import asyncio
from zoneinfo import ZoneInfo

import pytest

import test_auth_setup
from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig
from tests.timer_fakes import FakeTimerRepository

NY = ZoneInfo("America/New_York")


def _boot(tmp_path, monkeypatch):
    import atlas.app as app_module

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)
    repo = FakeTimerRepository()
    original = app_module._build_repositories

    def _with_timer_repo(config, engine):
        repositories = original(config, engine)
        repositories["timer_repo"] = repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_timer_repo)
    return client, repo


def _create_admin(client) -> None:
    response = client.post(
        "/api/auth/create-admin",
        json={
            "email": "timer-admin@example.invalid",
            "display_name": "Timer Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert response.status_code == 201, response.text


def _viewer_cookie() -> str:
    import atlas.app as app_module

    viewer = asyncio.run(
        app_module.app.state.account_repo.create_user(
            email="timer-viewer@example.invalid",
            display_name="Timer Viewer",
            password_hash="not-checked-by-this-test",
            role="viewer",
        )
    )
    return issue_access_token(user_id=viewer.id, role="viewer", security=SecurityConfig())


def _with_zone():
    import atlas.app as app_module

    app_module.app.state.server_timezone = NY


_TIMER = {"kind": "timer", "label": "pasta", "duration_seconds": 600}
_ALARM = {"kind": "alarm", "label": "work", "time": "07:00", "days": ["mon", "tue", "wed", "thu", "fri"]}


def test_create_a_timer_returns_201_with_a_response(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)

        response = client.post("/api/timers", json=_TIMER)

        assert response.status_code == 201, response.text
        body = response.json()
        assert body["kind"] == "timer"
        assert body["label"] == "pasta"
        assert body["paused"] is False
        assert body["remaining_seconds"] in (599, 600)
        assert body["duration_seconds"] == 600
        assert body["next_fire_at"].endswith("Z") or body["next_fire_at"].endswith("+00:00")


def test_create_an_alarm_and_the_no_zone_refusal(tmp_path, monkeypatch):
    client, repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        import atlas.app as app_module

        app_module.app.state.server_timezone = None
        refused = client.post("/api/timers", json=_ALARM)
        assert refused.status_code == 400
        assert "server.timezone" in refused.json()["detail"]
        assert asyncio.run(repo.list_timers()) == []

        _with_zone()
        response = client.post("/api/timers", json=_ALARM)
        assert response.status_code == 201, response.text
        body = response.json()
        assert body["time"] == "07:00"
        assert body["days"] == ["mon", "tue", "wed", "thu", "fri"]
        assert body["enabled"] is True
        assert body["next_fire_at"] is not None
        assert body["remaining_seconds"] is None


def test_list_get_and_unknown_id(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        _with_zone()
        timer_id = client.post("/api/timers", json=_TIMER).json()["id"]
        client.post("/api/timers", json=_ALARM)

        listed = client.get("/api/timers")
        assert listed.status_code == 200
        assert [entry["kind"] for entry in listed.json()] == ["timer", "alarm"]
        assert client.get(f"/api/timers/{timer_id}").json()["label"] == "pasta"
        assert client.get("/api/timers/999").status_code == 404


def test_patch_pause_off_time_cross_kind_and_unknown(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        _with_zone()
        timer_id = client.post("/api/timers", json=_TIMER).json()["id"]
        alarm = client.post("/api/timers", json=_ALARM).json()

        paused = client.patch(f"/api/timers/{timer_id}", json={"paused": True})
        assert paused.status_code == 200, paused.text
        assert paused.json()["paused"] is True
        assert paused.json()["remaining_seconds"] in (599, 600)
        assert paused.json()["next_fire_at"] is None

        off = client.patch(f"/api/timers/{alarm['id']}", json={"enabled": False})
        assert off.json()["enabled"] is False
        assert off.json()["next_fire_at"] is None

        moved = client.patch(f"/api/timers/{alarm['id']}", json={"enabled": True, "time": "08:00"})
        assert moved.json()["time"] == "08:00"
        assert moved.json()["next_fire_at"] is not None

        assert client.patch(f"/api/timers/{timer_id}", json={"time": "08:00"}).status_code == 400
        assert client.patch(f"/api/timers/{alarm['id']}", json={"paused": True}).status_code == 400
        assert client.patch("/api/timers/999", json={"label": "x"}).status_code == 404


def test_delete_then_get_is_404(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        timer_id = client.post("/api/timers", json=_TIMER).json()["id"]

        assert client.delete(f"/api/timers/{timer_id}").status_code == 204
        assert client.get(f"/api/timers/{timer_id}").status_code == 404
        assert client.delete(f"/api/timers/{timer_id}").status_code == 404


@pytest.mark.parametrize(
    "body",
    [
        {"kind": "timer", "duration_seconds": 0},
        {"kind": "timer", "duration_seconds": 60, "label": "pa\nsta"},
        {"kind": "timer", "duration_seconds": 60, "extra": 1},
        {"kind": "alarm", "time": "25:00"},
    ],
)
def test_bad_bodies_are_422(tmp_path, monkeypatch, body):
    client, repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        _with_zone()

        assert client.post("/api/timers", json=body).status_code == 422
        assert asyncio.run(repo.list_timers()) == []


def test_the_51st_entry_is_409(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        for _ in range(50):
            assert client.post("/api/timers", json={"kind": "timer", "duration_seconds": 60}).status_code == 201

        response = client.post("/api/timers", json={"kind": "timer", "duration_seconds": 60})

        assert response.status_code == 409
        assert "50" in response.json()["detail"]


def test_a_viewer_gets_403(tmp_path, monkeypatch):
    client, _repo = _boot(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        client.cookies.set(SecurityConfig().cookie_name, _viewer_cookie())

        assert client.get("/api/timers").status_code == 403
        assert client.post("/api/timers", json=_TIMER).status_code == 403
        assert client.patch("/api/timers/1", json={"label": "x"}).status_code == 403
        assert client.delete("/api/timers/1").status_code == 403
