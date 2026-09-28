"""Admin routes to list, add, and delete household members (D-01, D-03,
D-04, T-11-05, T-11-06, T-11-07, T-11-08, Phase 11, plan 11-03).

Boots the real application through `lifespan`, reusing `tests/
test_auth_setup.py`'s `_boot_with_empty_accounts` helper (the same shape
`tests/test_edge_devices_route.py` already uses), with an in-memory
`FakeSpeakerRepository` (`tests/speaker_repo_fakes.py`) layered onto its
own empty-repository dict.
"""

from __future__ import annotations

import asyncio

import pytest
from starlette.testclient import TestClient as StarletteTestClient

import test_auth_setup
from atlas.auth.tokens import issue_access_token
from atlas.config import SecurityConfig

from tests.speaker_repo_fakes import FakeSpeakerRepository


def _boot_with_speaker_repo(tmp_path, monkeypatch):
    """`_boot_with_empty_accounts`'s app, plus a `speaker_repo` its own
    fake repository dict does not carry -- layered on top rather than
    duplicating that helper's whole monkeypatch sequence. Returns the
    (un-entered) `TestClient` and the fake repository the routes below
    read/write through; enter the client as its own context manager
    (`with client:`) to run `lifespan`, the same requirement
    `_boot_with_empty_accounts` already carries."""
    import atlas.app as app_module

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)
    speaker_repo = FakeSpeakerRepository()
    original_build_repositories = app_module._build_repositories

    def _with_speaker_repo(config, engine):
        repositories = original_build_repositories(config, engine)
        repositories["speaker_repo"] = speaker_repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_speaker_repo)
    return client, speaker_repo


def _create_admin(client) -> None:
    response = client.post(
        "/api/auth/create-admin",
        json={
            "email": "speaker-admin@example.invalid",
            "display_name": "Speaker Admin",
            "password": "a-plainly-fictional-test-password",
        },
    )
    assert response.status_code == 201, response.text


def _operator_cookie() -> str:
    """A second, non-admin identity: created directly through the
    already-booted app's own `account_repo`, matching `test_edge_devices_
    route.py::_operator_cookie`'s own shape."""
    import atlas.app as app_module

    security = SecurityConfig()
    operator = asyncio.run(
        app_module.app.state.account_repo.create_user(
            email="speaker-operator@example.invalid",
            display_name="Speaker Operator",
            password_hash="not-checked-by-this-test",
            role="operator",
        )
    )
    return issue_access_token(user_id=operator.id, role="operator", security=security)


def test_create_then_list_returns_the_trimmed_member(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)

        create_response = client.post("/api/speakers", json={"display_name": "  Member A "})
        assert create_response.status_code == 201, create_response.text
        assert create_response.json()["display_name"] == "Member A"
        assert create_response.json()["linked_user_id"] is None

        list_response = client.get("/api/speakers")
        assert list_response.status_code == 200, list_response.text
        [listed] = list_response.json()
        assert listed["display_name"] == "Member A"


@pytest.mark.parametrize("display_name", ["", "   ", "x" * 65, "bad/name", "bad;name"])
def test_a_disallowed_display_name_is_refused(tmp_path, monkeypatch, display_name):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        response = client.post("/api/speakers", json={"display_name": display_name})
        assert response.status_code == 422, response.text


def test_a_second_member_with_the_same_display_name_returns_409_never_500(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        first = client.post("/api/speakers", json={"display_name": "Member A"})
        assert first.status_code == 201, first.text

        second = client.post("/api/speakers", json={"display_name": "Member A"})
        assert second.status_code == 409, second.text
        assert "detail" in second.json()


def test_linked_user_id_of_an_existing_user_is_stored(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)

        import atlas.app as app_module

        linked_user = asyncio.run(
            app_module.app.state.account_repo.create_user(
                email="linked-user@example.invalid",
                display_name="Linked User",
                password_hash="not-checked-by-this-test",
                role="viewer",
            )
        )

        response = client.post(
            "/api/speakers", json={"display_name": "Member A", "linked_user_id": linked_user.id}
        )
        assert response.status_code == 201, response.text
        assert response.json()["linked_user_id"] == linked_user.id


def test_linked_user_id_with_no_such_user_returns_422(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        response = client.post(
            "/api/speakers", json={"display_name": "Member A", "linked_user_id": 999999}
        )
        assert response.status_code == 422, response.text


def test_delete_returns_204_and_the_member_is_gone_from_the_list(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        created = client.post("/api/speakers", json={"display_name": "Member A"}).json()

        delete_response = client.delete(f"/api/speakers/{created['id']}")
        assert delete_response.status_code == 204, delete_response.text

        assert client.get("/api/speakers").json() == []


def test_delete_on_an_unknown_id_returns_404(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        response = client.delete("/api/speakers/999999")
        assert response.status_code == 404, response.text


def test_an_operator_session_gets_403_on_all_three_routes(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        created = client.post("/api/speakers", json={"display_name": "Member A"}).json()

        client.cookies.set(SecurityConfig().cookie_name, _operator_cookie())

        assert client.get("/api/speakers").status_code == 403
        assert client.post("/api/speakers", json={"display_name": "Member B"}).status_code == 403
        assert client.delete(f"/api/speakers/{created['id']}").status_code == 403


def test_no_session_gets_401_on_all_three_routes(tmp_path, monkeypatch):
    client, _repo = _boot_with_speaker_repo(tmp_path, monkeypatch)
    with client:
        _create_admin(client)
        created = client.post("/api/speakers", json={"display_name": "Member A"}).json()

        import atlas.app as app_module

        # A fresh, un-entered `TestClient` around the already-booted app --
        # no session cookie of its own, matching `test_edge_devices_
        # route.py::test_no_session_gets_401`'s own shape.
        anonymous = StarletteTestClient(app_module.app)
        assert anonymous.get("/api/speakers").status_code == 401
        assert anonymous.post("/api/speakers", json={"display_name": "Member B"}).status_code == 401
        assert anonymous.delete(f"/api/speakers/{created['id']}").status_code == 401
