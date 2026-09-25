"""Plan 09-07 Task 2: an admin changes how long ATLAS listens for the
answer to a calendar readback or a clarifying question (D-09, T-09-40,
T-09-41) -- `GET`/`PUT /api/settings/follow-up-window`, resolved once at
boot the same database-then-configuration way `_resolve_wake_threshold`
already resolves the wake threshold (`tests/test_wake_threshold_route.py`'s
own Task 3 precedent), and applied live through `app.state.
follow_up_window_s`, the one attribute both `SourceRunner` constructions in
`app.py` read.
"""

from __future__ import annotations

import asyncio

import pytest


# --- Route-level: GET/PUT, range, roles, live effect ------------------------
#
# `TestClient` against the real application, mirroring
# `tests/test_wake_threshold_route.py::_boot_authenticated_client` -- this
# route reads `app.state.settings_repo`/`app.state.follow_up_window_s`/
# `app.state.source_runners`, all of which only a real boot assembles.


def _boot_authenticated_client(tmp_path, monkeypatch, *, role):
    from fastapi.testclient import TestClient

    from atlas.auth.tokens import issue_access_token
    from atlas.config import SecurityConfig
    from atlas.plugins import manager as plugin_manager_module

    import atlas.app as app_module
    from tests import test_startup_smoke as smoke

    monkeypatch.setenv("ATLAS_SECRET_KEY", smoke._TEST_SECRET_KEY)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(smoke._write_fake_config(tmp_path, extra={"debug": {"dir": str(session_dir)}})),
    )
    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", smoke._fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", smoke._FakeCameraSource)

    client = TestClient(app_module.app)
    client.__enter__()
    security = SecurityConfig()
    if role is not None:
        if role == "admin":
            user_id = 1
        else:
            from datetime import datetime, timezone

            from atlas.db.repository import User

            account_repo = app_module.app.state.account_repo
            user_id = account_repo._next_user_id
            account_repo._next_user_id += 1
            account_repo.users[user_id] = User(
                id=user_id,
                email=f"{role}@example.invalid",
                display_name=role,
                password_hash="not-a-real-hash-never-checked",
                role=role,
                created_at=datetime.now(timezone.utc),
                disabled_at=None,
            )
        token = issue_access_token(user_id=user_id, role=role, security=security)
        client.cookies.set(security.cookie_name, token)
    return client, app_module


def test_get_with_nothing_stored_returns_the_configured_default(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="admin")
    try:
        response = client.get("/api/settings/follow-up-window")
        assert response.status_code == 200
        body = response.json()
        assert body == {"window_s": 6.0, "default_s": 6.0, "resolved_from": "config"}
    finally:
        client.__exit__(None, None, None)


def test_a_put_inside_range_stores_it_and_the_very_next_window_uses_it_live(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="admin")
    try:
        runner = app_module.app.state.source_runners[0]
        assert runner._follow_up_window_s() == 6.0

        response = client.put("/api/settings/follow-up-window", json={"window_s": 8})
        assert response.status_code == 200
        assert response.json() == {"window_s": 8.0, "default_s": 6.0, "resolved_from": "database"}

        # Live, no restart: the exact `SourceRunner` already running reads
        # the new value the very next time its own callable is invoked.
        assert runner._follow_up_window_s() == 8.0
        assert app_module.app.state.follow_up_window_s == 8.0

        assert client.get("/api/settings/follow-up-window").json()["resolved_from"] == "database"

        setting = asyncio.run(app_module.app.state.settings_repo.get_setting("follow_up_window_s"))
        assert setting is not None
        assert setting.value == 8.0
    finally:
        client.__exit__(None, None, None)


@pytest.mark.parametrize("bad_value", [2.9, 15.1, True, "8", None])
def test_a_put_outside_range_a_boolean_or_a_string_is_refused_with_400_naming_the_range(
    tmp_path, monkeypatch, bad_value
):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="admin")
    try:
        before = app_module.app.state.follow_up_window_s

        response = client.put("/api/settings/follow-up-window", json={"window_s": bad_value})

        assert response.status_code == 400
        assert "3" in response.json()["detail"]
        assert "15" in response.json()["detail"]
        assert app_module.app.state.follow_up_window_s == before
    finally:
        client.__exit__(None, None, None)


def test_a_put_with_an_extra_field_is_refused(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="admin")
    try:
        response = client.put(
            "/api/settings/follow-up-window", json={"window_s": 8, "echo_tail_ms": 100}
        )
        assert response.status_code == 422
    finally:
        client.__exit__(None, None, None)


def test_boundary_values_three_and_fifteen_are_both_accepted(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="admin")
    try:
        assert client.put("/api/settings/follow-up-window", json={"window_s": 3}).status_code == 200
        assert client.put("/api/settings/follow-up-window", json={"window_s": 15}).status_code == 200
    finally:
        client.__exit__(None, None, None)


def test_both_routes_are_403_for_viewer_and_operator_and_200_for_admin(tmp_path, monkeypatch):
    for role in ("viewer", "operator"):
        client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role=role)
        try:
            assert client.get("/api/settings/follow-up-window").status_code == 403
            assert client.put("/api/settings/follow-up-window", json={"window_s": 8}).status_code == 403
        finally:
            client.__exit__(None, None, None)

    client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role="admin")
    try:
        assert client.get("/api/settings/follow-up-window").status_code == 200
        assert client.put("/api/settings/follow-up-window", json={"window_s": 8}).status_code == 200
    finally:
        client.__exit__(None, None, None)


# --- Boot-time resolution: the stored value outlives the process -----------


def _repositories_with_stored_window(stored_value):
    """A `_build_repositories` replacement seeding `settings_repo` with a
    `follow_up_window_s` row before `lifespan` ever reads it -- `None`
    seeds nothing, matching "no stored value" (the shipped-default case)."""

    def _builder(config: object, engine: object) -> dict:
        from datetime import datetime, timezone

        from atlas.db.repository import Setting

        from tests import test_startup_smoke as smoke

        repositories = smoke._fake_build_repositories(config, engine)
        if stored_value is not None:
            settings_repo = repositories["settings_repo"]
            settings_repo.settings["follow_up_window_s"] = Setting(
                id=1,
                key="follow_up_window_s",
                value=stored_value,
                updated_at=datetime.now(timezone.utc),
                updated_by_user_id=1,
            )
        return repositories

    return _builder


def _boot_with_stored_window(tmp_path, monkeypatch, *, stored_value):
    from fastapi.testclient import TestClient
    from atlas.plugins import manager as plugin_manager_module

    import atlas.app as app_module
    from tests import test_startup_smoke as smoke

    monkeypatch.setenv("ATLAS_SECRET_KEY", smoke._TEST_SECRET_KEY)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(smoke._write_fake_config(tmp_path, extra={"debug": {"dir": str(session_dir)}})),
    )
    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _repositories_with_stored_window(stored_value))
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", smoke._FakeCameraSource)
    return TestClient(app_module.app), app_module


def test_booting_with_a_stored_value_resolves_from_the_database(tmp_path, monkeypatch):
    client, app_module = _boot_with_stored_window(tmp_path, monkeypatch, stored_value=9.5)
    with client:
        assert app_module.app.state.follow_up_window_s == 9.5
        assert app_module.app.state.source_runners[0]._follow_up_window_s() == 9.5


def test_booting_with_no_stored_value_uses_the_configured_default(tmp_path, monkeypatch):
    client, app_module = _boot_with_stored_window(tmp_path, monkeypatch, stored_value=None)
    with client:
        assert app_module.app.state.follow_up_window_s == 6.0


def test_a_stored_out_of_range_value_never_refuses_the_boot_and_falls_back_to_config(
    tmp_path, monkeypatch, caplog
):
    import logging

    client, app_module = _boot_with_stored_window(tmp_path, monkeypatch, stored_value=20.0)
    with caplog.at_level(logging.WARNING, logger="atlas.app"):
        with client:
            # A boot never refuses over a setting the route already
            # validated (T-09-41's own doctrine) -- unlike the wake
            # threshold's `ConfigError`, this falls back quietly.
            assert app_module.app.state.follow_up_window_s == 6.0
    assert any("outside [3, 15]" in record.message for record in caplog.records)


def test_the_boot_logs_which_source_the_window_came_from(tmp_path, monkeypatch, caplog):
    import logging

    client, app_module = _boot_with_stored_window(tmp_path, monkeypatch, stored_value=9.5)
    with caplog.at_level(logging.INFO, logger="atlas.app"):
        with client:
            pass
    assert any("follow-up window resolved from database" in record.message for record in caplog.records)


def test_the_listen_websocket_runner_is_wired_to_the_same_live_app_state_value():
    """`/ws/listen`'s own `SourceRunner` construction (`app.py`) reads
    `websocket.app.state.follow_up_window_s` too -- this application has
    exactly two `SourceRunner` constructions (the camera, at boot, and
    the browser listener, per connection) and both must read the one
    live value `PUT /api/settings/follow-up-window` writes; asserted
    against the source text directly, the same "read the wiring, not a
    second mechanism" discipline `tests/test_config.py`'s own
    `parse_wake_default_evidence`-adjacent source checks already use for
    a fact a full boot would be expensive to prove twice."""
    import atlas.app as app_module

    source_text = open(app_module.__file__).read()
    assert "follow_up_window_s=lambda: websocket.app.state.follow_up_window_s" in source_text
    assert "follow_up_window_s=lambda: app.state.follow_up_window_s" in source_text
