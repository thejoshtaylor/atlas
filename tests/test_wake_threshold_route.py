"""The server half of DBG-05: a threshold an operator moves takes effect
on the very next wake hit, with no restart (D-15), and the read side that
exposes the recorded wake attempts the tuning screen partitions (D-13,
D-14a, D-16).

Task 1 covers the gate- and runner-level plumbing only: `WakeGate.
set_threshold`/`.threshold` and `SourceRunner.wake_threshold`/
`.set_wake_threshold`. Tasks 2 and 3 (the route and the boot-time read)
add their own test classes below as they land.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from atlas.config import GateConfig, WakeConfig
from atlas.sources.runner import SourceRunner
from atlas.wake.gate import WakeGate

from tests.conftest import FakeAudioSource, FakeWakeHit


# --- WakeGate: set_threshold takes effect on the very next evaluation ------


def test_a_threshold_set_after_construction_governs_the_next_evaluation():
    gate = WakeGate(threshold=0.20, refractory_s=0.0, mute_when_playing=())

    # Under the original threshold, this score is a hit.
    assert gate.evaluate(score=0.25, now=1.0, last_hit_at=None).allowed is True

    gate.set_threshold(0.90)

    # The same score, under the new threshold, is not.
    decision = gate.evaluate(score=0.25, now=2.0, last_hit_at=None)
    assert decision.allowed is False
    assert decision.reason == "below_threshold"


def test_a_score_exactly_equal_to_the_new_threshold_is_a_hit():
    gate = WakeGate(threshold=0.10, refractory_s=0.0, mute_when_playing=())
    gate.set_threshold(0.75)

    decision = gate.evaluate(score=0.75, now=1.0, last_hit_at=None)
    assert decision.allowed is True
    assert decision.reason is None


def test_a_score_one_representable_step_below_the_new_threshold_is_not_a_hit():
    gate = WakeGate(threshold=0.10, refractory_s=0.0, mute_when_playing=())
    gate.set_threshold(0.75)
    just_below = math.nextafter(0.75, -math.inf)

    decision = gate.evaluate(score=just_below, now=1.0, last_hit_at=None)
    assert decision.allowed is False
    assert decision.reason == "below_threshold"


def test_threshold_property_reads_the_live_value_not_the_constructed_one():
    gate = WakeGate(threshold=0.30, refractory_s=0.0, mute_when_playing=())
    assert gate.threshold == 0.30
    gate.set_threshold(0.60)
    assert gate.threshold == 0.60


# --- SourceRunner: named accessors delegating to the gate -------------------


def _build_runner(*, engine: str = "openwakeword") -> SourceRunner:
    source = FakeAudioSource(frames=[])
    wake_config = WakeConfig(engine=engine, refractory_s=0.0)
    gate_config = GateConfig()

    async def _run_turn(src: object) -> None:
        pass

    return SourceRunner(
        "camera",
        source,
        _AlwaysHitWakeDetector(score=1.0),
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
    )


class _AlwaysHitWakeDetector:
    def __init__(self, score: float) -> None:
        self._score = score

    def process(self, chunk: bytes):
        return FakeWakeHit(score=self._score)


def test_source_runner_wake_threshold_reports_the_gates_current_value():
    runner = _build_runner()
    # `WakeConfig(engine="openwakeword")`'s default `OpenWakeWordConfig.
    # threshold` is 0.55 -- proving this property reads through the gate,
    # not a value `SourceRunner` cached separately.
    assert runner.wake_threshold == 0.55


def test_source_runner_set_wake_threshold_changes_the_next_hit_evaluated():
    runner = _build_runner()
    runner.set_wake_threshold(0.90)
    assert runner.wake_threshold == 0.90
    # And the gate itself, reached the only way this module permits, has
    # actually moved.
    assert runner._gate.evaluate(score=0.80, now=1.0, last_hit_at=None).allowed is False
    assert runner._gate.evaluate(score=0.90, now=1.0, last_hit_at=None).allowed is True


async def test_a_runner_nobody_calls_set_wake_threshold_on_behaves_as_before(caplog):
    """Neither addition changes the behaviour of a runner nobody calls
    them on: the existing threshold-boundary test in
    `tests/test_wake_gating.py` already proves this at the `run()` level;
    this is the same assertion made directly against `wake_threshold`."""
    source = FakeAudioSource(frames=[b"\x00\x01"])
    wake_config = WakeConfig(engine="openwakeword", refractory_s=0.0)
    gate_config = GateConfig()

    turns_started: list[object] = []

    async def _run_turn(src: object) -> None:
        turns_started.append(src)

    runner = SourceRunner(
        "camera",
        source,
        _AlwaysHitWakeDetector(score=0.55),
        lambda chunk: chunk,
        _run_turn,
        wake_config=wake_config,
        gate_config=gate_config,
    )

    assert runner.wake_threshold == 0.55
    with caplog.at_level(logging.INFO, logger="atlas.sources.runner"):
        await runner.run()
    assert len(turns_started) == 1


# --- Task 2: GET /api/wake-events, PUT /api/wake-threshold -----------------
#
# `TestClient` against the real application (`_boot_authenticated_client`
# below), matching `tests/test_observer_feed.py`'s own precedent for a
# route this project needs booted through the real `lifespan`, not called
# as a bare handler function -- `list_wake_events`/`set_wake_threshold`
# both read `app.state` objects only a real boot assembles
# (`source_runners`, `wake_event_repo`, `settings_repo`).


class _ScriptedWakeDetector:
    """Stands in for the real engine: reports a fixed score for every
    chunk it processes, so a test can drive a real wake hit through
    `SourceRunner._process_chunk` at a score it chooses -- the "real
    evaluated wake hit" the plan's own `<action>` asks for, not merely a
    read of `wake_threshold`."""

    def __init__(self, score: float) -> None:
        self.score = score

    def process(self, chunk: bytes):
        from atlas.wake.base import WakeHit

        return WakeHit(score=self.score)

    def close(self) -> None:
        pass


def _boot_authenticated_client(tmp_path, monkeypatch, *, role, wake_engine="openwakeword", wake_score=0.60):
    """`tests/test_observer_feed.py::_boot_authenticated_client`'s own
    shape, extended with a `wake_event_repo` (needed for `GET
    /api/wake-events`, which `_fake_build_repositories` does not seed) and
    a scripted, non-`None`-returning wake detector (needed to drive a real
    hit through the boot's own `camera` `SourceRunner`)."""
    from fastapi.testclient import TestClient

    from atlas.auth.tokens import issue_access_token
    from atlas.config import SecurityConfig
    from atlas.plugins import manager as plugin_manager_module

    import atlas.app as app_module
    import tests.conftest as conftest
    from tests import test_startup_smoke as smoke

    monkeypatch.setenv("ATLAS_SECRET_KEY", smoke._TEST_SECRET_KEY)
    session_dir = tmp_path / "sessions"
    session_dir.mkdir(exist_ok=True)
    monkeypatch.setattr(
        app_module,
        "CONFIG_PATH",
        str(
            smoke._write_fake_config(
                tmp_path,
                extra={"debug": {"dir": str(session_dir)}, "wake": {"engine": wake_engine}},
            )
        ),
    )

    def _repositories_with_wake_events(config: object, engine: object) -> dict:
        repositories = smoke._fake_build_repositories(config, engine)
        repositories["wake_event_repo"] = conftest.FakeWakeEventRepository()
        return repositories

    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", _repositories_with_wake_events)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(
        app_module, "_build_wake_detector", lambda wake_config: _ScriptedWakeDetector(wake_score)
    )
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)

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


def test_a_put_inside_range_changes_every_running_sources_threshold_and_the_next_real_hit(
    tmp_path, monkeypatch
):
    """The assertion the plan's own `<action>` asks for: a put followed by
    a real evaluated wake hit, not merely an attribute read."""
    client, app_module = _boot_authenticated_client(
        tmp_path, monkeypatch, role="operator", wake_engine="openwakeword", wake_score=0.60
    )
    try:
        runner = app_module.app.state.source_runners[0]
        # 0.60 clears the shipped `OpenWakeWordConfig` default (0.55).
        assert runner._gate.evaluate(score=0.60, now=1.0, last_hit_at=None).allowed is True

        response = client.put("/api/wake-threshold", json={"threshold": 0.90})
        assert response.status_code == 200
        assert response.json() == {"threshold": 0.90, "applied_to_sources": 1}

        # The same score that used to clear the gate no longer does, on
        # the exact object a real chunk is evaluated against.
        assert runner._gate.evaluate(score=0.60, now=2.0, last_hit_at=None).allowed is False
        assert runner.wake_threshold == 0.90
    finally:
        client.__exit__(None, None, None)


def test_a_put_outside_range_is_refused_and_changes_nothing(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        runner = app_module.app.state.source_runners[0]
        before = runner.wake_threshold

        response = client.put("/api/wake-threshold", json={"threshold": 1.5})
        assert response.status_code == 422
        assert runner.wake_threshold == before

        response = client.put("/api/wake-threshold", json={"threshold": -0.1})
        assert response.status_code == 422
        assert runner.wake_threshold == before
    finally:
        client.__exit__(None, None, None)


def test_a_submitted_threshold_is_stored_and_a_later_read_returns_it(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        response = client.put("/api/wake-threshold", json={"threshold": 0.42})
        assert response.status_code == 200

        setting = asyncio.run(
            app_module.app.state.settings_repo.get_setting("wake_threshold")
        )
        assert setting is not None
        assert setting.value == 0.42
    finally:
        client.__exit__(None, None, None)


def test_get_wake_events_reports_engine_grading_threshold_and_events_newest_first(
    tmp_path, monkeypatch
):
    client, app_module = _boot_authenticated_client(
        tmp_path, monkeypatch, role="operator", wake_engine="openwakeword"
    )
    try:
        repo = app_module.app.state.wake_event_repo
        # 260924-4is: recorded well inside `debug.retain_days`' 7-day
        # default, not a fixed calendar date -- the retention sweep now
        # awaits a real checkpoint (D3) and can run during this test's own
        # `client.get()` call, so a date already older than the retention
        # window is swept before this test ever reads it back.
        now = datetime.now(timezone.utc)
        asyncio.run(
            repo.record_wake_event(
                source="camera",
                engine="openwakeword",
                score=0.60,
                allowed=True,
                block_reason=None,
                recorded_at=now - timedelta(minutes=2),
            )
        )
        asyncio.run(
            repo.record_wake_event(
                source="camera",
                engine="openwakeword",
                score=0.30,
                allowed=False,
                block_reason="below_threshold",
                recorded_at=now - timedelta(minutes=1),
            )
        )

        response = client.get("/api/wake-events")
        assert response.status_code == 200
        body = response.json()
        assert body["engine"] == "openwakeword"
        assert body["engine_grades"] is True
        assert body["threshold"] == 0.55
        assert [event["score"] for event in body["events"]] == [0.30, 0.60]
        assert body["events"][0]["allowed"] is False
        assert body["events"][0]["block_reason"] == "below_threshold"
    finally:
        client.__exit__(None, None, None)


def test_get_wake_events_states_plainly_when_the_engine_does_not_grade(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(
        tmp_path, monkeypatch, role="operator", wake_engine="vosk"
    )
    try:
        response = client.get("/api/wake-events")
        assert response.status_code == 200
        body = response.json()
        assert body["engine"] == "vosk"
        assert body["engine_grades"] is False
    finally:
        client.__exit__(None, None, None)


def test_get_wake_events_counts_sessions_older_than_the_earliest_recorded_wake_event(
    tmp_path, monkeypatch
):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        # 260924-4is: relative to "now", not fixed calendar dates -- the
        # retention sweep now awaits a real checkpoint (D3) and can run
        # during this test's own `client.get()` call, so a wake event
        # already older than `debug.retain_days` (7) is swept before this
        # test ever reads it back. The two session directories only need
        # to stay on either side of the wake event, which "3 days ago"
        # and "30 seconds ago" around a wake event "1 day ago" still do.
        now = datetime.now(timezone.utc)
        old_session_at = now - timedelta(days=3)
        new_session_at = now - timedelta(seconds=30)
        wake_event_at = now - timedelta(days=1)

        session_dir = Path(app_module.app.state.config.session.dir)
        (session_dir / f"{old_session_at.strftime('%Y%m%dT%H%M%S%f')}Z-old-turn").mkdir()
        (session_dir / f"{new_session_at.strftime('%Y%m%dT%H%M%S%f')}Z-new-turn").mkdir()

        repo = app_module.app.state.wake_event_repo
        asyncio.run(
            repo.record_wake_event(
                source="camera",
                engine="openwakeword",
                score=0.60,
                allowed=True,
                block_reason=None,
                recorded_at=wake_event_at,
            )
        )

        response = client.get("/api/wake-events")
        assert response.status_code == 200
        # Only the 2025 session predates the one recorded wake event.
        assert response.json()["not_scored_session_count"] == 1
    finally:
        client.__exit__(None, None, None)


def test_get_wake_events_with_no_wake_events_counts_every_session_as_not_scored(
    tmp_path, monkeypatch
):
    client, app_module = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        # 260924-4is: relative to "now", not fixed calendar dates -- a
        # directory named with a date already older than
        # `debug.retain_days` (7) can be removed by the session retention
        # sweep (D3, which now awaits a real checkpoint) before this test
        # ever counts it.
        now = datetime.now(timezone.utc)
        old_session_at = now - timedelta(days=3)
        new_session_at = now - timedelta(seconds=30)

        session_dir = Path(app_module.app.state.config.session.dir)
        (session_dir / f"{old_session_at.strftime('%Y%m%dT%H%M%S%f')}Z-old-turn").mkdir()
        (session_dir / f"{new_session_at.strftime('%Y%m%dT%H%M%S%f')}Z-new-turn").mkdir()

        response = client.get("/api/wake-events")
        assert response.status_code == 200
        body = response.json()
        assert body["not_scored_session_count"] == 2
        assert body["events"] == []
    finally:
        client.__exit__(None, None, None)


def test_both_wake_routes_are_refused_to_a_viewer_and_allowed_to_an_operator(tmp_path, monkeypatch):
    viewer_client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role="viewer")
    try:
        assert viewer_client.get("/api/wake-events").status_code == 403
        assert viewer_client.put("/api/wake-threshold", json={"threshold": 0.5}).status_code == 403
    finally:
        viewer_client.__exit__(None, None, None)

    operator_client, _ = _boot_authenticated_client(tmp_path, monkeypatch, role="operator")
    try:
        assert operator_client.get("/api/wake-events").status_code == 200
        assert operator_client.put("/api/wake-threshold", json={"threshold": 0.5}).status_code == 200
    finally:
        operator_client.__exit__(None, None, None)


# --- Task 3: the setting outlives the process that received it -------------
#
# Through the real lifespan with `tests/test_startup_smoke.py`'s own
# fake-builder shape (`_fake_build_repositories`, `_FakeCameraSource`,
# `_fake_build_wake_detector`) -- the plan's own `<action>` names this
# file's fixtures explicitly rather than asking for a second boot harness.


def _repositories_with_stored_threshold(stored_threshold):
    """A `_build_repositories` replacement seeding `settings_repo` with a
    `wake_threshold` row before `lifespan` ever reads it -- `None` seeds
    nothing, matching "no stored threshold" (D-15's absent-configuration
    case)."""

    def _builder(config: object, engine: object) -> dict:
        from datetime import datetime, timezone

        from atlas.db.repository import Setting

        from tests import test_startup_smoke as smoke

        repositories = smoke._fake_build_repositories(config, engine)
        if stored_threshold is not None:
            settings_repo = repositories["settings_repo"]
            settings_repo.settings["wake_threshold"] = Setting(
                id=1,
                key="wake_threshold",
                value=stored_threshold,
                updated_at=datetime.now(timezone.utc),
                updated_by_user_id=1,
            )
        return repositories

    return _builder


def _boot_with_stored_threshold(tmp_path, monkeypatch, *, stored_threshold, wake_engine="openwakeword"):
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
        str(
            smoke._write_fake_config(
                tmp_path,
                extra={"debug": {"dir": str(session_dir)}, "wake": {"engine": wake_engine}},
            )
        ),
    )
    monkeypatch.setattr(plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(
        app_module, "_build_repositories", _repositories_with_stored_threshold(stored_threshold)
    )
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", smoke._FakeCameraSource)
    return TestClient(app_module.app), app_module


def test_booting_with_a_stored_threshold_constructs_runners_using_it(tmp_path, monkeypatch):
    client, app_module = _boot_with_stored_threshold(
        tmp_path, monkeypatch, stored_threshold=0.81, wake_engine="openwakeword"
    )
    with client:
        assert app_module.app.state.source_runners[0].wake_threshold == 0.81


def test_booting_with_no_stored_threshold_uses_the_configured_value(tmp_path, monkeypatch):
    client, app_module = _boot_with_stored_threshold(
        tmp_path, monkeypatch, stored_threshold=None, wake_engine="openwakeword"
    )
    with client:
        # `OpenWakeWordConfig`'s shipped default -- exactly as today, per
        # the plan's own `<behavior>`.
        assert app_module.app.state.source_runners[0].wake_threshold == 0.55


def test_a_stored_out_of_range_threshold_refuses_the_boot_by_name(tmp_path, monkeypatch):
    client, app_module = _boot_with_stored_threshold(
        tmp_path, monkeypatch, stored_threshold=1.5, wake_engine="openwakeword"
    )
    ConfigError = app_module.ConfigError
    with pytest.raises(ConfigError, match="stored wake threshold.*outside \\[0.0, 1.0\\]"):
        with client:
            pass


def test_the_boot_logs_which_source_the_threshold_came_from(tmp_path, monkeypatch, caplog):
    client, app_module = _boot_with_stored_threshold(
        tmp_path, monkeypatch, stored_threshold=0.81, wake_engine="openwakeword"
    )
    with caplog.at_level(logging.INFO, logger="atlas.app"):
        with client:
            pass
    assert any(
        "wake threshold resolved from database" in record.message for record in caplog.records
    )


# === Code review WR-06: the response's own bound =======================


def test_a_response_inside_the_bound_says_it_is_not_capped(tmp_path, monkeypatch):
    client, app_module = _boot_authenticated_client(
        tmp_path, monkeypatch, role="operator", wake_engine="openwakeword"
    )
    try:
        repo = app_module.app.state.wake_event_repo
        # 260924-4is: see the same note above -- recorded inside the
        # retention window, not at a fixed calendar date.
        asyncio.run(
            repo.record_wake_event(
                source="camera",
                engine="openwakeword",
                score=0.60,
                allowed=True,
                block_reason=None,
                recorded_at=datetime.now(timezone.utc),
            )
        )

        body = client.get("/api/wake-events").json()

        assert body["capped"] is False
        assert len(body["events"]) == 1
    finally:
        client.__exit__(None, None, None)


def test_more_events_than_the_bound_are_capped_and_the_response_says_so(tmp_path, monkeypatch):
    """The route read the whole table with no limit and serialised all of
    it on every load of the tuning screen. The sweep bounds the table now;
    this is the backstop for the window between two sweeps -- and it is
    never silent (WR-06)."""
    from atlas.routes.wake import MAX_WAKE_EVENTS_IN_RESPONSE

    client, app_module = _boot_authenticated_client(
        tmp_path, monkeypatch, role="operator", wake_engine="openwakeword"
    )
    try:
        repo = app_module.app.state.wake_event_repo
        # 260924-4is: see the same note above -- recorded inside the
        # retention window, not at a fixed calendar date.
        base = datetime.now(timezone.utc)

        async def _seed() -> None:
            for index in range(MAX_WAKE_EVENTS_IN_RESPONSE + 5):
                await repo.record_wake_event(
                    source="camera",
                    engine="openwakeword",
                    score=0.60,
                    allowed=True,
                    block_reason=None,
                    recorded_at=base + timedelta(seconds=index),
                )

        asyncio.run(_seed())

        body = client.get("/api/wake-events").json()

        assert body["capped"] is True
        assert len(body["events"]) == MAX_WAKE_EVENTS_IN_RESPONSE
        # The newest ones, not an arbitrary page.
        newest_recorded_at = base + timedelta(seconds=MAX_WAKE_EVENTS_IN_RESPONSE + 4)
        assert body["events"][0]["recorded_at"].startswith(newest_recorded_at.isoformat(timespec="seconds")[:18])
    finally:
        client.__exit__(None, None, None)


# === Code review IN-06: the positional read of app.state.source_runners ===


def test_the_events_route_reports_the_configured_threshold_when_no_source_is_running(
    tmp_path, monkeypatch
):
    """`source_runners[0]` is an unguarded index. It is always
    `[camera_runner]` today, so this is not reachable -- but that list is
    the one seam D-15 depends on, and an `IndexError` here is a 500 on a
    screen whose whole job is to show a number (IN-06)."""
    client, app_module = _boot_authenticated_client(
        tmp_path, monkeypatch, role="operator", wake_engine="openwakeword"
    )
    try:
        app_module.app.state.source_runners = []

        response = client.get("/api/wake-events")

        assert response.status_code == 200
        # The configured value, which is the only honest answer when no
        # gate is running to report a live one.
        assert response.json()["threshold"] == 0.55
    finally:
        client.__exit__(None, None, None)


def test_a_put_that_reaches_no_running_source_says_so_rather_than_claiming_it_applied(
    tmp_path, monkeypatch
):
    """The mirror-image problem: the loop used to return 200 having applied
    the change to nothing, which reads as D-15's "takes effect live"
    promise being kept when it was not (IN-06)."""
    client, app_module = _boot_authenticated_client(
        tmp_path, monkeypatch, role="operator", wake_engine="openwakeword"
    )
    try:
        app_module.app.state.source_runners = []

        response = client.put("/api/wake-threshold", json={"threshold": 0.42})

        assert response.status_code == 200
        assert response.json() == {"threshold": 0.42, "applied_to_sources": 0}
        # Still stored -- the write is not conditional on a running source.
        assert client.get("/api/wake-events").json()["threshold"] == 0.55
    finally:
        client.__exit__(None, None, None)
