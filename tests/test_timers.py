"""Quick task 260930-06x: timers and alarms.

Task 1 is the tracer: one spoken timer goes from the tool, to a repository,
to the poller, to a ring on the house speaker, through the real `lifespan`.
"""

from __future__ import annotations

import time
from datetime import datetime, timedelta, timezone

import pytest

from atlas.mcp_client import McpToolHostLookup
from atlas.timers.core import announcement, clock_text, spoken_duration
from atlas.timers.scheduler import RING_REPEATS, TimerScheduler
from atlas.timers.tool import TimerToolHost
from tests.timer_fakes import FakeTimerRepository

T0 = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)


def _host(repo, *, now=T0, zone=None):
    return TimerToolHost(repo, zone=zone, clock=lambda: now)


def _scheduler(repo, clock_box, spoken, *, zone=None):
    async def speak(text: str) -> None:
        spoken.append(text)

    return TimerScheduler(repo, speak, zone=zone, clock=lambda: clock_box[0])


async def test_set_timer_stores_a_timer_through_the_tool_lookup():
    repo = FakeTimerRepository()
    lookup = McpToolHostLookup([_host(repo)])

    result = await lookup.call_tool("set_timer", {"duration_seconds": 600, "label": "pasta"})

    assert not result.is_error
    assert result.structured_content["id"] == 1
    [timer] = await repo.list_timers()
    assert timer.kind == "timer"
    assert timer.label == "pasta"
    assert timer.due_at == T0 + timedelta(seconds=600)
    assert timer.duration_s == 600


async def test_a_timer_rings_three_times_at_its_due_time_and_is_then_gone():
    repo = FakeTimerRepository()
    await _host(repo).call_tool("set_timer", {"duration_seconds": 600, "label": "pasta"})
    clock_box = [T0 + timedelta(seconds=599)]
    spoken: list[str] = []
    scheduler = _scheduler(repo, clock_box, spoken)

    await scheduler._poll_once()
    assert spoken == []

    clock_box[0] = T0 + timedelta(seconds=600)
    await scheduler._poll_once()
    assert spoken == ["Your pasta timer is done."] * RING_REPEATS
    assert await repo.list_timers() == []

    await scheduler._poll_once()
    assert len(spoken) == RING_REPEATS
    assert scheduler.fired_count == 1


async def test_a_stale_timer_is_cleared_without_speaking():
    repo = FakeTimerRepository()
    await _host(repo).call_tool("set_timer", {"duration_seconds": 60})
    clock_box = [T0 + timedelta(seconds=60 + 601)]
    spoken: list[str] = []

    await _scheduler(repo, clock_box, spoken)._poll_once()

    assert spoken == []
    assert await repo.list_timers() == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"duration_seconds": 0},
        {"duration_seconds": 86401},
        {"duration_seconds": 60, "label": "pa\nsta"},
        {"duration_seconds": 60, "label": "<b>"},
        {"duration_seconds": 60, "extra": 1},
    ],
)
async def test_set_timer_refuses_bad_input_and_stores_nothing(arguments):
    repo = FakeTimerRepository()

    result = await _host(repo).call_tool("set_timer", arguments)

    assert result.is_error
    assert await repo.list_timers() == []


async def test_set_timer_refuses_the_51st_entry():
    repo = FakeTimerRepository()
    host = _host(repo)
    for _ in range(50):
        assert not (await host.call_tool("set_timer", {"duration_seconds": 60})).is_error

    result = await host.call_tool("set_timer", {"duration_seconds": 60})

    assert result.is_error
    assert "50" in result.content[0].text
    assert len(await repo.list_timers()) == 50


async def test_advance_fired_fake_rules():
    repo = FakeTimerRepository()
    await _host(repo).call_tool("set_timer", {"duration_seconds": 60})

    assert await repo.advance_fired(1, now=T0, next_due_at=None) is False
    assert await repo.advance_fired(99, now=T0, next_due_at=None) is False
    due = T0 + timedelta(seconds=60)
    assert await repo.advance_fired(1, now=due, next_due_at=None) is True
    assert await repo.advance_fired(1, now=due, next_due_at=None) is False


def test_text_helpers():
    assert spoken_duration(600) == "10 minutes"
    assert spoken_duration(432) == "7 minutes 12 seconds"
    assert spoken_duration(3900) == "1 hour 5 minutes"
    assert spoken_duration(86400) == "24 hours"
    assert spoken_duration(1) == "1 second"
    assert clock_text("07:00") == "7:00 AM"
    assert clock_text("00:30") == "12:30 AM"
    assert clock_text("12:00") == "12:00 PM"
    assert clock_text("23:59") == "11:59 PM"


# ---------------------------------------------------------------------------
# The tracer through the real lifespan
# ---------------------------------------------------------------------------


def _boot_with_timer_repo(tmp_path, monkeypatch, repo):
    import atlas.app as app_module
    import test_auth_setup

    client = test_auth_setup._boot_with_empty_accounts(tmp_path, monkeypatch)
    original = app_module._build_repositories

    def _with_timer_repo(config, engine):
        repositories = original(config, engine)
        repositories["timer_repo"] = repo
        return repositories

    monkeypatch.setattr(app_module, "_build_repositories", _with_timer_repo)
    return client


def test_lifespan_exposes_set_timer_and_rings_a_due_timer_on_the_camera_source(tmp_path, monkeypatch):
    import asyncio

    import atlas.app as app_module

    repo = FakeTimerRepository()
    asyncio.run(
        repo.create_timer(
            kind="timer",
            label="",
            due_at=datetime.now(timezone.utc) + timedelta(seconds=1.5),
            remaining_s=None,
            duration_s=1,
            time_of_day=None,
            repeat_days=0,
            enabled=True,
            created_at=datetime.now(timezone.utc),
        )
    )
    spoken: list[tuple[object, str]] = []
    cues: list[object] = []

    async def fake_speak(source, tts, timings, text, **kwargs):
        spoken.append((source, text))

    async def fake_cue(source, sink, lock):
        cues.append(source)

    monkeypatch.setattr(app_module, "_speak", fake_speak)
    monkeypatch.setattr(app_module, "_play_wake_cue", fake_cue)
    # The smoke boot's fake camera source has no `sink_format`, and its TTS
    # slot is degraded. The timer is due 1.5 s after boot, so the test gives
    # both a stand-in before the first ring.
    import test_startup_smoke as smoke
    from atlas.providers.tts_xai import SinkFormat

    monkeypatch.setattr(
        smoke._FakeCameraSource, "sink_format", lambda self: SinkFormat("alaw", 8000), raising=False
    )
    client = _boot_with_timer_repo(tmp_path, monkeypatch, repo)

    with client:
        names = [tool["function"]["name"] for tool in app_module.app.state.tools_schema]
        assert "set_timer" in names
        app_module.app.state.tts = object()
        deadline = time.monotonic() + 8
        while len(spoken) < RING_REPEATS and time.monotonic() < deadline:
            time.sleep(0.05)
        camera_source = app_module.app.state.camera_source

    assert [text for _source, text in spoken] == ["Your timer is done."] * RING_REPEATS
    assert all(source is camera_source for source, _text in spoken)
    assert cues and all(source is camera_source for source in cues)
