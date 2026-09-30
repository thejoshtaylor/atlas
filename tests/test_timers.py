"""Quick task 260930-06x: timers and alarms.

Task 1 is the tracer: one spoken timer goes from the tool, to a repository,
to the poller, to a ring on the house speaker, through the real `lifespan`.
"""

from __future__ import annotations

import asyncio
import time
from datetime import datetime, timedelta, timezone

import pytest

from atlas.mcp_client import McpToolHostLookup
from atlas.timers.core import announcement, clock_text, spoken_duration
from atlas.timers.scheduler import TimerScheduler
from atlas.timers.tool import TimerToolHost
from tests.timer_fakes import FakeTimerRepository

T0 = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)


async def _poll_and_stop_after(scheduler, spoken, rings=3):
    """Poll once, let the ring repeat `rings` times, then stop it."""
    poll = asyncio.create_task(scheduler._poll_once())
    async with asyncio.timeout(5):
        while len(spoken) < rings and not poll.done():
            await asyncio.sleep(0.005)
    scheduler.stop_ringing()
    await asyncio.wait_for(poll, 5)


def _host(repo, *, now=T0, zone=None):
    return TimerToolHost(repo, zone=zone, clock=lambda: now)


def _scheduler(repo, clock_box, spoken, *, zone=None):
    async def speak(text: str) -> None:
        spoken.append(text)
        await asyncio.sleep(0.005)

    return TimerScheduler(repo, speak, zone=zone, clock=lambda: clock_box[0], ring_gap_s=0.005)


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


async def test_a_timer_rings_at_its_due_time_until_stopped_and_is_then_gone():
    repo = FakeTimerRepository()
    await _host(repo).call_tool("set_timer", {"duration_seconds": 600, "label": "pasta"})
    clock_box = [T0 + timedelta(seconds=599)]
    spoken: list[str] = []
    scheduler = _scheduler(repo, clock_box, spoken)

    await scheduler._poll_once()
    assert spoken == []

    clock_box[0] = T0 + timedelta(seconds=600)
    poll = asyncio.create_task(scheduler._poll_once())
    async with asyncio.timeout(5):
        while len(spoken) < 3:
            await asyncio.sleep(0.005)
    assert scheduler.stop_ringing() is True
    await asyncio.wait_for(poll, 5)
    assert set(spoken) == {"Your pasta timer is done."}
    assert await repo.list_timers() == []

    count = len(spoken)
    await scheduler._poll_once()
    assert len(spoken) == count
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
        runner = app_module.app.state.source_runners[0]
        assert runner.ring_window is not None
        scheduler = app_module.app.state.timer_scheduler
        deadline = time.monotonic() + 15
        while len(spoken) < 3 and time.monotonic() < deadline:
            time.sleep(0.05)
        assert len(spoken) >= 3
        # The test thread is not the loop thread, so the stop goes through the portal.
        assert client.portal.call(scheduler.stop_ringing) is True
        time.sleep(0.3)
        rings_after_stop = len(spoken)
        time.sleep(1.5)
        assert len(spoken) == rings_after_stop
        assert client.portal.call(lambda: scheduler.ringing) is False
        camera_source = app_module.app.state.camera_source

    assert {text for _source, text in spoken} == {"Your timer is done."}
    assert all(source is camera_source for source, _text in spoken)
    assert cues and all(source is camera_source for source in cues)


# ---------------------------------------------------------------------------
# Task 2: alarms, the rest of voice CRUD, and the per-turn context block
# ---------------------------------------------------------------------------

import logging  # noqa: E402
import os  # noqa: E402
from zoneinfo import ZoneInfo  # noqa: E402

from pydantic import ValidationError  # noqa: E402

from atlas.timers import service  # noqa: E402
from atlas.timers.core import (  # noqa: E402
    AlarmSpec,
    TimerChanges,
    TimerError,
    TimerNotFoundError,
    TimerSpec,
    days_to_mask,
    describe_timers,
    next_alarm_at,
)

NY = ZoneInfo("America/New_York")
WEEKDAY_MASK = days_to_mask(["mon", "tue", "wed", "thu", "fri"])


def _utc(*args):
    return datetime(*args, tzinfo=timezone.utc)


def test_next_alarm_at_skips_to_the_next_allowed_weekday():
    # Friday 2027-01-08 08:00 local is 13:00Z. Monday is 2027-01-11.
    now = datetime(2027, 1, 8, 8, 0, tzinfo=NY).astimezone(timezone.utc)

    result = next_alarm_at("07:00", WEEKDAY_MASK, now, NY)

    assert result == datetime(2027, 1, 11, 7, 0, tzinfo=NY).astimezone(timezone.utc)


def test_next_alarm_at_once_is_today_before_the_time_and_strictly_after_now():
    before = datetime(2027, 1, 8, 6, 59, tzinfo=NY).astimezone(timezone.utc)
    assert next_alarm_at("07:00", 0, before, NY) == datetime(2027, 1, 8, 7, 0, tzinfo=NY).astimezone(
        timezone.utc
    )
    exactly = datetime(2027, 1, 8, 7, 0, tzinfo=NY).astimezone(timezone.utc)
    assert next_alarm_at("07:00", 0, exactly, NY) > exactly


def test_next_alarm_at_dst_gap_and_overlap_ignore_the_process_timezone(monkeypatch):
    if not hasattr(time, "tzset"):
        pytest.skip("no tzset on this platform")
    monkeypatch.setenv("TZ", "Asia/Tokyo")
    time.tzset()
    try:
        gap_now = datetime(2026, 3, 8, 0, 0, tzinfo=NY).astimezone(timezone.utc)
        assert next_alarm_at("02:30", 0, gap_now, NY) == _utc(2026, 3, 8, 7, 30)
        overlap_now = datetime(2026, 11, 1, 0, 0, tzinfo=NY).astimezone(timezone.utc)
        assert next_alarm_at("01:30", 0, overlap_now, NY) == _utc(2026, 11, 1, 5, 30)
    finally:
        monkeypatch.undo()
        time.tzset()


async def test_set_alarm_stores_an_alarm_with_its_next_ring():
    repo = FakeTimerRepository()
    now = datetime(2027, 1, 8, 8, 0, tzinfo=NY).astimezone(timezone.utc)
    host = TimerToolHost(repo, zone=NY, clock=lambda: now)

    result = await host.call_tool(
        "set_alarm", {"time": "07:00", "days": ["mon", "tue", "wed", "thu", "fri"], "label": "work"}
    )

    assert not result.is_error
    assert result.content[0].text == "work alarm set for 7:00 AM on mon tue wed thu fri"
    [alarm] = await repo.list_timers()
    assert alarm.kind == "alarm"
    assert alarm.time_of_day == "07:00"
    assert alarm.repeat_days == WEEKDAY_MASK
    assert alarm.due_at == next_alarm_at("07:00", WEEKDAY_MASK, now, NY)


async def test_set_alarm_without_a_zone_names_server_timezone():
    repo = FakeTimerRepository()

    result = await _host(repo, zone=None).call_tool("set_alarm", {"time": "07:00"})

    assert result.is_error
    assert "server.timezone" in result.content[0].text
    assert await repo.list_timers() == []


@pytest.mark.parametrize(
    "arguments",
    [
        {"time": "25:00"},
        {"time": "7:00"},
        {"time": "07:00", "days": ["mon", "mon"]},
        {"time": "07:00", "days": ["funday"]},
    ],
)
async def test_set_alarm_refuses_bad_input(arguments):
    repo = FakeTimerRepository()

    result = await _host(repo, zone=NY).call_tool("set_alarm", arguments)

    assert result.is_error
    assert await repo.list_timers() == []


def test_changes_models_refuse_empty_and_conflicting_changes():
    with pytest.raises(ValidationError):
        TimerChanges()
    with pytest.raises(ValidationError):
        TimerChanges(remaining_seconds=60, add_seconds=60)
    with pytest.raises(ValidationError):
        AlarmSpec(time="07:00", days=["mon", "mon"])


async def _alarm_at_seven(repo, *, zone=NY, now=None, days=()):
    now = now or datetime(2027, 1, 8, 8, 0, tzinfo=NY).astimezone(timezone.utc)
    return await service.create_alarm(
        repo, AlarmSpec(time="07:00", days=list(days), label="work"), now=now, zone=zone
    )


async def test_a_one_time_alarm_rings_then_stays_listed_off():
    repo = FakeTimerRepository()
    now = datetime(2027, 1, 8, 8, 0, tzinfo=NY).astimezone(timezone.utc)
    alarm = await _alarm_at_seven(repo, now=now)
    clock_box = [alarm.due_at]
    spoken: list[str] = []

    await _poll_and_stop_after(_scheduler(repo, clock_box, spoken, zone=NY), spoken)

    assert set(spoken) == {"It's 7:00 AM. Your work alarm."}
    assert len(spoken) >= 3
    [after] = await repo.list_timers()
    assert after.enabled is False
    assert after.due_at is None


async def test_a_repeating_alarm_moves_to_its_next_ring():
    repo = FakeTimerRepository()
    now = datetime(2027, 1, 8, 6, 0, tzinfo=NY).astimezone(timezone.utc)  # Friday
    alarm = await _alarm_at_seven(repo, now=now, days=["mon", "tue", "wed", "thu", "fri"])
    clock_box = [alarm.due_at]
    spoken: list[str] = []

    await _poll_and_stop_after(_scheduler(repo, clock_box, spoken, zone=NY), spoken)

    assert len(spoken) >= 3
    [after] = await repo.list_timers()
    assert after.enabled is True
    assert after.due_at == datetime(2027, 1, 11, 7, 0, tzinfo=NY).astimezone(timezone.utc)


async def test_a_repeating_alarm_without_a_zone_turns_off_and_warns(caplog):
    repo = FakeTimerRepository()
    now = datetime(2027, 1, 8, 6, 0, tzinfo=NY).astimezone(timezone.utc)
    alarm = await _alarm_at_seven(repo, now=now, days=["mon"])
    clock_box = [alarm.due_at]
    spoken: list[str] = []

    with caplog.at_level(logging.WARNING, logger="atlas.timers.scheduler"):
        await _poll_and_stop_after(_scheduler(repo, clock_box, spoken, zone=None), spoken)

    [after] = await repo.list_timers()
    assert after.enabled is False
    assert any("server.timezone" in record.getMessage() for record in caplog.records)


async def _running_timer(repo, *, seconds=600, label="pasta"):
    return await service.create_timer(repo, TimerSpec(duration_seconds=seconds, label=label), now=T0)


async def test_update_timer_pause_resume_and_time_edits():
    repo = FakeTimerRepository()
    timer = await _running_timer(repo)
    later = T0 + timedelta(seconds=100.5)

    paused = await service.update_timer(repo, timer.id, TimerChanges(paused=True), now=later, zone=None)
    assert paused.due_at is None
    assert paused.remaining_s == 500  # ceil(499.5)

    resumed = await service.update_timer(
        repo, timer.id, TimerChanges(paused=False), now=later, zone=None
    )
    assert resumed.due_at == later + timedelta(seconds=500)
    assert resumed.remaining_s is None

    extended = await service.update_timer(
        repo, timer.id, TimerChanges(add_seconds=300), now=later, zone=None
    )
    assert extended.due_at == resumed.due_at + timedelta(seconds=300)
    assert extended.duration_s == 900

    reset = await service.update_timer(
        repo, timer.id, TimerChanges(remaining_seconds=60), now=later, zone=None
    )
    assert reset.due_at == later + timedelta(seconds=60)
    assert reset.duration_s == 60

    renamed = await service.update_timer(repo, timer.id, TimerChanges(label="tea"), now=later, zone=None)
    assert renamed.label == "tea"


async def test_update_timer_refuses_out_of_range_and_cross_kind_and_unknown():
    repo = FakeTimerRepository()
    timer = await _running_timer(repo)
    alarm = await _alarm_at_seven(repo)

    with pytest.raises(TimerError):
        await service.update_timer(repo, timer.id, TimerChanges(add_seconds=86400), now=T0, zone=NY)
    with pytest.raises(TimerError):
        await service.update_timer(repo, timer.id, TimerChanges(add_seconds=-600), now=T0, zone=NY)
    with pytest.raises(TimerError):
        await service.update_timer(repo, timer.id, TimerChanges(time="08:00"), now=T0, zone=NY)
    with pytest.raises(TimerError):
        await service.update_timer(repo, timer.id, TimerChanges(enabled=False), now=T0, zone=NY)
    with pytest.raises(TimerError):
        await service.update_timer(repo, alarm.id, TimerChanges(paused=True), now=T0, zone=NY)
    with pytest.raises(TimerError):
        await service.update_timer(repo, alarm.id, TimerChanges(add_seconds=5), now=T0, zone=NY)
    with pytest.raises(TimerNotFoundError):
        await service.update_timer(repo, 999, TimerChanges(label="x"), now=T0, zone=NY)
    assert (await repo.get_timer(timer.id)) == timer


async def test_update_alarm_time_days_and_on_off():
    repo = FakeTimerRepository()
    now = datetime(2027, 1, 8, 6, 0, tzinfo=NY).astimezone(timezone.utc)
    alarm = await _alarm_at_seven(repo, now=now)

    moved = await service.update_timer(repo, alarm.id, TimerChanges(time="08:00"), now=now, zone=NY)
    assert moved.due_at == datetime(2027, 1, 8, 8, 0, tzinfo=NY).astimezone(timezone.utc)
    assert moved.time_of_day == "08:00"

    repeating = await service.update_timer(
        repo, alarm.id, TimerChanges(days=["sat", "sun"]), now=now, zone=NY
    )
    assert repeating.repeat_days == days_to_mask(["sat", "sun"])
    assert repeating.due_at == datetime(2027, 1, 9, 8, 0, tzinfo=NY).astimezone(timezone.utc)

    off = await service.update_timer(repo, alarm.id, TimerChanges(enabled=False), now=now, zone=NY)
    assert off.enabled is False
    assert off.due_at is None

    on = await service.update_timer(repo, alarm.id, TimerChanges(enabled=True), now=now, zone=NY)
    assert on.enabled is True
    assert on.due_at == next_alarm_at("08:00", days_to_mask(["sat", "sun"]), now, NY)

    renamed = await service.update_timer(repo, alarm.id, TimerChanges(label="gym"), now=now, zone=NY)
    assert renamed.label == "gym"
    assert renamed.due_at == on.due_at

    with pytest.raises(TimerError, match="server.timezone"):
        await service.update_timer(repo, alarm.id, TimerChanges(time="09:00"), now=now, zone=None)


async def test_delete_through_the_tool_and_unknown_id():
    repo = FakeTimerRepository()
    host = _host(repo)
    await host.call_tool("set_timer", {"duration_seconds": 60, "label": "pasta"})

    result = await host.call_tool("delete_timer_or_alarm", {"id": 1})
    assert not result.is_error
    assert result.content[0].text == "deleted the pasta timer"
    assert await repo.list_timers() == []

    missing = await host.call_tool("delete_timer_or_alarm", {"id": 1})
    assert missing.is_error


async def test_update_through_the_tool_builds_the_spoken_result_in_code():
    repo = FakeTimerRepository()
    host = _host(repo, zone=NY)
    await host.call_tool("set_timer", {"duration_seconds": 600, "label": "pasta"})

    result = await host.call_tool("update_timer_or_alarm", {"id": 1, "paused": True})

    assert not result.is_error
    assert result.content[0].text == "pasta timer paused with 10 minutes left"
    assert result.structured_content == {"id": 1, "kind": "timer"}
    empty = await host.call_tool("update_timer_or_alarm", {"id": 1})
    assert empty.is_error


def test_describe_timers_renders_exact_deterministic_text():
    from atlas.db.timer_repository import Timer

    def timer(**kwargs):
        base = dict(
            label="",
            due_at=None,
            remaining_s=None,
            duration_s=None,
            time_of_day=None,
            repeat_days=0,
            enabled=True,
            created_at=T0,
        )
        base.update(kwargs)
        return Timer(**base)

    timers = [
        timer(id=3, kind="timer", label="pasta", duration_s=600, due_at=T0 + timedelta(seconds=432)),
        timer(id=4, kind="timer", duration_s=300, remaining_s=120),
        timer(id=5, kind="alarm", label="work", time_of_day="07:00", repeat_days=WEEKDAY_MASK,
              due_at=T0 + timedelta(hours=9)),
        timer(id=6, kind="alarm", time_of_day="06:30", enabled=False),
    ]

    assert describe_timers(timers, T0) == "\n".join(
        [
            "Timers and alarms (use these ids with update_timer_or_alarm and delete_timer_or_alarm, "
            "never ask the user for an id):",
            '- id 3: 10 minute timer "pasta", 7 minutes 12 seconds left',
            "- id 4: 5 minute timer, paused with 2 minutes left",
            '- id 5: alarm "work" at 7:00 AM, mon tue wed thu fri, on',
            "- id 6: alarm at 6:30 AM, once, off",
        ]
    )
    assert describe_timers([], T0) == "Timers and alarms: none are set."


def test_state_message_carries_the_timers_block(monkeypatch):
    import atlas.app as app_module

    monkeypatch.setattr(app_module, "_current_moment", lambda: datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc))
    baseline = app_module._state_message({})
    assert "Timers and alarms" not in baseline

    monkeypatch.setattr(app_module, "_timers_view", lambda: ())
    assert app_module._state_message({}).endswith("Timers and alarms: none are set.")

    monkeypatch.setattr(app_module, "_timers_view", lambda: None)
    assert app_module._state_message({}).endswith(app_module._TIMERS_UNAVAILABLE_LINE)

    monkeypatch.setattr(app_module, "_timers_view", None)
    assert app_module._state_message({}) == baseline
