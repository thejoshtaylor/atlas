"""Phase 15 (D-14, D-15): the scheduler tells a listener which timer rings and
when the ring ends, and stops only the ring a caller names. Fake clocks and
fake labels only."""

from __future__ import annotations

import asyncio
import logging
from datetime import datetime, timezone

from atlas.db.timer_repository import Timer
from atlas.timers.scheduler import TimerScheduler
from tests.timer_fakes import FakeTimerRepository

T0 = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)
BOUND_S = 5.0


async def _due_timer(repo, label="pasta"):
    await repo.create_timer(
        kind="timer",
        label=label,
        due_at=T0,
        remaining_s=None,
        duration_s=60,
        time_of_day=None,
        repeat_days=0,
        enabled=True,
        created_at=T0,
    )


def _scheduler(repo, spoken, *, speak_s=0.005, **kwargs):
    async def speak(text: str) -> None:
        spoken.append(text)
        await asyncio.sleep(speak_s)

    kwargs.setdefault("ring_gap_s", 0.005)
    return TimerScheduler(repo, speak, zone=None, clock=lambda: T0, **kwargs)


async def _until(predicate, bound_s=BOUND_S):
    async with asyncio.timeout(bound_s):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_on_ring_event_gets_the_timer_at_start_and_at_the_stop():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    events: list[tuple[bool, Timer]] = []

    async def on_ring_event(ringing: bool, timer: Timer) -> None:
        events.append((ringing, timer))

    scheduler = _scheduler(repo, spoken, on_ring_event=on_ring_event)
    poll = asyncio.create_task(scheduler._poll_once())
    await _until(lambda: len(spoken) >= 1)
    assert [ringing for ringing, _ in events] == [True]
    assert events[0][1].id == 1
    assert events[0][1].kind == "timer"
    assert scheduler.stop_ringing() is True
    await asyncio.wait_for(poll, BOUND_S)

    assert [ringing for ringing, _ in events] == [True, False]
    assert events[1][1] is events[0][1]


async def test_the_end_call_also_runs_at_max_ring_s():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    now = [0.0]
    events: list[bool] = []

    async def speak(text: str) -> None:
        now[0] += 50.0

    async def on_ring_event(ringing: bool, timer: Timer) -> None:
        events.append(ringing)

    scheduler = TimerScheduler(
        repo,
        speak,
        zone=None,
        clock=lambda: T0,
        ring_gap_s=0.001,
        max_ring_s=120,
        monotonic=lambda: now[0],
        on_ring_event=on_ring_event,
    )
    await asyncio.wait_for(scheduler._poll_once(), BOUND_S)

    assert events == [True, False]
    assert not scheduler.ringing


async def test_the_end_call_also_runs_when_the_poll_task_is_cancelled():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    events: list[bool] = []

    async def on_ring_event(ringing: bool, timer: Timer) -> None:
        events.append(ringing)

    scheduler = _scheduler(repo, spoken, on_ring_event=on_ring_event)
    scheduler.start()
    await _until(lambda: scheduler.ringing)
    await asyncio.wait_for(scheduler.stop(), BOUND_S)

    assert events == [True, False]
    assert not scheduler.ringing


async def test_ring_timer_id_names_the_ringing_timer_only_during_the_ring():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    scheduler = _scheduler(repo, spoken)

    assert scheduler.ring_timer_id is None
    poll = asyncio.create_task(scheduler._poll_once())
    await _until(lambda: len(spoken) >= 1)
    assert scheduler.ring_timer_id == 1
    scheduler.stop_ringing()
    await asyncio.wait_for(poll, BOUND_S)

    assert scheduler.ring_timer_id is None


async def test_stop_ring_for_stops_only_the_ring_it_names():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    scheduler = _scheduler(repo, spoken)

    assert scheduler.stop_ring_for(1) is False  # no ring: nothing to stop

    poll = asyncio.create_task(scheduler._poll_once())
    await _until(lambda: len(spoken) >= 1)
    assert scheduler.stop_ring_for(99) is False
    count = len(spoken)
    await _until(lambda: len(spoken) > count)
    assert scheduler.ringing  # the stale id changed nothing

    assert scheduler.stop_ring_for(1) is True
    await asyncio.wait_for(poll, BOUND_S)
    assert not scheduler.ringing
    assert scheduler.stop_ring_for(1) is False  # the ring is over


async def test_a_raising_on_ring_event_never_stops_or_breaks_the_ring(caplog):
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    calls: list[bool] = []

    async def on_ring_event(ringing: bool, timer: Timer) -> None:
        calls.append(ringing)
        raise RuntimeError("a failing panel never stops a ring")

    scheduler = _scheduler(repo, spoken, on_ring_event=on_ring_event)
    with caplog.at_level(logging.WARNING, logger="atlas.timers.scheduler"):
        poll = asyncio.create_task(scheduler._poll_once())
        await _until(lambda: len(spoken) >= 3)
        assert scheduler.ringing
        scheduler.stop_ringing()
        await asyncio.wait_for(poll, BOUND_S)  # does not raise

    assert calls == [True, False]
    assert [r for r in caplog.records if "ring event" in r.getMessage()]


async def test_on_ring_still_gets_true_then_false_beside_on_ring_event():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    led: list[bool] = []
    panel: list[bool] = []

    async def on_ring(ringing: bool) -> None:
        led.append(ringing)

    async def on_ring_event(ringing: bool, timer: Timer) -> None:
        panel.append(ringing)

    scheduler = _scheduler(repo, spoken, on_ring=on_ring, on_ring_event=on_ring_event)
    poll = asyncio.create_task(scheduler._poll_once())
    await _until(lambda: len(spoken) >= 1)
    scheduler.stop_ringing()
    await asyncio.wait_for(poll, BOUND_S)

    assert led == [True, False]
    assert panel == [True, False]
