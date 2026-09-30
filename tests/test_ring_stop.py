"""Quick task 260930-e3r: stop a ringing timer or alarm with one bare word.

The matcher is a table. The scheduler ring is proven with fake clocks. The
tracer runs a real scheduler, a real serial `SourceRunner` and a real
`RingStopWindow` over a fake STT. No house data appears here.
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
from datetime import datetime, timedelta, timezone

import pytest

from atlas.providers.tts_xai import SinkFormat
from atlas.sources.runner import SourceRunner
from atlas.timers.core import announcement
from atlas.timers.ring_stop import RingStopWindow, is_stop_command, make_stt_transcribe
from atlas.timers.scheduler import TimerScheduler
from atlas.transports.base import SourceFormat
from atlas.turn.controller import SpeechResult
from tests.conftest import FakeStt, FakeWakeDetector, FinalTranscript
from tests.timer_fakes import FakeTimerRepository

T0 = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)
BOUND_S = 5.0


# --- the matcher ---------------------------------------------------------------


@pytest.mark.parametrize(
    "text",
    [
        "stop",
        "Stop.",
        "OKAY!",
        "ok",
        "thanks",
        "thank you",
        "shut up",
        "enough",
        "That's enough.",
        "cancel",
        "off",
        "stop it",
        "stop the timer",
        "Stop the alarm.",
        "turn it off",
        "stop, stop",
        "okay stop",
        "stop please",
        "hey atlas stop",
        "Atlas, stop the timer.",
    ],
)
def test_stop_commands_match(text):
    assert is_stop_command(text)


def test_stop_command_with_the_ring_text_echoed_before_it():
    assert is_stop_command("your pasta timer is done stop", ring_text="Your pasta timer is done.")


def _announcements():
    from atlas.db.timer_repository import Timer

    def make(kind, label, time_of_day=None):
        return Timer(
            id=1,
            kind=kind,
            label=label,
            due_at=T0,
            remaining_s=None,
            duration_s=60 if kind == "timer" else None,
            time_of_day=time_of_day,
            repeat_days=0,
            enabled=True,
            created_at=T0,
        )

    return [
        announcement(make("timer", "")),
        announcement(make("timer", "pasta")),
        announcement(make("alarm", "", "07:00")),
        announcement(make("alarm", "gym", "07:00")),
    ]


@pytest.mark.parametrize(
    "text",
    [
        "",
        "stop the music",
        "turn off the lights",
        "don't stop",
        "cancel my meeting",
        "okay google",
        "thanks for dinner",
        "what time is it",
        "stop stop stop stop stop stop stop",
    ],
)
def test_other_speech_does_not_match(text):
    assert not is_stop_command(text)


def test_the_ring_echo_alone_does_not_match():
    assert not is_stop_command("your pasta timer is done", ring_text="Your pasta timer is done.")
    for form in _announcements():
        assert not is_stop_command(form), form
        assert not is_stop_command(form, ring_text=form), form


# --- the scheduler ring --------------------------------------------------------


async def _due_timer(repo):
    await repo.create_timer(
        kind="timer",
        label="pasta",
        due_at=T0,
        remaining_s=None,
        duration_s=60,
        time_of_day=None,
        repeat_days=0,
        enabled=True,
        created_at=T0,
    )


def _scheduler(repo, spoken, *, speak_s=0.0, **kwargs):
    async def speak(text: str) -> None:
        spoken.append(text)
        await asyncio.sleep(speak_s)

    kwargs.setdefault("ring_gap_s", 0.005)
    return TimerScheduler(repo, speak, zone=None, clock=lambda: T0, **kwargs)


async def _until(predicate, bound_s=BOUND_S):
    async with asyncio.timeout(bound_s):
        while not predicate():
            await asyncio.sleep(0.005)


async def test_a_due_timer_rings_until_stop_ringing():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    scheduler = _scheduler(repo, spoken, speak_s=0.005)

    poll = asyncio.create_task(scheduler._poll_once())
    await _until(lambda: len(spoken) >= 3)
    assert scheduler.ringing
    assert scheduler.ring_text == "Your pasta timer is done."
    assert scheduler.stop_ringing() is True
    await asyncio.wait_for(poll, BOUND_S)

    assert not scheduler.ringing
    assert scheduler.ring_text is None
    assert await repo.list_timers() == []
    assert scheduler.fired_count == 1
    count = len(spoken)
    await asyncio.sleep(0.05)
    assert len(spoken) == count


async def test_the_ring_goes_quiet_at_max_ring_s(caplog):
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    now = [0.0]

    async def speak(text: str) -> None:
        spoken.append(text)
        now[0] += 50.0

    scheduler = TimerScheduler(
        repo, speak, zone=None, clock=lambda: T0, ring_gap_s=0.001, max_ring_s=120, monotonic=lambda: now[0]
    )
    with caplog.at_level(logging.INFO, logger="atlas.timers.scheduler"):
        await asyncio.wait_for(scheduler._poll_once(), BOUND_S)

    assert len(spoken) == 3
    assert not scheduler.ringing
    assert await repo.list_timers() == []
    assert [r for r in caplog.records if "went quiet" in r.getMessage()]


async def test_idle_scheduler_controls_are_no_ops():
    scheduler = _scheduler(FakeTimerRepository(), [])

    assert scheduler.stop_ringing() is False
    await asyncio.wait_for(scheduler.wait_ring_over(), 1)


async def test_stop_during_a_ring_leaves_no_ring_task():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    scheduler = _scheduler(repo, spoken, speak_s=0.005)
    scheduler.start()
    await _until(lambda: scheduler.ringing)

    await asyncio.wait_for(scheduler.stop(), BOUND_S)

    assert not scheduler.ringing
    assert scheduler._ring_task is None
    count = len(spoken)
    await asyncio.sleep(0.05)
    assert len(spoken) == count


async def test_a_speak_that_raises_ends_that_ring_and_the_next_poll_works(caplog):
    repo = FakeTimerRepository()
    await _due_timer(repo)
    await _due_timer(repo)
    calls: list[str] = []

    async def speak(text: str) -> None:
        calls.append(text)
        if len(calls) == 1:
            raise RuntimeError("speaker is gone")
        await asyncio.sleep(0)
        scheduler.stop_ringing()

    scheduler = TimerScheduler(repo, speak, zone=None, clock=lambda: T0, ring_gap_s=0.001)
    with caplog.at_level(logging.ERROR, logger="atlas.timers.scheduler"):
        await asyncio.wait_for(scheduler._poll_once(), BOUND_S)

    assert len(calls) == 2
    assert [r for r in caplog.records if "failed" in r.getMessage()]
    assert await repo.list_timers() == []


async def test_a_stale_timer_is_cleared_without_a_ring():
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    scheduler = TimerScheduler(
        repo, lambda text: spoken.append(text), zone=None, clock=lambda: T0 + timedelta(seconds=60 + 700)
    )

    await scheduler._poll_once()

    assert spoken == []
    assert await repo.list_timers() == []


# --- the tracer: a real scheduler, a real serial runner, a real window ---------


class _TickSource:
    """Yields one frame every 10 ms for as long as it is read."""

    async def frames(self):
        while True:
            await asyncio.sleep(0.01)
            yield b"\x00" * 160

    def source_format(self) -> SourceFormat:
        return SourceFormat("alaw", 8000)


async def _ring_with_window(stt_text: str, *, max_ring_s: float):
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    scheduler = _scheduler(repo, spoken, speak_s=0.01, max_ring_s=max_ring_s)
    turns: list[object] = []

    async def run_turn_fn(source):
        turns.append(source)

    stt = FakeStt([FinalTranscript(stt_text)])
    window = RingStopWindow(lambda: scheduler, make_stt_transcribe(lambda: stt, max_utterance_s=2.0))
    runner = SourceRunner(
        "camera",
        _TickSource(),
        FakeWakeDetector(fire_at_call=-1),
        lambda chunk: chunk,
        run_turn_fn,
        ring_window=window,
    )
    return repo, scheduler, runner, turns, spoken


async def test_a_bare_stop_silences_the_ring_and_reaches_no_turn():
    repo, scheduler, runner, turns, _spoken = await _ring_with_window("stop", max_ring_s=5)
    runner_task = asyncio.create_task(runner.run())
    try:
        started = asyncio.get_running_loop().time()
        await asyncio.wait_for(scheduler._poll_once(), BOUND_S)
        assert asyncio.get_running_loop().time() - started < 2
    finally:
        runner_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner_task

    assert turns == []
    assert await repo.list_timers() == []


async def test_other_speech_in_the_window_reaches_no_turn_and_the_ring_goes_on(caplog):
    repo, scheduler, runner, turns, spoken = await _ring_with_window("turn off the lights", max_ring_s=0.3)
    runner_task = asyncio.create_task(runner.run())
    try:
        with caplog.at_level(logging.WARNING, logger="atlas.timers.ring_stop"):
            await asyncio.wait_for(scheduler._poll_once(), BOUND_S)
    finally:
        runner_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await runner_task

    assert turns == []
    assert len(spoken) >= 3
    assert len([r for r in caplog.records if "gave up" in r.getMessage()]) == 1
    assert not [r for r in caplog.records if "lights" in r.getMessage()]


async def test_the_window_gives_up_with_one_warning_when_stt_is_unavailable(caplog):
    repo = FakeTimerRepository()
    await _due_timer(repo)
    spoken: list[str] = []
    scheduler = _scheduler(repo, spoken, speak_s=0.01, max_ring_s=0.2)
    window = RingStopWindow(lambda: scheduler, make_stt_transcribe(lambda: None, max_utterance_s=2.0))
    poll = asyncio.create_task(scheduler._poll_once())
    await _until(lambda: scheduler.ringing)

    with caplog.at_level(logging.WARNING, logger="atlas.timers.ring_stop"):
        assert window.active()
        await asyncio.wait_for(window.run(_TickSource()), BOUND_S)
        assert not window.active()
    await asyncio.wait_for(poll, BOUND_S)

    assert len([r for r in caplog.records if "gave up" in r.getMessage()]) == 1
    assert len(spoken) >= 2
    assert not window.active()


# --- pacing one ring repetition to real playback -------------------------------


async def test_wait_for_playback_end_sleeps_to_the_estimated_end():
    from atlas.app import _wait_for_playback_end

    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)

    sink = SinkFormat("alaw", 8000)
    await _wait_for_playback_end(
        SpeechResult(8000, 0.0, 0.0), sink, clock=lambda: 0.0, sleep=fake_sleep
    )
    await _wait_for_playback_end(None, sink, clock=lambda: 0.0, sleep=fake_sleep)
    await _wait_for_playback_end(
        SpeechResult(8000 * 600, 0.0, 0.0), sink, clock=lambda: 0.0, sleep=fake_sleep
    )
    await _wait_for_playback_end(
        SpeechResult(8000, 0.0, 0.0), sink, clock=lambda: 5.0, sleep=fake_sleep
    )

    assert slept == [1.0, 30.0]
