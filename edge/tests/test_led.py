"""`LedController` against a fake USB device (260928-m11). No real hardware
anywhere in this file."""

from __future__ import annotations

import asyncio
import logging
import struct
import threading

from atlas_edge import led as led_module
from atlas_edge.led import (
    EFFECT_OFF,
    EFFECT_RING,
    EFFECT_SINGLE_COLOR,
    FRAME_INTERVAL_S,
    LISTENING_COLOR,
    REPLYING_COLOR,
    THINKING_COLOR,
    THINKING_TAIL,
    RINGING_COLOR,
    LedController,
    pulse_frame,
    ring_frame,
)
from atlas_edge.xvf3800 import XvfNotFound

EFFECT = 12
COLOR = 16
RING = 19


class FakeDevice:
    """Records `(value, index, payload, thread ident)` for each write."""

    def __init__(self) -> None:
        self.transfers: list[tuple[int, int, bytes, int]] = []
        self.error: "Exception | None" = None

    def ctrl_transfer(self, bm, req, value, index, data, timeout):
        if self.error is not None:
            raise self.error
        self.transfers.append((value, index, bytes(data), threading.get_ident()))
        return len(data)

    def writes(self) -> list[tuple[int, bytes]]:
        return [(value, data) for value, _index, data, _tid in self.transfers]


def _controller(device, **kwargs) -> LedController:
    kwargs.setdefault("sleep", _instant_sleep)
    return LedController(lambda: device, **kwargs)


async def _instant_sleep(_seconds: float) -> None:
    await asyncio.sleep(0)


def _effect(effect: int) -> tuple[int, bytes]:
    return (EFFECT, bytes([effect]))


def _color(color: int) -> tuple[int, bytes]:
    return (COLOR, struct.pack("<I", color))


def _ring(head: int) -> tuple[int, bytes]:
    return (RING, struct.pack("<12I", *ring_frame(head)))


def test_the_three_state_colors_differ():
    assert len({LISTENING_COLOR, REPLYING_COLOR, THINKING_COLOR}) == 3


def test_the_frame_rate_is_between_10_and_20_hz():
    assert 0.05 <= FRAME_INTERVAL_S <= 0.1


def test_ring_frame_puts_the_full_color_at_the_head_and_fades_behind_it():
    frame = ring_frame(5)

    assert len(frame) == 12
    assert frame[5] == THINKING_COLOR
    for offset, level in enumerate(THINKING_TAIL):
        assert frame[(5 - offset) % 12] == led_module._scale(THINKING_COLOR, level)
    lit = {(5 - offset) % 12 for offset in range(len(THINKING_TAIL))}
    assert all(frame[i] == 0 for i in range(12) if i not in lit)


def test_ring_frame_wraps_the_tail_past_led_zero():
    frame = ring_frame(0)

    assert frame[0] == THINKING_COLOR
    assert frame[11] == led_module._scale(THINKING_COLOR, THINKING_TAIL[1])
    assert frame[10] == led_module._scale(THINKING_COLOR, THINKING_TAIL[2])


async def test_idle_writes_effect_off_once():
    device = FakeDevice()
    controller = _controller(device)

    controller.set_state("idle")
    await controller.wait()

    assert device.writes() == [_effect(EFFECT_OFF)]
    await controller.close()


async def test_listening_and_replying_write_a_color_then_the_single_color_effect():
    device = FakeDevice()
    controller = _controller(device)

    controller.set_state("listening")
    await controller.wait()
    controller.set_state("replying")
    await controller.wait()

    assert device.writes() == [
        _color(LISTENING_COLOR),
        _effect(EFFECT_SINGLE_COLOR),
        _color(REPLYING_COLOR),
        _effect(EFFECT_SINGLE_COLOR),
    ]
    await controller.close()


async def test_thinking_animates_the_ring_at_the_frame_interval():
    device = FakeDevice()
    sleeps: list[float] = []
    enough = asyncio.Event()

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        if len(sleeps) >= 3:
            enough.set()
        await asyncio.sleep(0)

    controller = _controller(device, sleep=fake_sleep)

    controller.set_state("thinking")
    await asyncio.wait_for(enough.wait(), timeout=2.0)
    controller.set_state("idle")
    await controller.wait()

    writes = device.writes()
    assert writes[:2] == [_ring(0), _effect(EFFECT_RING)]
    assert writes[2:5] == [_ring(1), _ring(2), _ring(3)]
    assert sleeps[:3] == [FRAME_INTERVAL_S] * 3
    await controller.close()


async def test_a_state_change_stops_the_animation():
    device = FakeDevice()
    frames = 0
    started = asyncio.Event()

    async def fake_sleep(_seconds: float) -> None:
        nonlocal frames
        frames += 1
        if frames >= 2:
            started.set()
        await asyncio.sleep(0)

    controller = _controller(device, sleep=fake_sleep)
    controller.set_state("thinking")
    await asyncio.wait_for(started.wait(), timeout=2.0)

    controller.set_state("idle")
    await controller.wait()

    writes = device.writes()
    assert writes[-1] == _effect(EFFECT_OFF)
    last_off = len(writes) - 1
    assert all(value != RING for value, _data in writes[last_off:])
    await asyncio.sleep(0.05)
    assert device.writes() == writes
    await controller.close()


async def test_the_same_state_twice_writes_once():
    device = FakeDevice()
    controller = _controller(device)

    controller.set_state("listening")
    controller.set_state("listening")
    await controller.wait()

    assert device.writes() == [_color(LISTENING_COLOR), _effect(EFFECT_SINGLE_COLOR)]
    await controller.close()


async def test_idle_after_replying_waits_for_the_playback_drain():
    device = FakeDevice()
    pending = iter([0.4, 0.0])
    sleeps: list[float] = []
    events: list[str] = []

    async def fake_sleep(seconds: float) -> None:
        sleeps.append(seconds)
        events.append(f"sleep:{seconds}")

    controller = _controller(device, pending_playback_s=lambda: next(pending), sleep=fake_sleep)
    controller.set_state("replying")
    await controller.wait()

    controller.set_state("idle")
    await controller.wait()

    assert sleeps == [0.4]
    assert device.writes()[-1] == _effect(EFFECT_OFF)
    await controller.close()


async def test_a_new_state_during_the_drain_wait_cancels_the_idle():
    device = FakeDevice()
    waiting = asyncio.Event()
    release = asyncio.Event()
    pending = [0.4]

    async def fake_sleep(_seconds: float) -> None:
        waiting.set()
        pending[0] = 0.0  # the backlog drains while the idle waits
        await release.wait()

    controller = _controller(device, pending_playback_s=lambda: pending[0], sleep=fake_sleep)
    controller.set_state("replying")
    await controller.wait()

    controller.set_state("idle")
    await asyncio.wait_for(waiting.wait(), timeout=2.0)
    controller.set_state("listening")
    await controller.wait()

    writes = device.writes()
    assert writes[-2:] == [_color(LISTENING_COLOR), _effect(EFFECT_SINGLE_COLOR)]
    assert _effect(EFFECT_OFF) not in writes
    await controller.close()


async def test_a_failing_device_never_raises_and_logs_one_warning_per_streak(caplog):
    device = FakeDevice()
    device.error = OSError("usb gone")
    controller = _controller(device)

    with caplog.at_level(logging.DEBUG, logger="atlas_edge.led"):
        for state in ("listening", "thinking"):
            controller.set_state(state)
            await asyncio.sleep(0.05)
        controller.set_state("idle")
        await controller.wait()
        await controller.close()

    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1


async def test_writes_resume_and_the_device_is_looked_up_again_after_a_failure(caplog):
    device = FakeDevice()
    lookups = 0

    def find_device():
        nonlocal lookups
        lookups += 1
        if lookups == 1:
            raise XvfNotFound("none attached")
        return device

    controller = LedController(find_device, sleep=_instant_sleep)

    controller.set_state("listening")
    await controller.wait()
    assert device.writes() == []
    assert lookups == 1

    controller.set_state("replying")
    await controller.wait()

    assert lookups == 2
    assert device.writes() == [_color(REPLYING_COLOR), _effect(EFFECT_SINGLE_COLOR)]
    await controller.close()


async def test_every_write_runs_off_the_event_loop_thread():
    device = FakeDevice()
    controller = _controller(device)

    controller.set_state("listening")
    await controller.wait()
    controller.set_state("idle")
    await controller.wait()

    assert device.transfers
    assert all(tid != threading.get_ident() for *_rest, tid in device.transfers)
    assert len({tid for *_rest, tid in device.transfers}) == 1
    await controller.close()


async def test_an_unknown_state_is_ignored_with_a_warning(caplog):
    device = FakeDevice()
    controller = _controller(device)

    with caplog.at_level(logging.WARNING, logger="atlas_edge.led"):
        controller.set_state("bogus")
    await controller.wait()

    assert device.writes() == []
    assert "bogus" in caplog.text
    await controller.close()


async def test_close_stops_the_animation_and_turns_the_ring_off():
    device = FakeDevice()
    controller = _controller(device)
    controller.set_state("thinking")
    await asyncio.sleep(0.02)

    await controller.close()

    writes = device.writes()
    assert writes[-1] == _effect(EFFECT_OFF)
    await asyncio.sleep(0.02)
    assert device.writes() == writes


def test_pulse_frame_lights_every_led_dim_then_bright():
    dim = pulse_frame(0.0)
    bright = pulse_frame(0.5)
    assert len(set(dim)) == 1 and len(set(bright)) == 1
    assert bright[0] == RINGING_COLOR
    assert 0 < (dim[0] >> 16) < (bright[0] >> 16)


def test_ringing_pulses_the_ring_until_the_next_state():
    async def scenario():
        device = FakeDevice()
        controller = _controller(device)
        controller.set_state("ringing")
        for _ in range(20):
            await asyncio.sleep(0)
        controller.set_state("idle")
        await controller.wait()
        return device.writes()

    writes = asyncio.run(scenario())
    frames = [data for value, data in writes if value == RING]
    assert _effect(EFFECT_RING) in writes
    assert len({frame for frame in frames}) > 1
    assert writes[-1] == _effect(EFFECT_OFF)
