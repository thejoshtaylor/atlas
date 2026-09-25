"""Real assertions against `EdgeAudioSource.serve` (10-02-PLAN.md Task 2):
connection identity (reconnect/rival), size and rate bounds, and the
disconnect-mid-segment fallback -- driven through a scripted
`FakeEdgeSocket` (`tests/edge_fakes.py`), never a mock of the class under
test.
"""

from __future__ import annotations

import asyncio
import json

import pytest

from atlas.config import EdgeSourceConfig
from atlas.transports.edge import (
    CLOSE_BUSY,
    CLOSE_POLICY_VIOLATION,
    CLOSE_SUPERSEDED,
    MAX_AUDIO_FRAME_BYTES,
    MAX_INVALID_MESSAGES,
    MAX_QUEUED_FRAMES,
    MAX_TEXT_FRAME_BYTES,
    EdgeAudioSource,
)

from tests.edge_fakes import FakeEdgeSocket, fake_edge_device


def _measured_config(**overrides) -> EdgeSourceConfig:
    base = dict(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300)
    base.update(overrides)
    return EdgeSourceConfig(**base)


async def _wait_until(predicate, *, timeout: float = 2.0, interval: float = 0.01) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(interval)
    raise AssertionError("condition never became true within the timeout")


async def test_same_device_reconnect_supersedes():
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)

    socket_a = FakeEdgeSocket()
    task_a = asyncio.create_task(source.serve(socket_a, device))
    await _wait_until(lambda: socket_a.sent_text != [])

    socket_b = FakeEdgeSocket()
    task_b = asyncio.create_task(source.serve(socket_b, device))
    await _wait_until(lambda: socket_b.sent_text != [])

    await _wait_until(lambda: socket_a.close_calls != [])
    assert socket_a.close_calls == [(CLOSE_SUPERSEDED, None)]

    await _wait_until(task_a.done)
    assert task_a.cancelled()

    # Still streaming on the new connection: a binary frame reaches
    # frames(), proving `socket_b`'s connection is the live one.
    socket_b.push_bytes(b"\x00\x01\x00\x02")
    frames_iter = source.frames()
    chunk = await asyncio.wait_for(frames_iter.__anext__(), timeout=2.0)
    assert chunk == b"\x00\x01\x00\x02"

    socket_b.push_disconnect()
    await asyncio.wait_for(task_b, timeout=2.0)


async def test_reconnect_succeeds_when_the_previous_socket_is_already_dead():
    """10-REVIEW.md CR-02: a previous connection whose socket is already
    gone (Wi-Fi drop, the Pi's own process restarting) must not block the
    new connection from taking over -- before the fix, `close()` raising
    here left `self._websocket` naming the dead connection forever, and
    every subsequent reconnect attempt hit the same raising `close()` again."""
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)

    socket_a = FakeEdgeSocket(raise_on_close=RuntimeError("already gone"))
    task_a = asyncio.create_task(source.serve(socket_a, device))
    await _wait_until(lambda: socket_a.sent_text != [])

    socket_b = FakeEdgeSocket()
    task_b = asyncio.create_task(source.serve(socket_b, device))
    await _wait_until(lambda: socket_b.sent_text != [])

    # The new connection took over despite the old socket's close() raising.
    assert source.connected_device_id == device.id

    await _wait_until(lambda: socket_a.close_calls != [])
    assert socket_a.close_calls == [(CLOSE_SUPERSEDED, None)]

    await _wait_until(task_a.done)
    assert task_a.cancelled()

    socket_b.push_bytes(b"\x00\x01\x00\x02")
    frames_iter = source.frames()
    chunk = await asyncio.wait_for(frames_iter.__anext__(), timeout=2.0)
    assert chunk == b"\x00\x01\x00\x02"

    socket_b.push_disconnect()
    await asyncio.wait_for(task_b, timeout=2.0)


async def test_other_device_is_refused_while_one_is_connected():
    source = EdgeAudioSource(_measured_config())
    device_a = fake_edge_device(device_id=1)
    device_b = fake_edge_device(device_id=2)

    socket_a = FakeEdgeSocket()
    task_a = asyncio.create_task(source.serve(socket_a, device_a))
    await _wait_until(lambda: socket_a.sent_text != [])

    socket_b = FakeEdgeSocket()
    await asyncio.wait_for(source.serve(socket_b, device_b), timeout=2.0)

    assert socket_b.close_calls == [(CLOSE_BUSY, None)]
    assert socket_b.sent_text == []  # no hello -- refused before any protocol traffic.

    # The first connection is completely untouched.
    assert socket_a.close_calls == []
    assert not task_a.done()
    assert source.connected_device_id == device_a.id

    socket_a.push_disconnect()
    await asyncio.wait_for(task_a, timeout=2.0)


async def test_drop_mid_segment_publishes_vad_end():
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    published: list[dict] = []
    source.speech_signals.subscribe(published.append, replay_segment=False)

    socket.push_text(json.dumps({"type": "vad.start", "seq": 7}))
    await _wait_until(lambda: source.speech_signals.in_speech)

    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)

    assert not source.speech_signals.in_speech
    synthetic = [event for event in published if event.get("reason") == "disconnected"]
    assert synthetic == [{"type": "vad.end", "seq": 7, "reason": "disconnected"}]


async def test_oversize_binary_frame_is_dropped_and_counted():
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    socket.push_bytes(b"\x00" * (MAX_AUDIO_FRAME_BYTES + 2))
    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert source._frames_queue.qsize() == 0


async def test_misaligned_binary_frame_is_dropped_and_counted():
    source = EdgeAudioSource(_measured_config(channels=2))
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    # 3 bytes cannot be a whole number of 2-channel PCM16 frames (4 bytes each).
    socket.push_bytes(b"\x00\x01\x02")
    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert source._frames_queue.qsize() == 0


async def test_oversize_text_frame_is_dropped_and_counted():
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    oversize = json.dumps({"type": "vad.start", "seq": 1, "padding": "x" * MAX_TEXT_FRAME_BYTES})
    assert len(oversize.encode("utf-8")) > MAX_TEXT_FRAME_BYTES
    socket.push_text(oversize)
    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)
    assert not source.speech_signals.in_speech  # the oversize vad.start never landed.


async def test_twenty_malformed_messages_close_with_1008():
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    for _ in range(MAX_INVALID_MESSAGES):
        socket.push_text(json.dumps({"type": "not.a.real.type"}))

    await asyncio.wait_for(task, timeout=2.0)
    assert socket.close_calls == [(CLOSE_POLICY_VIOLATION, None)]


async def test_frame_queue_drops_oldest_when_full():
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    for i in range(MAX_QUEUED_FRAMES + 5):
        socket.push_bytes(bytes([i % 256, 0, 0, 0]))
    await _wait_until(lambda: source._frames_queue.qsize() == MAX_QUEUED_FRAMES)

    # The oldest 5 frames (i=0..4) were dropped -- the first item still in
    # the queue is frame index 5.
    first_remaining = source._frames_queue._queue[0]  # type: ignore[attr-defined]
    assert first_remaining == bytes([5, 0, 0, 0])

    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)


async def test_send_audio_with_no_connected_device_drops_the_chunk():
    source = EdgeAudioSource(_measured_config())
    # Never connected -- send_audio must not raise.
    await source.send_audio(b"reply-bytes")
    assert source.connected_device_id is None
