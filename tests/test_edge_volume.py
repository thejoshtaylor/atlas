"""Voice volume control, server side and end to end (260929-mii): the
`build_volume` clamp, the `volume.result` parser, `EdgeAudioSource.set_volume`,
and the `set_speaker_volume` tool. The end-to-end tests stand in for the Pi
with the real Pi code (`atlas_edge.protocol` and `atlas_edge.volume`) and a
fake `amixer` runner.
"""

from __future__ import annotations

import asyncio
import json
from typing import Any

import pytest

from atlas.config import ConfigError, EdgeSourceConfig, EdgeVolumeConfig
from atlas.speaker.volume_tool import SET_SPEAKER_VOLUME_TOOL_NAME, VolumeToolHost, set_current_turn_target
from atlas.transports.edge import (
    EdgeAudioSource,
    EdgeProtocolError,
    EdgeVolumeError,
    build_volume,
    parse_edge_event,
)
from atlas_edge import protocol as pi_protocol
from atlas_edge.volume import VolumeControl

from tests.edge_fakes import FakeEdgeSocket, fake_edge_device


def _config(**kwargs: Any) -> EdgeSourceConfig:
    return EdgeSourceConfig(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300, **kwargs)


async def _wait_until(predicate, *, timeout: float = 2.0) -> None:
    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    while loop.time() < deadline:
        if predicate():
            return
        await asyncio.sleep(0.01)
    raise AssertionError("condition never became true within the timeout")


async def _serve(source: EdgeAudioSource, socket: FakeEdgeSocket) -> "asyncio.Task[None]":
    task = asyncio.create_task(source.serve(socket, fake_edge_device(device_id=1)))
    await _wait_until(lambda: socket.sent_text != [])
    return task


async def _stop(socket: FakeEdgeSocket, task: "asyncio.Task[None]") -> None:
    socket.push_disconnect()
    await asyncio.wait_for(task, timeout=2.0)


def _volume_messages(socket: FakeEdgeSocket) -> "list[dict[str, Any]]":
    messages = [json.loads(text) for text in socket.sent_text]
    return [m for m in messages if m.get("type") == "volume"]


class _FakeAmixer:
    """Records each argv and returns canned `amixer` output in order."""

    def __init__(self, *outputs: str) -> None:
        self.calls: "list[list[str]]" = []
        self._outputs = list(outputs)

    async def __call__(self, args) -> str:
        self.calls.append(list(args))
        return self._outputs.pop(0)


async def _pi_answers(socket: FakeEdgeSocket, control: VolumeControl, *, error: "str | None" = None) -> None:
    """Act as the Pi: parse the sent `volume` message with the real Pi
    parser, run the real `VolumeControl`, and push the real reply text."""
    await _wait_until(lambda: _volume_messages(socket) != [])
    request = pi_protocol.parse_server_message(json.dumps(_volume_messages(socket)[-1]))
    if error is not None:
        socket.push_text(pi_protocol.volume_result(request.id, error=error))
        return
    level = await control.apply(request)
    socket.push_text(pi_protocol.volume_result(request.id, level=level))


# --- build_volume -----------------------------------------------------------


def test_build_volume_clamps_an_absolute_level_into_the_limits() -> None:
    limits = EdgeVolumeConfig()
    low = json.loads(build_volume(1, level=0, volume=limits))
    assert (low["level"], low["min_percent"], low["max_percent"]) == (30, 30, 100)
    capped = json.loads(build_volume(1, level=100, volume=EdgeVolumeConfig(max_percent=80)))
    assert capped["level"] == 80
    assert json.loads(build_volume(1, level=55, volume=limits))["level"] == 55


def test_build_volume_relative_carries_step_and_limits_and_no_level() -> None:
    message = json.loads(build_volume(2, direction="up", volume=EdgeVolumeConfig()))
    assert message == {
        "type": "volume",
        "id": 2,
        "direction": "up",
        "step_percent": 10,
        "min_percent": 30,
        "max_percent": 100,
    }
    assert "level" not in message


@pytest.mark.parametrize(
    "kwargs",
    [{"level": 5, "direction": "up"}, {}, {"direction": "sideways"}],
)
def test_build_volume_rejects_bad_arguments(kwargs: "dict[str, Any]") -> None:
    with pytest.raises(ValueError):
        build_volume(1, volume=EdgeVolumeConfig(), **kwargs)


def test_edge_volume_config_validates() -> None:
    assert EdgeSourceConfig.from_config({"volume": {"min_percent": 20}}).volume.min_percent == 20
    assert EdgeSourceConfig.from_config(None).volume == EdgeVolumeConfig()
    for bad in ({"min_percent": 90, "max_percent": 50}, {"step_percent": 0}, {"max_percent": True}, {"loud": 1}):
        with pytest.raises(ConfigError):
            EdgeVolumeConfig.from_config(bad)
    with pytest.raises(ConfigError):
        EdgeVolumeConfig.from_config([1])


# --- parse_edge_event -------------------------------------------------------


def test_parse_volume_result_accepts_level_and_error() -> None:
    assert parse_edge_event('{"type":"volume.result","id":3,"level":60}') == {
        "type": "volume.result",
        "id": 3,
        "level": 60,
    }
    assert parse_edge_event('{"type":"volume.result","id":3,"error":"x"}') == {
        "type": "volume.result",
        "id": 3,
        "error": "x",
    }
    cut = parse_edge_event(json.dumps({"type": "volume.result", "id": 3, "error": "e" * 300}))
    assert len(cut["error"]) == 200


@pytest.mark.parametrize(
    "payload",
    [
        {"id": 3, "level": 101},
        {"id": 3, "level": True},
        {"id": "3", "level": 5},
        {"id": 3, "level": 5, "error": "x"},
        {"id": 3},
        {"id": -1, "level": 5},
    ],
)
def test_parse_volume_result_rejects_bad_shapes(payload: "dict[str, Any]") -> None:
    with pytest.raises(EdgeProtocolError):
        parse_edge_event(json.dumps({"type": "volume.result", **payload}))


# --- end to end -------------------------------------------------------------


async def test_absolute_level_round_trips_and_is_clamped_to_the_server_minimum() -> None:
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    amixer = _FakeAmixer("Mono: Playback 18 [30%] [-12.00dB]")
    set_current_turn_target(source)
    host = VolumeToolHost()
    call = asyncio.create_task(host.call_tool(SET_SPEAKER_VOLUME_TOOL_NAME, {"level": 5}))
    await _pi_answers(socket, VolumeControl("Array", "PCM,0", run=amixer))
    result = await asyncio.wait_for(call, timeout=2.0)
    assert amixer.calls == [["-c", "Array", "sset", "PCM,0", "30%"]]
    assert not result.is_error
    assert result.content[0].text == "speaker volume is now 30 percent"
    assert result.structured_content == {"level": 30}
    await _stop(socket, task)


async def test_step_up_reads_the_level_first_and_reports_the_new_one() -> None:
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    amixer = _FakeAmixer("Mono: Playback 30 [50%]", "Mono: Playback 36 [60%]")
    set_current_turn_target(source)
    call = asyncio.create_task(VolumeToolHost().call_tool(SET_SPEAKER_VOLUME_TOOL_NAME, {"direction": "up"}))
    await _pi_answers(socket, VolumeControl("Array", "PCM,0", run=amixer))
    result = await asyncio.wait_for(call, timeout=2.0)
    assert amixer.calls == [["-c", "Array", "sget", "PCM,0"], ["-c", "Array", "sset", "PCM,0", "60%"]]
    assert result.structured_content == {"level": 60}
    await _stop(socket, task)


async def test_a_pi_error_becomes_an_error_result() -> None:
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    set_current_turn_target(source)
    call = asyncio.create_task(VolumeToolHost().call_tool(SET_SPEAKER_VOLUME_TOOL_NAME, {"level": 50}))
    await _pi_answers(socket, VolumeControl("Array", "PCM,0"), error="amixer is not installed on this device")
    result = await asyncio.wait_for(call, timeout=2.0)
    assert result.is_error
    assert result.content[0].text == "amixer is not installed on this device"
    await _stop(socket, task)


async def test_no_target_or_bad_arguments_send_nothing() -> None:
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    host = VolumeToolHost()
    set_current_turn_target(None)
    assert (await host.call_tool(SET_SPEAKER_VOLUME_TOOL_NAME, {"level": 50})).is_error
    set_current_turn_target(source)
    assert (await host.call_tool(SET_SPEAKER_VOLUME_TOOL_NAME, {"level": 50, "direction": "up"})).is_error
    assert (await host.call_tool(SET_SPEAKER_VOLUME_TOOL_NAME, {"level": 150})).is_error
    assert (await host.call_tool(SET_SPEAKER_VOLUME_TOOL_NAME, {})).is_error
    assert (await host.call_tool("no_such_tool", {})).is_error
    assert _volume_messages(socket) == []
    set_current_turn_target(None)
    await _stop(socket, task)


# --- set_volume failure modes -----------------------------------------------


async def test_set_volume_with_no_device_raises() -> None:
    with pytest.raises(EdgeVolumeError, match="no edge device"):
        await EdgeAudioSource(_config()).set_volume(level=50)


async def test_set_volume_times_out_and_leaves_nothing_pending() -> None:
    source = EdgeAudioSource(_config(), volume_reply_timeout_s=0.05)
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    with pytest.raises(EdgeVolumeError, match="did not answer"):
        await source.set_volume(level=50)
    assert source._pending_volume == {}
    await _stop(socket, task)


async def test_a_disconnect_fails_a_waiting_request() -> None:
    source = EdgeAudioSource(_config())
    socket = FakeEdgeSocket()
    task = await _serve(source, socket)
    waiting = asyncio.create_task(source.set_volume(level=50))
    await _wait_until(lambda: _volume_messages(socket) != [])
    await _stop(socket, task)
    with pytest.raises(EdgeVolumeError, match="disconnected"):
        await asyncio.wait_for(waiting, timeout=2.0)


async def test_an_unmatched_volume_result_counts_as_invalid() -> None:
    source = EdgeAudioSource(_config())
    assert source._handle_event({"type": "volume.result", "id": 99, "level": 50}) is False
