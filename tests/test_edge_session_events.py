"""D-17 and the ROADMAP's added-delay bullet (10-07-PLAN.md Task 2):

- A Pi's own `vad.start`/`vad.end`/`doa` events land in that turn's
  session, prefixed `edge.`, recorded only -- never published to the
  observer feed (`turn/controller.py::run_turn`'s `speech_signals`
  subscription).
- `EdgeAudioSource`'s ping/pong round trip yields a measured
  `added_delay_ms_p95` on every `latency` event, with no shared clock
  between the Pi and this server (`transports/edge.py`).

Driven through a real `EdgeAudioSource` and a scripted `FakeEdgeSocket`
(`tests/edge_fakes.py`), never a mock of either class under test.
"""

from __future__ import annotations

import asyncio
import json
import logging

from atlas.config import EdgeSourceConfig, SessionConfig
from atlas.providers.base import BrainReply, FinalTranscript
from atlas.session.observers import ObserverPublishingSource, ObserverRegistry
from atlas.session.recorder import SessionRecorder
from atlas.timing import TurnTimings
from atlas.transports.edge import ADDED_DELAY_BUDGET_MS, MAX_INVALID_MESSAGES, CLOSE_POLICY_VIOLATION, EdgeAudioSource
from atlas.turn.controller import run_turn

from tests.edge_fakes import FakeEdgeSocket, fake_edge_device


def _measured_config(**overrides) -> EdgeSourceConfig:
    base = dict(sample_rate=16000, channels=2, asr_channel=1, pre_roll_ms=200, tail_ms=300, ping_interval_s=0.02)
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


class _FinalizeGatedStt:
    """Yields its scripted final only once `finalize` is set -- the same
    double `tests/test_early_finalize.py` already uses, so this turn's
    `run_turn` call stays open exactly until the segment's own `vad.end`
    arrives, never finishing early on a race."""

    def __init__(self, text: str) -> None:
        self._text = text

    async def stream(self, frames, source_format=None, *, finalize=None):
        if finalize is not None:
            await finalize.wait()
        yield FinalTranscript(text=self._text)


# ---------------------------------------------------------------------------
# DoA and VAD land in the session, never the observer feed
# ---------------------------------------------------------------------------


async def test_doa_and_vad_are_recorded_not_published(tmp_path, fake_brain, fake_tts):
    """The segment begins (`vad.start`, then one `doa`) before this turn
    ever subscribes -- the wake hit lands after the first `doa`, the same
    order a real wake-word hit would see it. `replay_segment=True`
    (`SpeechSignals.subscribe`'s own default) is what makes the segment's
    own history still reach this turn's session."""
    config = _measured_config()
    source = EdgeAudioSource(config)
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    serve_task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    socket.push_text(json.dumps({"type": "vad.start", "seq": 1}))
    await _wait_until(lambda: source.speech_signals.in_speech)
    socket.push_text(json.dumps({"type": "doa", "azimuth_deg": [45.0]}))
    await _wait_until(lambda: len(source.speech_signals._segment_events) >= 2)

    registry = ObserverRegistry()
    observed_source = ObserverPublishingSource(source, "edge", registry)
    observer_queue = registry.subscribe()

    session_config = SessionConfig(dir=str(tmp_path))
    timings = TurnTimings()
    recorder = SessionRecorder(session_config, timings)

    stt = _FinalizeGatedStt("turn on the fan")
    brain = fake_brain(replies=[BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])

    async def _rest_of_segment() -> None:
        await asyncio.sleep(0.02)
        socket.push_text(json.dumps({"type": "doa", "azimuth_deg": [200.0]}))
        await asyncio.sleep(0.01)
        socket.push_text(json.dumps({"type": "vad.end", "seq": 2}))

    asyncio.ensure_future(_rest_of_segment())

    await run_turn(
        observed_source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )

    lines = (recorder.directory / "events.jsonl").read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    edge_events = [event for event in parsed if event["type"].startswith("edge.")]
    assert [event["type"] for event in edge_events] == [
        "edge.vad.start",
        "edge.doa",
        "edge.doa",
        "edge.vad.end",
    ]
    assert edge_events[1]["azimuth_deg"] == [45.0]
    assert edge_events[2]["azimuth_deg"] == [200.0]

    observed_types = []
    while not observer_queue.empty():
        observed_types.append(observer_queue.get_nowait().get("type"))
    assert not any(t and t.startswith("edge.") for t in observed_types)

    socket.push_disconnect()
    await asyncio.wait_for(serve_task, timeout=2.0)


async def test_an_edge_event_arriving_after_the_turn_ended_is_not_recorded(tmp_path, fake_brain, fake_tts):
    config = _measured_config()
    source = EdgeAudioSource(config)
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    serve_task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    socket.push_text(json.dumps({"type": "vad.start", "seq": 1}))
    await _wait_until(lambda: source.speech_signals.in_speech)

    session_config = SessionConfig(dir=str(tmp_path))
    timings = TurnTimings()
    recorder = SessionRecorder(session_config, timings)

    stt = _FinalizeGatedStt("turn on the fan")
    brain = fake_brain(replies=[BrainReply(text="the fan is on")])
    tts = fake_tts(chunks=[b"\x01\x02"])

    async def _end_segment_soon() -> None:
        await asyncio.sleep(0.01)
        socket.push_text(json.dumps({"type": "vad.end", "seq": 2}))

    asyncio.ensure_future(_end_segment_soon())

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )

    before = (recorder.directory / "events.jsonl").read_text(encoding="utf-8")

    # The turn has already unsubscribed and closed its recorder -- an edge
    # event published now must never reach it.
    source.speech_signals.publish({"type": "doa", "azimuth_deg": [10.0]})

    after = (recorder.directory / "events.jsonl").read_text(encoding="utf-8")
    assert after == before

    socket.push_disconnect()
    await asyncio.wait_for(serve_task, timeout=2.0)


# ---------------------------------------------------------------------------
# Ping/pong -> measured added delay
# ---------------------------------------------------------------------------


async def test_added_delay_is_capture_to_send_plus_half_rtt():
    fake_now_s = [0.0]

    def clock() -> float:
        return fake_now_s[0]

    source = EdgeAudioSource(_measured_config(), clock=clock)
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    serve_task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.latest_ping() is not None)

    ping = socket.latest_ping()
    fake_now_s[0] += 0.008  # 8ms round trip
    socket.push_pong(ping)
    await _wait_until(lambda: source.last_rtt_ms is not None)
    assert source.last_rtt_ms == 8.0

    captured: list[dict] = []
    source.speech_signals.subscribe(captured.append, replay_segment=False)
    socket.push_text(
        json.dumps(
            {
                "type": "latency",
                "capture_to_send_ms_p50": 15.0,
                "capture_to_send_ms_p95": 21.0,
                "capture_to_send_ms_max": 30.0,
                "frames": 5,
            }
        )
    )
    await _wait_until(lambda: captured != [])

    assert captured[0]["rtt_ms"] == 8.0
    assert captured[0]["added_delay_ms_p95"] == 25.0

    socket.push_disconnect()
    await asyncio.wait_for(serve_task, timeout=2.0)


async def test_a_latency_event_before_any_pong_records_null_rtt_and_added_delay():
    source = EdgeAudioSource(_measured_config())
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    serve_task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    captured: list[dict] = []
    source.speech_signals.subscribe(captured.append, replay_segment=False)
    socket.push_text(
        json.dumps(
            {
                "type": "latency",
                "capture_to_send_ms_p50": 5.0,
                "capture_to_send_ms_p95": 9.0,
                "capture_to_send_ms_max": 12.0,
                "frames": 3,
            }
        )
    )
    await _wait_until(lambda: captured != [])

    assert captured[0]["rtt_ms"] is None
    assert captured[0]["added_delay_ms_p95"] is None
    assert source.last_rtt_ms is None

    socket.push_disconnect()
    await asyncio.wait_for(serve_task, timeout=2.0)


async def test_an_unmatched_pong_never_changes_last_rtt_ms_and_counts_as_invalid():
    source = EdgeAudioSource(_measured_config(ping_interval_s=1000.0))
    device = fake_edge_device(device_id=1)
    socket = FakeEdgeSocket()
    serve_task = asyncio.create_task(source.serve(socket, device))
    await _wait_until(lambda: socket.sent_text != [])

    assert source.last_rtt_ms is None
    for _ in range(MAX_INVALID_MESSAGES):
        socket.push_text(json.dumps({"type": "pong", "id": 999, "server_t_ms": 0}))

    await asyncio.wait_for(serve_task, timeout=2.0)
    assert source.last_rtt_ms is None
    assert socket.close_calls == [(CLOSE_POLICY_VIOLATION, None)]


async def test_added_delay_over_budget_warns_once_and_disconnect_logs_worst_and_last(caplog):
    assert ADDED_DELAY_BUDGET_MS == 50
    fake_now_s = [0.0]

    def clock() -> float:
        return fake_now_s[0]

    source = EdgeAudioSource(_measured_config(), clock=clock)
    device = fake_edge_device(device_id=7)
    socket = FakeEdgeSocket()

    def _push_latency(p95: float) -> None:
        socket.push_text(
            json.dumps(
                {
                    "type": "latency",
                    "capture_to_send_ms_p50": p95,
                    "capture_to_send_ms_p95": p95,
                    "capture_to_send_ms_max": p95,
                    "frames": 1,
                }
            )
        )

    def _warning_lines() -> list[str]:
        return [record.getMessage() for record in caplog.records if "exceeds the" in record.getMessage()]

    with caplog.at_level(logging.INFO, logger="atlas.transports.edge"):
        serve_task = asyncio.create_task(source.serve(socket, device))
        await _wait_until(lambda: socket.latest_ping() is not None)

        ping = socket.latest_ping()
        fake_now_s[0] += 0.100  # 100ms RTT -> comfortably over the 50ms budget
        socket.push_pong(ping)
        await _wait_until(lambda: source.last_rtt_ms is not None)

        _push_latency(40.0)  # 40 + 50 = 90ms, over budget -- warns once
        await _wait_until(lambda: _warning_lines() != [])
        assert len(_warning_lines()) == 1
        assert str(device.id) in _warning_lines()[0]

        _push_latency(60.0)  # 60 + 50 = 110ms, still over budget -- no second warning
        await asyncio.sleep(0.05)
        assert len(_warning_lines()) == 1

        socket.push_disconnect()
        await asyncio.wait_for(serve_task, timeout=2.0)

    disconnect_lines = [record.getMessage() for record in caplog.records if "connection ended" in record.getMessage()]
    assert len(disconnect_lines) == 1
    assert "worst_added_delay_ms_p95=110.0" in disconnect_lines[0]
    assert "last_added_delay_ms_p95=110.0" in disconnect_lines[0]
