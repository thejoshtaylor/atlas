"""Regression tests for quick task 260924-4is: the event loop stalls that
got the pod liveness-killed in production (D1, D2, D5).

Every fake source here drains a real `asyncio.Queue` the same shape
`CameraAudioSource.frames()` uses (module docstring, `transports/
camera.py`): `await queue.get()` on a non-empty queue never yields, so a
backlog of chunks with no real await between them drains in one
uninterrupted loop step -- the mechanism the prod evidence pointed at.
Every heartbeat measurement below proves that step no longer blocks the
rest of the process once `sources/runner.py` moves detector work onto its
own worker thread (D2).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import threading
import time
from datetime import UTC

from atlas.loop_stall import LoopStallReporter
from atlas.sources.runner import BargeInMonitor, SourceRunner
from atlas.wake.base import WakeHit

_CHUNK = bytes(320)  # 160 samples of 16-bit silence -- shape only, never read as real audio


class _QueueSource:
    """The same queue-backed `frames()` shape `CameraAudioSource` uses:
    `await queue.get()` on a non-empty queue never yields."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._queue: asyncio.Queue = asyncio.Queue()
        for chunk in chunks:
            self._queue.put_nowait(chunk)
        self._queue.put_nowait(None)

    async def frames(self):
        while True:
            chunk = await self._queue.get()
            if chunk is None:
                return
            yield chunk


async def _unused_run_turn_fn(turn_source) -> None:
    raise AssertionError("run_turn_fn should not be called in this test")


class _SleepingDetector:
    """`process()` blocks the calling thread for `sleep_s`, then reports
    no hit. Every call's thread id is appended to `idents` under `lock`."""

    def __init__(self, sleep_s: float = 0.0, idents: list[int] | None = None, lock: threading.Lock | None = None) -> None:
        self._sleep_s = sleep_s
        self._idents = idents
        self._lock = lock

    def process(self, chunk: bytes) -> WakeHit | None:
        if self._sleep_s:
            time.sleep(self._sleep_s)
        if self._idents is not None and self._lock is not None:
            with self._lock:
                self._idents.append(threading.get_ident())
        return None

    def close(self) -> None:
        pass


def _recording_decode(idents: list[int], lock: threading.Lock, sleep_s: float = 0.0):
    def decode(chunk: bytes) -> bytes:
        if sleep_s:
            time.sleep(sleep_s)
        with lock:
            idents.append(threading.get_ident())
        return chunk

    return decode


async def _heartbeat(gaps: list[float], stop_event: asyncio.Event) -> None:
    last = time.monotonic()
    while not stop_event.is_set():
        await asyncio.sleep(0.01)
        now = time.monotonic()
        gaps.append(now - last)
        last = now


# ---------------------------------------------------------------------------
# heartbeat_backlog: a 20-chunk backlog with a 50ms-per-chunk detector
# ---------------------------------------------------------------------------


async def test_heartbeat_backlog_stays_under_300ms():
    source = _QueueSource([_CHUNK] * 20)
    detector = _SleepingDetector(sleep_s=0.05)
    runner = SourceRunner("camera", source, detector, lambda chunk: chunk, _unused_run_turn_fn)

    gaps: list[float] = []
    stop_event = asyncio.Event()
    heartbeat_task = asyncio.ensure_future(_heartbeat(gaps, stop_event))
    await asyncio.sleep(0)  # let the heartbeat get its first sleep in flight before the drain starts

    run_task = asyncio.ensure_future(runner.run())
    await run_task

    stop_event.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(heartbeat_task, timeout=1)

    assert gaps, "the heartbeat never ran at all"
    assert max(gaps) < 0.3, f"largest heartbeat gap was {max(gaps):.3f}s"


# ---------------------------------------------------------------------------
# heartbeat_barge_in: the same heartbeat check through _watch_barge_in
# ---------------------------------------------------------------------------


async def test_heartbeat_barge_in_stays_under_300ms():
    source = _QueueSource([_CHUNK] * 20)

    def slow_decode(chunk: bytes) -> bytes:
        time.sleep(0.05)
        return chunk

    detector = _SleepingDetector()
    runner = SourceRunner("camera", source, detector, slow_decode, _unused_run_turn_fn)
    monitor = BargeInMonitor(floor=0.0, min_duration_s=1.0, guard_window_s=0.0, enabled=True)
    monitor.mark_transcript_done()

    gaps: list[float] = []
    stop_event = asyncio.Event()
    heartbeat_task = asyncio.ensure_future(_heartbeat(gaps, stop_event))
    await asyncio.sleep(0)

    await runner._watch_barge_in(monitor)

    stop_event.set()
    with contextlib.suppress(asyncio.CancelledError):
        await asyncio.wait_for(heartbeat_task, timeout=1)
    runner._detector_executor.shutdown(wait=True)

    assert gaps, "the heartbeat never ran at all"
    assert max(gaps) < 0.3, f"largest heartbeat gap was {max(gaps):.3f}s"


# ---------------------------------------------------------------------------
# off_loop: every process()/decode call runs on one worker, never the loop
# ---------------------------------------------------------------------------


async def test_detector_and_decode_never_run_on_the_loop_thread():
    loop_thread_id = threading.get_ident()
    idents: list[int] = []
    lock = threading.Lock()
    decode = _recording_decode(idents, lock)
    detector = _SleepingDetector(idents=idents, lock=lock)

    source = _QueueSource([_CHUNK, _CHUNK])
    runner = SourceRunner("camera", source, detector, decode, _unused_run_turn_fn)

    await runner._process_chunk(_CHUNK)
    await runner._process_chunk(_CHUNK)

    monitor = BargeInMonitor(floor=0.0, min_duration_s=1.0, guard_window_s=0.0, enabled=True)
    monitor.mark_transcript_done()
    await runner._watch_barge_in(monitor)

    runner._detector_executor.shutdown(wait=True)

    assert idents, "no call was ever recorded"
    assert loop_thread_id not in idents
    assert len(set(idents)) == 1, f"more than one worker thread was used: {set(idents)}"


# ---------------------------------------------------------------------------
# no_overlap: one worker per runner serializes the wake and barge-in calls
# ---------------------------------------------------------------------------


class _NoHitDetector:
    def process(self, chunk: bytes) -> WakeHit | None:
        return None

    def close(self) -> None:
        pass


async def test_no_two_native_calls_for_one_source_ever_overlap():
    lock = threading.Lock()
    state = {"active": 0, "peak": 0}
    started = threading.Event()

    def blocking_decode(chunk: bytes) -> bytes:
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        started.set()
        time.sleep(0.2)
        with lock:
            state["active"] -= 1
        return chunk

    source = _QueueSource([_CHUNK, _CHUNK, _CHUNK])
    runner = SourceRunner("camera", source, _NoHitDetector(), blocking_decode, _unused_run_turn_fn)
    monitor = BargeInMonitor(floor=0.0, min_duration_s=1.0, guard_window_s=0.0, enabled=True)
    monitor.mark_transcript_done()

    listener_task = asyncio.ensure_future(runner._watch_barge_in(monitor))
    await asyncio.to_thread(started.wait)
    listener_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await listener_task

    # The cancelled listener's own decode call is very likely still running
    # in the worker thread right now -- cancelling the `await` never stops
    # a call already in progress there. This proves the single worker
    # queues the two call sites behind each other rather than running them
    # side by side.
    await runner._process_chunk(_CHUNK)

    runner._detector_executor.shutdown(wait=True)

    assert state["peak"] == 1, f"peak concurrent native calls was {state['peak']}"


# ---------------------------------------------------------------------------
# in_flight: run() only returns once its own in-flight call has finished
# ---------------------------------------------------------------------------


class _SlowDetector:
    def __init__(self, block_s: float) -> None:
        self._block_s = block_s
        self.started = threading.Event()
        self._lock = threading.Lock()
        self.calls_in_progress = 0
        self.calls_finished = 0

    def process(self, chunk: bytes) -> WakeHit | None:
        with self._lock:
            self.calls_in_progress += 1
        self.started.set()
        time.sleep(self._block_s)
        with self._lock:
            self.calls_in_progress -= 1
            self.calls_finished += 1
        return None

    def close(self) -> None:
        pass


async def test_run_returns_only_after_its_in_flight_call_finishes():
    detector = _SlowDetector(block_s=0.2)
    source = _QueueSource([_CHUNK])
    runner = SourceRunner("camera", source, detector, lambda chunk: chunk, _unused_run_turn_fn)

    run_task = asyncio.ensure_future(runner.run())
    await asyncio.to_thread(detector.started.wait)
    # Off the loop, the call is still running in its own worker thread at
    # this point, so `run()` has not returned yet -- proving cancellation
    # can actually land while a call is genuinely in flight, not after the
    # (single, un-interruptible) chunk has already finished processing on
    # the loop thread itself.
    still_in_flight = not run_task.done()
    run_task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await run_task

    assert still_in_flight, "the detector call had already finished before cancellation could reach it"
    assert detector.calls_in_progress == 0
    assert detector.calls_finished == 1


# ---------------------------------------------------------------------------
# stall_reporter: LoopStallReporter logs a blocked/unblocked pair
# ---------------------------------------------------------------------------


def _block_the_loop_on_purpose() -> None:
    time.sleep(0.6)


async def test_stall_reporter_logs_blocked_then_unblocked(caplog):
    caplog.set_level(logging.WARNING, logger="atlas.loop_stall")

    reporter = LoopStallReporter(threshold_s=0.2, poll_s=0.02)
    reporter.start()
    try:
        _block_the_loop_on_purpose()
        await asyncio.sleep(0.1)
    finally:
        reporter.stop()

    assert reporter._thread is not None
    assert not reporter._thread.is_alive()

    # Match on the message's own prefix, not a bare substring: this test's
    # own function name contains the word "unblocked", and that name
    # appears inside the "blocked" record's captured stack trace too.
    blocked = [r for r in caplog.records if r.name == "atlas.loop_stall" and r.getMessage().startswith("event loop blocked")]
    unblocked = [r for r in caplog.records if r.name == "atlas.loop_stall" and r.getMessage().startswith("event loop unblocked")]
    assert len(blocked) == 1, caplog.text
    assert "_block_the_loop_on_purpose" in blocked[0].getMessage()
    assert len(unblocked) == 1, caplog.text


async def test_stall_reporter_logs_nothing_for_ordinary_awaits(caplog):
    caplog.set_level(logging.WARNING, logger="atlas.loop_stall")

    reporter = LoopStallReporter(threshold_s=0.2, poll_s=0.02)
    reporter.start()
    try:
        for _ in range(30):
            await asyncio.sleep(0.01)
    finally:
        reporter.stop()

    stall_records = [r for r in caplog.records if r.name == "atlas.loop_stall"]
    assert stall_records == []


# ---------------------------------------------------------------------------
# Task 2 (D3): the other blocking calls -- recorder writes, retention
# rmtree, Piper resample, and the ha child's full /api/states parse.
# ---------------------------------------------------------------------------


async def test_recorder_close_runs_off_the_loop(fake_audio_source, fake_stt, fake_brain, fake_tts, tmp_path, monkeypatch):
    from atlas.config import SessionConfig
    from atlas.providers.base import BrainReply, FinalTranscript
    from atlas.session.recorder import SessionRecorder
    from atlas.timing import TurnTimings
    from atlas.turn.controller import run_turn

    loop_thread_id = threading.get_ident()
    idents: list[int] = []
    close_calls = {"n": 0}
    original_close = SessionRecorder.close

    def recording_close(self, timings_arg):
        idents.append(threading.get_ident())
        close_calls["n"] += 1
        return original_close(self, timings_arg)

    monkeypatch.setattr(SessionRecorder, "close", recording_close)

    config = SessionConfig(dir=str(tmp_path), record_audio=True)
    timings = TurnTimings()
    recorder = SessionRecorder(config, timings)

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="turned on the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host=None,
        tools_schema=[],
        system_prompt="",
        max_tool_rounds=3,
        timings=timings,
        session_recorder=recorder,
    )

    assert close_calls["n"] == 1, "close() must be idempotent -- called exactly once"
    assert idents and idents[0] != loop_thread_id
    assert (recorder.directory / "events.jsonl").exists()


async def test_retention_sweep_runs_off_the_loop(tmp_path, monkeypatch):
    from datetime import datetime

    import atlas.session.retention as retention_module
    from atlas.session.retention import RetentionScheduler

    loop_thread_id = threading.get_ident()
    idents: list[int] = []

    expired_dir = tmp_path / "20200101T000000000000Z-deadbeef"
    expired_dir.mkdir()
    (expired_dir / "events.jsonl").write_text("", encoding="utf-8")

    real_sweep = retention_module.sweep_expired_sessions

    def recording_sweep(*args, **kwargs):
        idents.append(threading.get_ident())
        return real_sweep(*args, **kwargs)

    monkeypatch.setattr(retention_module, "sweep_expired_sessions", recording_sweep)

    async def _instant_sleep(_seconds: float) -> None:
        await asyncio.sleep(0)

    scheduler = RetentionScheduler(
        tmp_path,
        retain_days=7,
        interval_s=0.01,
        clock=lambda: datetime.now(UTC),
        sleep=_instant_sleep,
    )
    scheduler.start()
    try:
        for _ in range(200):
            if scheduler.sweep_count >= 1:
                break
            await asyncio.sleep(0.01)
        else:
            raise AssertionError("the sweep never ran")
    finally:
        await scheduler.stop()

    assert idents and idents[0] != loop_thread_id
    assert not expired_dir.exists()


async def test_piper_resample_runs_off_the_loop(tmp_path, monkeypatch):
    import numpy as np

    import atlas.providers.tts_piper as tts_piper_module
    from atlas.config import TtsConfig
    from atlas.providers.tts_piper import PiperTts
    from atlas.providers.tts_xai import SinkFormat

    loop_thread_id = threading.get_ident()
    idents: list[int] = []
    real_resample = tts_piper_module._resample_pcm16

    def recording_resample(pcm16, from_rate, to_rate):
        idents.append(threading.get_ident())
        return real_resample(pcm16, from_rate, to_rate)

    monkeypatch.setattr(tts_piper_module, "_resample_pcm16", recording_resample)

    class _FakeAudioChunk:
        def __init__(self, pcm16: bytes, sample_rate: int) -> None:
            self.audio_int16_bytes = pcm16
            self.sample_rate = sample_rate

    class _FakeVoice:
        def __init__(self, chunks) -> None:
            self.chunks = chunks

        def synthesize(self, text: str, syn_config=None):
            return iter(self.chunks)

    voice_path = tmp_path / "voice.onnx"
    voice_path.write_bytes(b"fake-voice-weights")
    config_path = tmp_path / "voice.onnx.json"
    config_path.write_text("{}", encoding="utf-8")
    config = TtsConfig(piper_voice_path=str(voice_path), piper_config_path=str(config_path))

    pcm16 = (np.arange(2205, dtype=np.int16) - 1000).tobytes()  # 100ms @ 22050 Hz
    fake_voice = _FakeVoice([_FakeAudioChunk(pcm16, 22050)])
    tts = PiperTts(config, load_voice=lambda cfg: fake_voice)

    audio = await tts.synthesize_once("turn on the lights", sink=SinkFormat(codec="alaw", sample_rate=8000))

    assert idents and idents[0] != loop_thread_id
    assert len(audio) == 800  # same camera-sink expectation test_local_providers.py already asserts


async def test_ha_list_entities_parse_runs_off_the_loop(fake_ha, monkeypatch):
    import atlas_mcp.ha as ha_module
    from atlas_mcp.safety import Policy

    loop_thread_id = threading.get_ident()
    idents: list[int] = []
    real_allow_read = ha_module.allow_read

    def recording_allow_read(entity_id):
        idents.append(threading.get_ident())
        return real_allow_read(entity_id)

    monkeypatch.setattr(ha_module, "allow_read", recording_allow_read)

    entities = await ha_module.handle_list_entities(Policy(), fake_ha.client, "http://ha.invalid", "test-token")

    assert idents and idents[0] != loop_thread_id
    assert entities
    assert {"entity_id", "friendly_name", "state"} <= set(entities[0].keys())
