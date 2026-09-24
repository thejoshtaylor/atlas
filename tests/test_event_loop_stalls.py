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
