"""run_service: capture -> VAD -> segmenter -> session (D-05, D-06).

`capture` opens once for the process's life -- a reconnect never reopens
ALSA. Each session (each `run_forever` retry) gets a fresh `Segmenter`
and a fresh gate, so a segment never spans a reconnect (D-05). This
module never imports `sounddevice` or `sherpa_onnx` directly; it only
calls `capture`/`gate_factory`, which own those imports.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any, AsyncIterator, Callable

import numpy as np

from atlas_edge import protocol
from atlas_edge.client import LiveFrame, run_forever
from atlas_edge.protocol import Hello
from atlas_edge.segmenter import AudioOut, Segmenter, VadEvent, frames_for_ms

logger = logging.getLogger(__name__)

OutboundItem = "bytes | str | LiveFrame"


async def run_service(
    config: Any,
    *,
    capture: Any,
    gate_factory: Callable[[], Any],
    on_reply_audio: Callable[[bytes], Any],
    on_live_frame_sent: "Callable[[float, float], Any] | None" = None,
    events_hook: "Callable[[], AsyncIterator[str]] | None" = None,
    runner: Callable[..., Any] = run_forever,
    stop: "asyncio.Event | None" = None,
) -> None:
    """Opens `capture` once, then hands `runner` (`run_forever` in
    production) a `make_outbound(hello)` closure that builds a fresh
    `Segmenter`/gate per session and turns captured frames into the
    outbound protocol stream."""
    capture.start()
    try:

        def make_outbound(hello: Hello) -> AsyncIterator[Any]:
            return _outbound_for_session(hello, capture, gate_factory, events_hook)

        await runner(
            config,
            make_outbound=make_outbound,
            on_reply_audio=on_reply_audio,
            on_live_frame_sent=on_live_frame_sent,
            stop=stop,
        )
    finally:
        capture.stop()


def _hello_matches_capture(hello: Hello, capture: Any) -> bool:
    return (
        hello.sample_rate == capture.sample_rate
        and hello.channels == capture.channels
        and hello.frame_samples == capture.frame_samples
    )


async def _outbound_for_session(
    hello: Hello,
    capture: Any,
    gate_factory: Callable[[], Any],
    events_hook: "Callable[[], AsyncIterator[str]] | None",
) -> AsyncIterator[Any]:
    if not _hello_matches_capture(hello, capture):
        logger.error(
            "protocol mismatch: hello (sample_rate=%s channels=%s frame_samples=%s) "
            "does not match what capture opened (sample_rate=%s channels=%s "
            "frame_samples=%s) -- sending no audio this session",
            hello.sample_rate,
            hello.channels,
            hello.frame_samples,
            capture.sample_rate,
            capture.channels,
            capture.frame_samples,
        )
        return

    segmenter = Segmenter(
        frames_for_ms(hello.pre_roll_ms, hello.frame_samples, hello.sample_rate),
        frames_for_ms(hello.tail_ms, hello.frame_samples, hello.sample_rate),
    )
    gate = gate_factory()

    capture_stream = _from_capture(hello, capture, segmenter, gate)
    if events_hook is None:
        async for item in capture_stream:
            yield item
        return

    async for item in _merge(capture_stream, events_hook()):
        yield item


async def _from_capture(
    hello: Hello, capture: Any, segmenter: Segmenter, gate: Any
) -> AsyncIterator[Any]:
    async for frame, captured_at in capture.frames():
        # A numpy stride view -- no copy -- picks out only the channel
        # the server's hello named as the ASR beam (D-09).
        samples = np.frombuffer(frame, dtype=np.int16)
        channel_samples = samples[hello.asr_channel :: hello.channels]
        is_speech = gate.push(channel_samples.tobytes())
        for item in segmenter.push(frame, is_speech, captured_at):
            if isinstance(item, VadEvent):
                if item.kind == "vad.start":
                    yield protocol.vad_start(item.seq)
                else:
                    yield protocol.vad_end(item.seq)
            elif isinstance(item, AudioOut):
                if item.live:
                    yield LiveFrame(item.frame, item.captured_at)
                else:
                    yield item.frame


async def _merge(a: AsyncIterator[Any], b: AsyncIterator[Any]) -> AsyncIterator[Any]:
    """Interleaves two async iterators, ending once both are exhausted.
    This is how `events_hook` (plan 10-09's DoA/latency messages) merges
    into the outbound stream without reshaping `_from_capture`'s loop."""
    queue: "asyncio.Queue[Any]" = asyncio.Queue()
    sentinel = object()

    async def _drain(iterator: AsyncIterator[Any]) -> None:
        async for item in iterator:
            await queue.put(item)
        await queue.put(sentinel)

    task_a = asyncio.ensure_future(_drain(a))
    task_b = asyncio.ensure_future(_drain(b))
    remaining = 2
    try:
        while remaining > 0:
            item = await queue.get()
            if item is sentinel:
                remaining -= 1
                continue
            yield item
    finally:
        for task in (task_a, task_b):
            if not task.done():
                task.cancel()
        await asyncio.gather(task_a, task_b, return_exceptions=True)
