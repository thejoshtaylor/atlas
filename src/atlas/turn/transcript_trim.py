"""Re-transcribe a turn that a speaker change ended early (Phase 12, D-04).

A voice that speaks right after the kept voice, in the same VAD segment,
ends the kept turn at the change point. The turn's first transcript still
holds the next voice's first words, because speech-to-text ran on the whole
stream. The next voice now has its own turn (plan 12-06), so those words
would be answered twice.

This module transcribes again, from the turn's own first frame to the split
frame. It needs no word timestamps, so it works with every provider (Research
Open Question 2). It runs only for a turn that a split ended, which is rare.
The `transcript.trim` event records the method and the time, so the real runs
show how often it runs and what it costs.

The frames are the turn's own history slice, so the next voice's frames never
reach the second transcription (T-12-26).
"""

from __future__ import annotations

import asyncio
import contextlib
import logging
import time
from typing import Any, AsyncIterator

from atlas.audio.channels import stt_view
from atlas.providers.base import FinalTranscript

logger = logging.getLogger("atlas.turn.transcript_trim")

TRANSCRIPT_TRIM_EVENT = "transcript.trim"

# The frames were read from the source, so `stt_view` picks the ASR channel
# the same way the first transcription did.


async def _aclose(stream: Any) -> None:
    aclose = getattr(stream, "aclose", None)
    if aclose is not None:
        with contextlib.suppress(Exception):
            await aclose()


async def retranscribe(stt: Any, frames: list[bytes], source_format: Any, *, timeout_s: float) -> str | None:
    """The final transcript's text for exactly `frames`, or `None` when the
    stream raises, gives no final transcript, or `timeout_s` passes.

    The `finalize` event is set after the last frame, because a history slice
    has no trailing silence for the provider's own endpointing."""
    finalize = asyncio.Event()

    async def feed() -> AsyncIterator[bytes]:
        for frame in frames:
            yield frame
        finalize.set()

    feed_stream = feed()
    stream: Any = None

    async def read() -> str | None:
        nonlocal stream
        view, fmt = stt_view(feed_stream, source_format)
        stream = stt.stream(view, fmt, finalize=finalize)
        final: FinalTranscript | None = None
        async for event in stream:
            if isinstance(event, FinalTranscript):
                final = event
        return final.text if final is not None else None

    try:
        return await asyncio.wait_for(read(), timeout=timeout_s)
    except asyncio.CancelledError:
        raise
    except asyncio.TimeoutError:
        logger.warning("re-transcription did not finish within %.1fs", timeout_s)
        return None
    except Exception:
        logger.exception("re-transcription failed; keeping the first transcript")
        return None
    finally:
        if stream is not None:
            await _aclose(stream)
        await _aclose(feed_stream)


async def trim_at_split(
    turn_context: Any, speaker_span: Any, stt: Any, source_format: Any, *, timeout_s: float
) -> str | None:
    """The new transcript for a turn a speaker change ended, or `None` when
    the turn keeps its first one: no split, no frame history, or a failed
    second transcription. Records one `transcript.trim` event per attempt."""
    if turn_context is None or speaker_span is None or turn_context.replay_until is None:
        return None
    split_event = getattr(speaker_span, "split_event", None)
    split_frame_index = getattr(speaker_span, "split_frame_index", None)
    if split_event is None or not split_event.is_set() or split_frame_index is None:
        return None
    frames = turn_context.replay_until(split_frame_index)
    if frames is None:
        # The frames left the history buffer.
        turn_context.record_event({"type": TRANSCRIPT_TRIM_EVENT, "method": "unavailable"})
        return None
    started = time.monotonic()
    text = await retranscribe(stt, frames, source_format, timeout_s=timeout_s)
    turn_context.record_event(
        {
            "type": TRANSCRIPT_TRIM_EVENT,
            "method": "retranscribe",
            "frames": len(frames),
            "duration_ms": (time.monotonic() - started) * 1000.0,
            "ok": text is not None,
        }
    )
    return text
