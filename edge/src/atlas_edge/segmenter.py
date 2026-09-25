"""Pure pre-roll / live / tail segmenter (D-05, D-10).

No I/O and no clock -- `Segmenter.push` is a deterministic state machine
driven entirely by its arguments. The caller (`service.py`) supplies
`is_speech` from `vad.py`'s `SileroGate` and `captured_at` from
`capture.py`'s own monotonic capture time.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class VadEvent:
    """`kind` is `"vad.start"` or `"vad.end"`. `seq` matches a `vad.end`
    to the `vad.start` that opened the segment it closes -- it only
    changes when a new segment opens, never on `vad.end` itself."""

    kind: str
    seq: int


@dataclass(frozen=True)
class AudioOut:
    """One frame of audio to send. `live=False` marks buffered pre-roll
    history (sent once, right after `vad.start`); `live=True` marks a
    frame captured in real time, including every frame of the tail.
    `captured_at` is always the value this frame itself was pushed with,
    never the value of whatever frame is currently being processed."""

    frame: bytes
    live: bool
    captured_at: float


def frames_for_ms(ms: int, frame_samples: int, sample_rate: int) -> int:
    """How many `frame_samples`-sized frames at `sample_rate` cover `ms`
    milliseconds, rounded *up* -- so a hello's `pre_roll_ms`/`tail_ms`
    never under-covers when it doesn't divide evenly."""
    frame_ms = frame_samples * 1000 / sample_rate
    return math.ceil(ms / frame_ms)


class Segmenter:
    """The idle / speech / tail state machine (D-05, D-10):

    - idle, not speech: buffer the frame as pre-roll history; emit nothing.
    - idle, speech: emit `vad.start`, then the buffered pre-roll oldest
      first, then this frame live; move to speech.
    - speech, speech: emit this frame live.
    - speech, not speech: emit `vad.end` at once, then up to `tail_frames`
      frames live (this frame is the first of them); move to tail (or
      straight back to idle if `tail_frames` is 0).
    - tail, speech: emit a new `vad.start` (no pre-roll replay -- the
      audio never stopped) and this frame live; move to speech.
    - tail, not speech: emit this frame live, counting down the tail;
      move to idle once the count reaches zero.
    """

    def __init__(self, pre_roll_frames: int, tail_frames: int) -> None:
        self._pre_roll_frames = pre_roll_frames
        self._tail_frames = tail_frames
        self._pre_roll: "deque[tuple[bytes, float]]" = deque(maxlen=pre_roll_frames)
        self._state = "idle"
        self._tail_remaining = 0
        self._next_seq = 0
        self._active_seq = 0

    def _begin_speech(
        self, frame: bytes, captured_at: float, *, replay_pre_roll: bool
    ) -> "list[VadEvent | AudioOut]":
        self._active_seq = self._next_seq
        self._next_seq += 1
        out: "list[VadEvent | AudioOut]" = [VadEvent("vad.start", self._active_seq)]
        if replay_pre_roll:
            for pre_frame, pre_captured_at in self._pre_roll:
                out.append(AudioOut(pre_frame, False, pre_captured_at))
            self._pre_roll.clear()
        out.append(AudioOut(frame, True, captured_at))
        self._state = "speech"
        return out

    def push(self, frame: bytes, is_speech: bool, captured_at: float) -> "list[VadEvent | AudioOut]":
        if self._state == "idle":
            if is_speech:
                return self._begin_speech(frame, captured_at, replay_pre_roll=True)
            self._pre_roll.append((frame, captured_at))
            return []

        if self._state == "speech":
            if is_speech:
                return [AudioOut(frame, True, captured_at)]
            out: "list[VadEvent | AudioOut]" = [VadEvent("vad.end", self._active_seq)]
            self._tail_remaining = self._tail_frames
            if self._tail_remaining > 0:
                out.append(AudioOut(frame, True, captured_at))
                self._tail_remaining -= 1
            self._state = "tail" if self._tail_remaining > 0 else "idle"
            return out

        # tail
        if is_speech:
            return self._begin_speech(frame, captured_at, replay_pre_roll=False)
        out = [AudioOut(frame, True, captured_at)]
        self._tail_remaining -= 1
        if self._tail_remaining <= 0:
            self._state = "idle"
        return out
