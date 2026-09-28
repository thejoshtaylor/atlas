"""Speech accumulates into fixed-duration windows, ready to embed (D-09).

Only frames inside a running Pi segment count, and only when the frame's
RMS (`atlas.audio.energy.rms_amplitude`) is at or above `speech_rms_floor`
-- a frame below the floor never enters a window and never advances the
cadence, so pre-roll silence and pauses do not count as speech. Windows are
sequential and never overlap: accumulated speech is tracked in whole PCM16
samples, and a window closes the instant it holds `window_ms` of speech --
even mid-frame, splitting the frame that crosses the boundary so a window's
own `speech_ms` lands on the configured target rather than drifting by up
to one frame's duration every time it closes.

This module is pure: no I/O, no imports from `turn/`, `providers/`, `db/`,
or `transports/`. It only shapes bytes that already arrived.
"""

from __future__ import annotations

from dataclasses import dataclass

from atlas.audio.energy import rms_amplitude

_BYTES_PER_SAMPLE = 2  # PCM16, little-endian -- the only encoding this module handles.


@dataclass(frozen=True)
class SpeechWindow:
    """One closed (or flushed) window of accumulated speech.

    `first_frame_index`/`last_frame_index` name the raw frame indices this
    window's PCM was drawn from -- when a frame straddles a window
    boundary, both the closing window and the window that continues after
    it record that same frame's index, since a sample-level split has no
    single frame index of its own to report.
    """

    segment_seq: int
    first_frame_index: int
    last_frame_index: int
    started_at: float
    speech_ms: float
    pcm: bytes


class SpeechWindowAccumulator:
    """Turns a stream of raw PCM16 frames into `SpeechWindow`s.

    `window_ms`/`min_window_ms`/`speech_rms_floor` are the D-09 knobs the
    Phase 11 spike measures; this class only applies the rule, it never
    guesses a default for any of them.
    """

    def __init__(
        self,
        *,
        sample_rate: int,
        window_ms: int,
        min_window_ms: int,
        speech_rms_floor: float,
    ) -> None:
        self._sample_rate = sample_rate
        self._window_ms = window_ms
        self._min_window_ms = min_window_ms
        self._speech_rms_floor = speech_rms_floor
        self._window_samples = round(window_ms * sample_rate / 1000)
        self._segment_seq: "int | None" = None
        self._buffer_pcm = bytearray()
        self._buffer_first_frame_index: "int | None" = None
        self._buffer_started_at: "float | None" = None
        self._last_frame_index: "int | None" = None
        self._last_at: "float | None" = None

    def start_segment(self, seq: int) -> None:
        """Begin segment `seq`, discarding any partial buffer left over
        from whatever segment came before it -- a window never spans two
        segments."""
        self._segment_seq = seq
        self._reset_buffer_state()

    def push(self, mono_pcm16: bytes, frame_index: int, at: float) -> "SpeechWindow | None":
        """Feed one raw frame. Returns a closed `SpeechWindow` the instant
        accumulated speech reaches `window_ms`, else `None`.

        RMS is checked once, against the whole incoming frame (the same
        per-frame granularity the wake detector and barge-in reader already
        use) -- there is no sub-frame speech/silence decision.
        """
        if self._segment_seq is None:
            raise RuntimeError("SpeechWindowAccumulator.push() called before start_segment()")
        if rms_amplitude(mono_pcm16) < self._speech_rms_floor:
            return None
        if not self._buffer_pcm:
            self._buffer_first_frame_index = frame_index
            self._buffer_started_at = at
        self._buffer_pcm.extend(mono_pcm16)
        self._last_frame_index = frame_index
        self._last_at = at
        if len(self._buffer_pcm) // _BYTES_PER_SAMPLE >= self._window_samples:
            return self._close_window(self._window_samples)
        return None

    def flush(self, at: float) -> "SpeechWindow | None":
        """Emit the current partial buffer as a window if it already holds
        `min_window_ms` of speech, without ending the segment. `at` is
        unused when nothing is emitted; kept for a symmetrical call site
        with `push`/`end_segment`."""
        del at
        num_samples = len(self._buffer_pcm) // _BYTES_PER_SAMPLE
        if num_samples > 0 and self._ms_for(num_samples) >= self._min_window_ms:
            return self._close_window(num_samples)
        return None

    def end_segment(self) -> "SpeechWindow | None":
        """Close out the running segment: emit a tail window if it holds
        `min_window_ms` of speech, else discard the partial buffer
        silently. Either way, the segment is over after this call."""
        num_samples = len(self._buffer_pcm) // _BYTES_PER_SAMPLE
        window: "SpeechWindow | None" = None
        if num_samples > 0 and self._ms_for(num_samples) >= self._min_window_ms:
            window = self._close_window(num_samples)
        self._segment_seq = None
        self._reset_buffer_state()
        return window

    def _ms_for(self, num_samples: int) -> float:
        return num_samples / self._sample_rate * 1000.0

    def _reset_buffer_state(self) -> None:
        self._buffer_pcm = bytearray()
        self._buffer_first_frame_index = None
        self._buffer_started_at = None
        self._last_frame_index = None
        self._last_at = None

    def _close_window(self, num_samples: int) -> SpeechWindow:
        num_bytes = num_samples * _BYTES_PER_SAMPLE
        pcm = bytes(self._buffer_pcm[:num_bytes])
        window = SpeechWindow(
            segment_seq=self._segment_seq,  # type: ignore[arg-type]
            first_frame_index=self._buffer_first_frame_index,  # type: ignore[arg-type]
            last_frame_index=self._last_frame_index,  # type: ignore[arg-type]
            started_at=self._buffer_started_at,  # type: ignore[arg-type]
            speech_ms=self._ms_for(num_samples),
            pcm=pcm,
        )
        remainder = self._buffer_pcm[num_bytes:]
        self._buffer_pcm = bytearray(remainder)
        if remainder:
            # The frame that closed this window straddled the boundary --
            # the leftover samples start a new, still-open window that
            # continues from that same frame's index/timestamp.
            self._buffer_first_frame_index = self._last_frame_index
            self._buffer_started_at = self._last_at
        else:
            self._buffer_first_frame_index = None
            self._buffer_started_at = None
        return window
