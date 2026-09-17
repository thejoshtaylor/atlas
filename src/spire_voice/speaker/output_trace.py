"""`EmittedAudioTrace`: what was written toward the speaker, indexed by where
playback has got to -- never by when the bytes were written.

**The one governing decision, stated plainly because it is the whole point
of this module.** xAI's text-to-speech returns one utterance as a single
REST response: `_speak` (`turn/controller.py`) receives the whole thing and
writes it to the speaker FIFO as fast as the write loop can run, far faster
than real time. `ffmpeg_supervisor.py`'s `build_ffmpeg_argv` then reads that
FIFO as raw A-law at 8 kHz and pushes it with `-c copy` into a live stream --
draining it at real time. So "when was this chunk written" and "when will
this chunk be heard" are two different quantities, and a microphone reading
taken at a real wall-clock moment can only ever be compared against the
second one. Every entry here is therefore indexed by cumulative *playback*
duration from the start of the utterance -- derived from byte count and the
declared encoding -- and never by a `time.monotonic()` read taken at
`append()` time. There is deliberately no clock anywhere in this module.

Bounded by a configured span of audio rather than a count of entries, so
memory use is predictable regardless of chunk size (mirroring `audio/
ring.py`'s `PrerollBuffer`, which bounds the same way for the same reason).
Reset per utterance (`reset()`), so a fresh utterance's offsets are never
read against the filler's bytes -- `sources/runner.py`'s `BargeInMonitor.
mark_playback_started` calls it every time a new utterance starts playing,
filler and answer alike, the same call site that already restarts the
guard window.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

from spire_voice.audio.alaw import alaw_to_pcm16, bytes_per_sample
from spire_voice.audio.energy import rms_amplitude

# Long enough to hold any single reply this system plausibly speaks (even a
# multi-minute answer) in full, so an ordinary correlation query is never
# starved by the bound below -- short enough that a runaway utterance still
# cannot grow this trace without limit. Not a configured key: unlike
# `BargeInConfig`'s floor/duration/guard-window (measured or provisional
# quantities an operator might reasonably retune), this is a memory bound
# with no acoustic meaning of its own.
_DEFAULT_SPAN_S = 120.0


@dataclass(frozen=True)
class _Entry:
    start_s: float
    duration_s: float
    level: float


class EmittedAudioTrace:
    """A bounded record of what was written toward the speaker, one entry
    per chunk, each carrying the playback offset at which that chunk
    begins, its duration, and its root-mean-square level in
    `audio/energy.py`'s own normalized unit.

    `encoding`/`sample_rate` are supplied once, at construction, from the
    same `SourceFormat` `PrerollBuffer` is already sized from
    (`camera_source.source_format()`) -- `append()` itself takes only the
    raw chunk. `_speak` holds no codec (module docstring, `turn/
    controller.py`); this class is what makes that possible without
    guessing at a chunk's shape.
    """

    def __init__(self, *, encoding: str, sample_rate: int, span_s: float = _DEFAULT_SPAN_S) -> None:
        self._encoding = encoding
        self._sample_rate = sample_rate
        self._span_s = span_s
        self._bytes_per_sample = bytes_per_sample(encoding)
        self._entries: deque[_Entry] = deque()
        self._cursor_s = 0.0
        self._held_s = 0.0

    def reset(self) -> None:
        """Start this utterance's offsets over at zero, holding nothing
        from whatever utterance played before it."""
        self._entries.clear()
        self._cursor_s = 0.0
        self._held_s = 0.0

    def append(self, chunk: bytes) -> None:
        """Record one chunk actually written toward the speaker.

        The chunk's own byte count and this trace's declared encoding are
        the only inputs to its duration -- never a clock (module
        docstring). A-law and PCM16 renderings of the same audio produce
        the same level: both are converted to PCM16 (a no-op for PCM16
        itself) before `rms_amplitude` measures them, so the trace
        measures the sound, not the encoding it happened to arrive in.
        """
        if not chunk:
            return
        n_samples = len(chunk) // self._bytes_per_sample
        if n_samples == 0:
            return
        duration_s = n_samples / self._sample_rate
        pcm16 = chunk if self._encoding == "pcm" else alaw_to_pcm16(chunk)
        level = rms_amplitude(pcm16)
        self._entries.append(_Entry(start_s=self._cursor_s, duration_s=duration_s, level=level))
        self._cursor_s += duration_s
        self._held_s += duration_s
        while self._held_s > self._span_s and len(self._entries) > 1:
            oldest = self._entries.popleft()
            self._held_s -= oldest.duration_s

    def __len__(self) -> int:
        return len(self._entries)

    def level_at(self, playback_offset_s: float) -> float | None:
        """The level of whichever entry's playback span contains
        `playback_offset_s`, or `None` for an offset before the first
        entry or past the end of what has been written.

        `None` is a real answer -- "nothing was playing then" -- and is
        never conflated with `0.0`, which would read as "silence was
        playing." `sources/runner.py`'s correlation treats the two
        differently: silence explains nothing louder than the floor;
        "nothing was playing" explains nothing at all.
        """
        if not self._entries:
            return None
        first = self._entries[0]
        if playback_offset_s < first.start_s:
            return None
        last = self._entries[-1]
        if playback_offset_s >= last.start_s + last.duration_s:
            return None
        for entry in self._entries:
            if entry.start_s <= playback_offset_s < entry.start_s + entry.duration_s:
                return entry.level
        return None
