"""Where a reply is in time, on the Pi's playback queue.

TTS is batch. `_speak_direct` writes a whole reply to the Pi in a burst, so
the write time of a chunk is not the time it plays (Phase 13, RESEARCH
Finding 1). `ReplyCursor` keeps the end of the queue as the monitor sees it:
the moment the last byte written so far finishes playing. It also keeps one
record per finished utterance, so a later plan can tell what the reply said
and when.

The assistant's own reply can say "Atlas". If residual echo of that word
reaches the wake detector, the reply would interrupt itself. D-02: the
server does not compare text. It computes a time window around each
wake word in the reply (`wake_word_windows`), and a hit that falls inside a
window is the reply's own echo (`in_any_window`). Character position stands
in for time, so the windows are wide on purpose.

D-B (debug edge-interrupt-handover-unverified): during playback the listener
accepts the keyword alone ("atlas"), because the echo suppressor clips the
"hey". A reply that says the keyword anywhere (`says_wake_word`) gives that
up: for that reply the listener needs the full phrase, as when idle.

Pure: no asyncio, no clock reads. The caller passes `now`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class ReplyUtterance:
    """One utterance that was written to the speaker.

    `start` and `end` are seconds in the `time.monotonic()` domain. `end` is
    the estimated moment the last byte finishes playing.
    """

    text: str
    start: float
    end: float


def bytes_per_second(sink) -> int | None:
    """How many bytes one second of audio takes in `sink`'s format, or
    `None` for no sink. The rule is the one `estimate_playback_end` uses:
    16-bit PCM is two bytes per sample, A-law and mu-law are one."""
    if sink is None:
        return None
    return sink.sample_rate * (2 if sink.codec == "pcm" else 1)


class ReplyCursor:
    """The end of the audio queued on the speaker, and the record of each
    utterance that made it there."""

    def __init__(self) -> None:
        self.playing_until: float | None = None
        self.utterances: list[ReplyUtterance] = []
        self._start: float | None = None
        self._bytes = 0
        self._bytes_per_second: int | None = None

    def note_audio_written(self, nbytes: int, now: float, sink, *, utterance_start: bool) -> None:
        """Count `nbytes` that reached the speaker at `now`.

        An utterance starts at `now`, or at the queue end when an earlier
        utterance (a filler) is still playing, so an answer queued behind a
        filler is placed after it. Does nothing with no sink or no bytes.
        """
        rate = bytes_per_second(sink)
        if rate is None or nbytes <= 0:
            return
        if utterance_start or self._start is None:
            self._start = now if self.playing_until is None else max(now, self.playing_until)
            self._bytes = 0
        self._bytes_per_second = rate
        self._bytes += nbytes
        self.playing_until = self._start + self._bytes / rate

    def note_utterance_end(self, text: str) -> None:
        """Record the utterance that was just written, and clear its state.
        Does nothing when no byte was written for it."""
        if self._start is None or self._bytes <= 0 or self.playing_until is None:
            self._start = None
            self._bytes = 0
            return
        self.utterances.append(ReplyUtterance(text, self._start, self.playing_until))
        self._start = None
        self._bytes = 0

    def spoken_s(self, now: float) -> float:
        """Seconds of the latest utterance that were already heard at `now`.
        `0.0` when nothing was written."""
        if self._start is not None and self._bytes_per_second:
            start, end = self._start, self._start + self._bytes / self._bytes_per_second
        elif self.utterances:
            start, end = self.utterances[-1].start, self.utterances[-1].end
        else:
            return 0.0
        return max(0.0, min(now - start, end - start))


def _wake_word_pattern(wake_phrase: str) -> "re.Pattern[str] | None":
    """A whole-word, case-insensitive match for the wake word: the last word
    of `wake_phrase` ("hey atlas" gives "atlas"). `None` for an empty phrase."""
    words = [w.strip(".,!?;:\"'()[]") for w in wake_phrase.lower().split()]
    words = [w for w in words if w]
    if not words:
        return None
    return re.compile(rf"\b{re.escape(words[-1])}\b", re.IGNORECASE)


def says_wake_word(text: str, wake_phrase: str) -> bool:
    """True when `text` says the wake word as a whole word, in any case
    ("Atlas," counts, "Atlantic" does not). False for an empty phrase.

    D-B: a reply for which this is True needs the full wake phrase to be
    interrupted during playback, so its own echo of the word cannot stop it."""
    pattern = _wake_word_pattern(wake_phrase)
    return pattern is not None and pattern.search(text) is not None


def wake_word_windows(
    utterances,
    *,
    wake_phrase: str,
    margin_s: float,
    echo_delay_s: float,
) -> tuple[tuple[float, float], ...]:
    """The time windows in which a wake hit is the reply's own echo.

    The wake word is the last word of `wake_phrase` ("hey atlas" gives
    "atlas"). Each whole-word, case-insensitive match in an utterance's text
    gets one window. The word's start and end are estimated from its
    character position, as a share of the utterance's play time. The window
    adds `margin_s` on both sides, and `echo_delay_s` (the calibrated echo
    delay) on the high side only. An empty phrase gives no windows.
    """
    pattern = _wake_word_pattern(wake_phrase)
    if pattern is None:
        return ()
    windows: list[tuple[float, float]] = []
    for utterance in utterances:
        length = max(1, len(utterance.text))
        span = utterance.end - utterance.start
        for match in pattern.finditer(utterance.text):
            word_start = utterance.start + span * (match.start() / length)
            word_end = utterance.start + span * (match.end() / length)
            windows.append((word_start - margin_s, word_end + margin_s + echo_delay_s))
    return tuple(windows)


def in_any_window(t: float, windows) -> bool:
    """True when `t` is inside any `(low, high)` window, edges included."""
    return any(low <= t <= high for low, high in windows)
