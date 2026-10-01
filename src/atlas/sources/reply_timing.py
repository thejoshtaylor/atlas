"""Where a reply is in time, on the Pi's playback queue.

TTS is batch. `_speak_direct` writes a whole reply to the Pi in a burst, so
the write time of a chunk is not the time it plays (Phase 13, RESEARCH
Finding 1). `ReplyCursor` keeps the end of the queue as the monitor sees it:
the moment the last byte written so far finishes playing. It also keeps one
record per finished utterance, so a later plan can tell what the reply said
and when.

Pure: no asyncio, no clock reads. The caller passes `now`.
"""

from __future__ import annotations

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
