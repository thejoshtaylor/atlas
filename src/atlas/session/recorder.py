"""Per-turn on-disk session directory: exactly what a turn heard, said, and
timed -- the raw material Phase 8's tuning surface and plan 02-09's offline
scoring harness both read from (D-13, DBG-01, DBG-02).

Construct one `SessionRecorder` per turn, taking `SessionConfig` as an
explicit constructor parameter -- borrowed from `providers/tts_cache.py`'s
own habit of taking what it needs rather than reaching into a global
configuration from inside a write call. Borrowed, and NOT copied: that
cache is idempotent, content-keyed, and re-read by the process that wrote
it; this recorder is append-only, unique per turn (keyed by
`TurnTimings.turn_id`, never a second id generated here), and never
re-read by the process that wrote it. A reader coming from `tts_cache.py`
should not inherit that cache's mental model here -- the two modules only
share a directory-creation habit, nothing about how their files are used
afterward.

Privacy boundary, extended verbatim from `timing.py`: the files this
module writes may hold audio by design (D-13); its own log lines may not.
A log line names a path, a byte count, and an event count -- never
transcript text, reply text, or audio bytes -- because the log stream has
a different retention and access posture than the directory the
retention sweep (plan 02-08) governs.

D-15 is structural here, not a matter of discipline: `record_audio_chunk`
only ever receives what `turn/controller.py` taps off the frame iterator
`run_turn` already drains -- this module opens no second read of any
source and has no code path by which a continuous rolling capture could
be written.

This module writes all four of the session directory's artifacts: the raw
audio, the JSONL event stream, and the serialized `TurnTimings` record
directly, and the merged timeline through a deferred call into
`session/timeline.py` (see `close()`'s own comment for why that import is
deferred rather than module-level).
"""

from __future__ import annotations

import dataclasses
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from atlas.config import SessionConfig
from atlas.timing import TurnTimings

logger = logging.getLogger("atlas.session.recorder")

# Shared with `session/timeline.py`, which imports these two names rather
# than re-declaring them -- one place these filenames are spelled, so the
# recorder and the timeline reader can never disagree about where to look.
EVENTS_FILENAME = "events.jsonl"
TIMING_FILENAME = "timing.json"


class SessionRecorder:
    """Owns one turn's directory, created eagerly at construction.

    Eager creation -- not lazy, on first write -- is what makes a turn
    that crashes before producing anything still leave a folder behind:
    an absent folder would make a failed turn indistinguishable from a
    turn that never happened (T-02-22), and the folder existing is not
    allowed to depend on how far the turn got.
    """

    def __init__(
        self,
        config: SessionConfig,
        timings: TurnTimings,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config = config
        self._turn_id = timings.turn_id
        self._clock = clock
        self._audio_format: dict[str, Any] | None = None
        self._preroll_bytes = 0
        self._audio_chunks: list[bytes] = []
        self._events: list[dict[str, Any]] = []
        self._closed = False
        self.directory = Path(config.dir) / self._directory_name()
        self.directory.mkdir(parents=True, exist_ok=True)

    def _directory_name(self) -> str:
        """A sortable UTC timestamp plus the turn id -- never the
        timestamp alone (CD-5), so two turns starting in the same second,
        or even the same microsecond on a fast enough host, cannot
        collide. `turn_id` comes from `TurnTimings`, not a second id
        generated here, so a session folder and its timing log line
        always name the same turn.
        """
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        return f"{stamp}-{self._turn_id}"

    def set_audio_format(self, encoding: str, sample_rate: int) -> None:
        """Record which format the audio bytes are in, taken from the
        source's own declaration (`AudioSource.source_format()`) -- never
        assumed -- so a reader in Phase 8 can play them back without
        guessing.
        """
        self._audio_format = {"encoding": encoding, "sample_rate": sample_rate}

    def set_preroll_bytes(self, byte_count: int) -> None:
        """Record how many bytes at the head of the audio this turn is
        about to receive are replayed pre-roll, taken from the source's
        own declaration (`PrerollReplayingSource.preroll_bytes`) -- never
        assumed -- mirroring `set_audio_format`'s habit exactly (plan
        08-11, DBG-03).

        `close()` serializes the honest number, `min(declared, len of
        bytes actually received)`, rather than this declared count --
        this method only records what the source claims it will provide.
        """
        self._preroll_bytes = byte_count

    def record_audio_chunk(self, chunk: bytes) -> None:
        """Append one raw audio chunk, exactly as captured -- no
        transcode. Called only from the tap on the frame iterator
        `run_turn` already drains (D-15); never from a second,
        independent read of the source.
        """
        self._audio_chunks.append(chunk)

    def record_event(self, event: dict[str, Any]) -> None:
        """Append one event, in the order it occurred.

        `recorded_at` uses the same `time.monotonic()` clock domain
        `TurnTimings` uses, so `session/timeline.py` can compare an
        event's moment against a named stage's moment directly -- not a
        second, disagreeing clock. This is an event-sequencing mark, not
        one of `timing.py`'s own eight named stages, and duplicates none
        of them.

        The clock reading always wins (WR-02): `event` is spread first so
        that if a caller ever passes a dict already carrying a
        `"recorded_at"` key, this method's own reading overwrites it
        rather than being silently overwritten by it -- `dict` literal
        merge order means whichever key comes last wins, and
        `render_timeline`'s sort depends on every entry's `recorded_at`
        being this clock's own domain, never a caller-supplied value that
        might not be.
        """
        self._events.append({**event, "recorded_at": self._clock()})

    def close(self, timings: TurnTimings) -> None:
        """Write every artifact and mark this recorder closed.

        Idempotent: a second call is a no-op, so every exit path in
        `run_turn` can call this unconditionally from one `finally` block
        without needing to know which one actually got there first.
        """
        if self._closed:
            return
        self._closed = True

        events_path = self.directory / EVENTS_FILENAME
        event_lines = [json.dumps(event, sort_keys=True) for event in self._events]
        events_path.write_text("\n".join(event_lines) + ("\n" if event_lines else ""), encoding="utf-8")

        # The honest count, not the declared one (plan 08-11): a turn that
        # ended before draining its whole pre-roll records fewer pre-roll
        # bytes than the source declared, and a deployment that writes no
        # audio at all -- `self._config.record_audio and audio_bytes`,
        # the same guard the audio-file write below already uses -- must
        # report zero rather than a count against bytes that never
        # reached disk.
        audio_bytes = b"".join(self._audio_chunks)
        preroll_bytes = min(self._preroll_bytes, len(audio_bytes)) if self._config.record_audio and audio_bytes else 0

        timing_payload = _serialize_timings(timings, self._audio_format, preroll_bytes)
        timing_path = self.directory / TIMING_FILENAME
        timing_path.write_text(json.dumps(timing_payload, sort_keys=True, indent=2), encoding="utf-8")

        if self._config.record_audio and audio_bytes:
            encoding = (self._audio_format or {}).get("encoding", "raw")
            (self.directory / f"audio.{encoding}").write_bytes(audio_bytes)

        # Deferred, not module-level: `session/timeline.py` imports
        # `EVENTS_FILENAME`/`TIMING_FILENAME` back from this module at load
        # time, so a module-level import here would deadlock the two
        # modules' load order. By the time `close()` actually runs, this
        # module has always finished loading, so importing `timeline` here
        # only ever fetches or finishes a module that cannot be mid-load on
        # this side -- the same deferred-import shape `turn/controller.py`
        # already uses for `app.py`.
        from atlas.session import timeline

        rendered_timeline = timeline.render_timeline(self._events, timing_payload)
        timeline.write_timeline(self.directory, rendered_timeline)

        logger.info(
            "session recorded",
            extra={
                "turn_id": self._turn_id,
                "directory": str(self.directory),
                "audio_bytes": len(audio_bytes),
                "event_count": len(self._events),
            },
        )


def _serialize_timings(
    timings: TurnTimings, audio_format: dict[str, Any] | None, preroll_bytes: int
) -> dict[str, Any]:
    """The `TurnTimings` record, serialized from the object the turn
    already built -- this module measures nothing itself; there is one
    source of stage timings in this system and it is `timing.py`.

    `dataclasses.asdict` preserves an unreached stage as `None` (never a
    fabricated zero) and keeps two equal-valued stages as two separate
    keys -- neither collapsed into the other. Durations come from
    `timings.stage_durations_ms()` -- `timing.py`'s own derivation,
    measured from the previous stage actually reached, never recomputed
    here.

    `preroll_bytes` (plan 08-11, DBG-03) counts bytes at the head of the
    audio file actually written to disk -- the caller (`close()`) has
    already reduced it to `min(declared, len(audio_bytes))`, or `0` when
    no audio file is written, so this key reports truthfully rather than
    optimistically for a turn that ended early or a deployment that
    records no audio at all.
    """
    payload = dataclasses.asdict(timings)
    payload["stage_durations_ms"] = timings.stage_durations_ms()
    payload["end_of_speech_to_first_audio_ms"] = timings.end_of_speech_to_first_audio_ms
    payload["end_of_speech_to_answer_audio_ms"] = timings.end_of_speech_to_answer_audio_ms
    # 260924-4iv (item e): the true end-of-speech numbers, next to the
    # older end_of_speech_to_* pair above -- speech_end_at/
    # brain_first_round_at themselves arrive through dataclasses.asdict
    # already, needing no extra code here.
    payload["endpointing_delay_ms"] = timings.endpointing_delay_ms
    payload["speech_end_to_first_audio_ms"] = timings.speech_end_to_first_audio_ms
    payload["speech_end_to_answer_audio_ms"] = timings.speech_end_to_answer_audio_ms
    payload["audio_format"] = audio_format
    payload["preroll_bytes"] = preroll_bytes
    return payload
