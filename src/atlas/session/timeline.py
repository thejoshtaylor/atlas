"""The merged per-turn timeline: `session/recorder.py`'s JSONL events and
`timing.py`'s named stage marks, interleaved in the order they actually
happened -- derived, and never the source of truth (D-13).

Mirrors `timing.py`'s derive-never-store shape exactly: `render_timeline`
computes its view fresh from the state handed to it on every call, and
this module holds no state of its own between calls. Deleting a rendered
timeline loses nothing -- that claim is demonstrable, not merely asserted,
because `regenerate_timeline` can rebuild one for an existing session
directory from exactly the files `SessionRecorder` wrote
(`events.jsonl`, `timing.json`). A reader who finds the timeline and the
JSONL disagreeing should believe the JSONL: the JSONL is
`SessionRecorder`'s append-as-it-happens log, and the timeline is a
rendering of it, never the other way around.

The merge orders by each entry's own timestamp -- an event's
`recorded_at` and a stage's own mark, both drawn from `TurnTimings`'
`time.monotonic()` clock domain, so comparing them is comparing one clock
to itself, never a re-sort against a second, disagreeing one. Ties are
broken by Python's stable `sorted()` alone: an event keeps its place
relative to another event recorded at the same instant, because it was
appended to the input list first, not because this module invented a
second tiebreak rule.

`preroll_offset_s` (plan 08-11, DBG-03) extends the same derive-never-store
discipline to one more quantity: how far `turn_started_at` sits into the
recorded audio, given the pre-roll byte count `SessionRecorder.close`
wrote into `timing.json`. It is a pure function of `timing_payload`, never
a value this module stores or caches, matching `render_timeline` above.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from atlas.audio.ring import bytes_per_ms
from atlas.session.recorder import EVENTS_FILENAME, TIMING_FILENAME
from atlas.transports.base import SourceFormat

TIMELINE_FILENAME = "timeline.jsonl"

# The closed set of encodings this codebase models exactly
# (`transports/base.py::SourceFormat`'s own docstring). Anything outside
# this set is a payload `preroll_offset_s` cannot convert exactly, never a
# rate it guesses at.
_MODELLED_ENCODINGS = ("pcm", "alaw")

# `timing.py`'s own `_STAGE_ORDER`, plus one legacy label
# (`brain_first_token_at`) kept for an old recording that still carries it
# -- not re-declared there and imported here, because `timing.py`'s own
# privacy boundary keeps it free of any dependency on this phase's session
# store. 260924-4iv (item e) renamed `brain_first_token_at` to
# `brain_first_round_at` and inserted `speech_end_at` after
# `first_partial_at`; the legacy label sits directly before the new one so
# an old `timing.json` (which carries `brain_first_token_at` and no
# `brain_first_round_at`) still renders its own stage, and a new one
# (which carries `brain_first_round_at` and never the old key) never
# renders both.
_STAGE_LABELS = (
    "turn_started_at",
    "stt_socket_open_at",
    "first_partial_at",
    "speech_end_at",
    "stt_final_at",
    "brain_first_token_at",
    "brain_first_round_at",
    "tool_rounds_done_at",
    "first_audio_at",
    "answer_audio_at",
)


def render_timeline(events: list[dict[str, Any]], timing_payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Merge `events` (already in the order they were written) with the
    named stage marks present in `timing_payload`, in ascending timestamp
    order.

    A stage absent from `timing_payload` (`None`, or missing outright) is
    skipped -- a stage the turn never reached leaves no entry, the same
    "never happened" fact `timing.py` itself preserves rather than a
    fabricated zero-duration entry.
    """
    entries: list[dict[str, Any]] = []
    for event in events:
        rest = {key: value for key, value in event.items() if key != "recorded_at"}
        entries.append({"ts": event.get("recorded_at"), "kind": "event", **rest})
    for label in _STAGE_LABELS:
        value = timing_payload.get(label)
        if value is not None:
            entries.append({"ts": value, "kind": "stage", "stage": label})
    return sorted(entries, key=lambda entry: entry["ts"])


def preroll_offset_s(timing_payload: dict[str, Any]) -> float:
    """How far into the recorded audio `turn_started_at` sits, in seconds
    (plan 08-11, DBG-03): `timing_payload["preroll_bytes"]` divided by the
    recording's own byte rate, `bytes_per_ms(SourceFormat(...)) * 1000.0`
    -- the same conversion that bounded `audio/ring.py::PrerollBuffer` in
    the first place, inverted here, so the bound and its inverse cannot
    drift.

    A derived value: this module stores nothing (the module docstring's
    own discipline), and this function reads `timing_payload` fresh on
    every call rather than caching anything.

    Returns `0.0` -- never raises, never guesses -- for every payload it
    cannot convert exactly: a missing, non-numeric, or negative
    `preroll_bytes`; a missing `audio_format`; a missing or non-positive
    `sample_rate`; a `channels` value that is not an integer of 1 or more
    (10-07-PLAN.md, D-09); or an `encoding` outside the closed set `{"pcm",
    "alaw"}` this codebase models exactly (`transports/base.py::
    SourceFormat`). `0.0` is the pre-fix behavior, correct for every path
    that never built a pre-roll buffer; a guessed byte rate would
    silently move a recording by the wrong amount (T-08-23).

    `channels` defaults to `1` when the key is absent -- a session
    recorded before this plan carries no such key, and the one-channel
    reading is exactly what it was recorded at.
    """
    preroll_bytes = timing_payload.get("preroll_bytes")
    if not isinstance(preroll_bytes, (int, float)) or isinstance(preroll_bytes, bool) or preroll_bytes < 0:
        return 0.0

    audio_format = timing_payload.get("audio_format")
    if not isinstance(audio_format, dict):
        return 0.0

    encoding = audio_format.get("encoding")
    sample_rate = audio_format.get("sample_rate")
    if encoding not in _MODELLED_ENCODINGS:
        return 0.0
    if not isinstance(sample_rate, (int, float)) or isinstance(sample_rate, bool) or sample_rate <= 0:
        return 0.0

    channels = audio_format.get("channels", 1)
    if isinstance(channels, bool) or not isinstance(channels, int) or channels < 1:
        return 0.0

    byte_rate = bytes_per_ms(SourceFormat(encoding, sample_rate, channels=channels)) * 1000.0
    return preroll_bytes / byte_rate


def write_timeline(directory: Path, timeline: list[dict[str, Any]]) -> Path:
    """Render `timeline` to `TIMELINE_FILENAME` under `directory`, one JSON
    object per line, deterministically (`sort_keys=True`) -- the same
    inputs always produce the same bytes, which is what makes
    `regenerate_timeline` provably byte-identical rather than merely
    plausible.
    """
    path = Path(directory) / TIMELINE_FILENAME
    lines = [json.dumps(entry, sort_keys=True) for entry in timeline]
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return path


def regenerate_timeline(directory: Path) -> Path:
    """Rebuild `TIMELINE_FILENAME` for an existing session directory from
    exactly the files `SessionRecorder` wrote -- proof that the timeline
    is regenerable, not merely a claim in a docstring.
    """
    directory = Path(directory)
    events = _read_jsonl(directory / EVENTS_FILENAME)
    timing_payload = json.loads((directory / TIMING_FILENAME).read_text(encoding="utf-8"))
    rendered = render_timeline(events, timing_payload)
    return write_timeline(directory, rendered)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [json.loads(line) for line in text.splitlines() if line.strip()]
