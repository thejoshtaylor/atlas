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
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from spire_voice.session.recorder import EVENTS_FILENAME, TIMING_FILENAME

TIMELINE_FILENAME = "timeline.jsonl"

# Encounter order, matching `timing.py`'s own `_STAGE_ORDER` exactly -- not
# re-declared there and imported here, because `timing.py`'s own privacy
# boundary keeps it free of any dependency on this phase's session store.
_STAGE_LABELS = (
    "turn_started_at",
    "stt_socket_open_at",
    "first_partial_at",
    "stt_final_at",
    "brain_first_token_at",
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
