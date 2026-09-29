#!/usr/bin/env python3
"""Tunes the D-06 global speaker-id threshold on real record-mode turns
(11-CONTEXT.md D-06, orchestrator clarification 1). Record mode already
writes one `speaker.result` event per turn into `events.jsonl`, carrying
each enrolled member's cosine score -- this script needs no model, no
database, and no second pass over audio. It only reads what
`speaker_id.turn_gate.evaluate_turn_speaker` already wrote (11-04-SUMMARY.md).

Two subcommands. `label` walks a time window of sessions under
`--sessions` and writes `{session_id: member_id or "other"}` into a labels
file (default `/data/speaker-labels.json`, under the data root Docker/Helm
already ignore) -- the labels file holds session ids and member ids only,
nothing else (T-11-31). `tune` turns those labels into one recommended
global threshold, built genuine/impostor trial by trial from each labeled
session's own scores, and scored with the same `equal_error_rate`/
`threshold_at_far` functions the Phase 11 spike used
(`atlas.speaker_id.evaluation`, 11-05-SUMMARY.md).

Neither subcommand writes to a session directory (T-11-32) -- only to the
labels file `label` names. The printed `recommended_threshold` is written
into `speaker_id.threshold` by hand; this script writes nothing back into
config.

`atlas` is not an installed package -- like every script here, this needs
`PYTHONPATH=src`. Run inside the application container, against the real
session data:

    docker compose exec app env PYTHONPATH=src .venv/bin/python \\
        scripts/tune_speaker_threshold.py label \\
        --sessions /data/sessions --labels /data/speaker-labels.json \\
        --since 2026-09-01T00:00:00 --until 2026-09-02T00:00:00 \\
        --speaker-id 1

    docker compose exec app env PYTHONPATH=src .venv/bin/python \\
        scripts/tune_speaker_threshold.py tune \\
        --sessions /data/sessions --labels /data/speaker-labels.json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# `atlas` is not on `sys.path` without `PYTHONPATH=src` -- every `atlas.*`
# import stays inside the function that needs it (mirroring
# `scripts/speaker_spike.py`'s own native-import discipline) so `--help`
# and any caller that never reaches real session data need no `PYTHONPATH`
# at all.

DEFAULT_LABELS_PATH = "/data/speaker-labels.json"
MIN_TRIALS = 10


class TuningError(RuntimeError):
    """Raised when the labeled corpus cannot produce a tuned threshold --
    the message names exactly what is short, for `main`'s exit-1 path."""


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def iter_sessions(root: "Path | str", since: "datetime | None" = None, until: "datetime | None" = None) -> "list[Path]":
    """Every session directory under `root` whose name
    `atlas.session.retention.parse_session_timestamp` parses, within
    `[since, until]` inclusive. A directory whose name does not parse is
    skipped -- never opened, never read (T-11-32)."""
    from atlas.session.retention import parse_session_timestamp

    root_path = Path(root)
    sessions: list[Path] = []
    if not root_path.is_dir():
        return sessions
    for entry in sorted(root_path.iterdir()):
        if not entry.is_dir():
            continue
        timestamp = parse_session_timestamp(entry.name)
        if timestamp is None:
            continue
        if since is not None and timestamp < since:
            continue
        if until is not None and timestamp > until:
            continue
        sessions.append(entry)
    return sessions


def read_speaker_result(directory: "Path | str") -> "dict[str, Any] | None":
    """The one `speaker.result` event in `directory`'s `events.jsonl`, or
    `None` when the file is missing, unreadable, or holds no such event."""
    from atlas.session.recorder import EVENTS_FILENAME

    events_path = Path(directory) / EVENTS_FILENAME
    try:
        text = events_path.read_text(encoding="utf-8")
    except OSError:
        return None
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "speaker.result":
            return event
    return None


def _load_labels(path: "Path | str") -> "dict[str, Any]":
    labels_path = Path(path)
    if not labels_path.is_file():
        return {}
    return json.loads(labels_path.read_text(encoding="utf-8"))


def _save_labels(path: "Path | str", labels: "dict[str, Any]") -> None:
    labels_path = Path(path)
    labels_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.write_text(json.dumps(labels, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def label_sessions(
    sessions_root: "Path | str",
    labels_path: "Path | str",
    *,
    since: datetime,
    until: datetime,
    speaker_id: "int | None" = None,
    other: bool = False,
    overwrite: bool = False,
) -> "dict[str, Any]":
    """Label every session in `[since, until]` that holds a `speaker.result`
    with `scores` -- a session with no scores is skipped, and an existing
    label stays unless `overwrite`. Exactly one of `speaker_id`/`other` must
    be given; the label value is `speaker_id` (an int) or the string
    `"other"`."""
    if other == (speaker_id is not None):
        raise ValueError("label_sessions: exactly one of speaker_id or other is required")
    label_value: "int | str" = "other" if other else speaker_id  # type: ignore[assignment]

    labels = _load_labels(labels_path)
    labeled: list[str] = []
    skipped_no_scores: list[str] = []
    skipped_existing: list[str] = []

    for directory in iter_sessions(sessions_root, since, until):
        session_id = directory.name
        result = read_speaker_result(directory)
        if result is None or not result.get("scores"):
            skipped_no_scores.append(session_id)
            continue
        if session_id in labels and not overwrite:
            skipped_existing.append(session_id)
            continue
        labels[session_id] = label_value
        labeled.append(session_id)

    _save_labels(labels_path, labels)
    return {
        "labeled": labeled,
        "skipped_no_scores": skipped_no_scores,
        "skipped_existing": skipped_existing,
    }


def collect_trials(
    labels: "dict[str, Any]", sessions_root: "Path | str", *, model_id: "str | None" = None
) -> "dict[str, Any]":
    """Genuine and impostor trials from every labeled session: for a session
    labeled member N, the score for N is one genuine trial and every other
    member's score is one impostor trial; for a session labeled `"other"`,
    every score is an impostor trial.

    Only one model id may appear among the labeled sessions unless
    `model_id` picks one -- raises `TuningError` naming both when more than
    one is present and `model_id` was not given.
    """
    session_results: "dict[str, dict[str, Any]]" = {}
    model_ids_seen: "set[str]" = set()
    for session_id, _label in labels.items():
        result = read_speaker_result(Path(sessions_root) / session_id)
        if result is None or not result.get("scores"):
            continue
        session_results[session_id] = result
        seen_model_id = result.get("model_id")
        if seen_model_id is not None:
            model_ids_seen.add(seen_model_id)

    resolved_model_id = model_id
    if resolved_model_id is None:
        if len(model_ids_seen) > 1:
            raise TuningError(
                "labeled sessions use more than one model id ("
                + ", ".join(sorted(model_ids_seen))
                + "); pass --model-id to pick one"
            )
        resolved_model_id = next(iter(model_ids_seen), None)

    genuine: list[float] = []
    impostor: list[float] = []
    sessions_used = 0
    for session_id, result in session_results.items():
        if resolved_model_id is not None and result.get("model_id") != resolved_model_id:
            continue
        label = labels[session_id]
        scores: "dict[str, Any]" = result["scores"]
        sessions_used += 1
        if label == "other":
            impostor.extend(float(v) for v in scores.values())
            continue
        member_key = str(label)
        for score_key, score_value in scores.items():
            if score_key == member_key:
                genuine.append(float(score_value))
            else:
                impostor.append(float(score_value))

    return {
        "model_id": resolved_model_id,
        "sessions_used": sessions_used,
        "genuine_trials": genuine,
        "impostor_trials": impostor,
    }


def tune_threshold(
    sessions_root: "Path | str",
    labels_path: "Path | str",
    *,
    model_id: "str | None" = None,
    target_far: float = 0.01,
) -> "dict[str, Any]":
    """The `tune` report: `equal_error_rate` and `threshold_at_far` over the
    trials `collect_trials` builds from the labels file. Raises
    `TuningError` naming the shortfall when either trial count is below
    `MIN_TRIALS`."""
    from atlas.speaker_id.evaluation import equal_error_rate, threshold_at_far

    labels = _load_labels(labels_path)
    trials = collect_trials(labels, sessions_root, model_id=model_id)
    genuine = trials["genuine_trials"]
    impostor = trials["impostor_trials"]
    if len(genuine) < MIN_TRIALS:
        raise TuningError(f"only {len(genuine)} genuine trial(s), need at least {MIN_TRIALS}")
    if len(impostor) < MIN_TRIALS:
        raise TuningError(f"only {len(impostor)} impostor trial(s), need at least {MIN_TRIALS}")

    eer, eer_threshold = equal_error_rate(genuine, impostor)
    far_threshold = threshold_at_far(impostor, target_far)

    return {
        "model_id": trials["model_id"],
        "sessions_used": trials["sessions_used"],
        "genuine_trials": len(genuine),
        "impostor_trials": len(impostor),
        "eer": eer,
        "eer_threshold": eer_threshold,
        "target_far": target_far,
        "far_threshold": far_threshold,
        "recommended_threshold": round(eer_threshold, 2),
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Turn labeled record-mode sessions into one recommended global "
            "speaker-id threshold (D-06)."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    label_parser = subparsers.add_parser("label", help="label sessions by who really spoke")
    label_parser.add_argument("--sessions", type=Path, required=True, help="session directory root")
    label_parser.add_argument("--labels", type=Path, default=Path(DEFAULT_LABELS_PATH))
    label_parser.add_argument("--since", required=True, help="ISO timestamp, inclusive")
    label_parser.add_argument("--until", required=True, help="ISO timestamp, inclusive")
    speaker_group = label_parser.add_mutually_exclusive_group(required=True)
    speaker_group.add_argument("--speaker-id", type=int, default=None)
    speaker_group.add_argument("--other", action="store_true")
    label_parser.add_argument("--overwrite", action="store_true")

    tune_parser = subparsers.add_parser("tune", help="tune the threshold from labeled sessions")
    tune_parser.add_argument("--sessions", type=Path, required=True, help="session directory root")
    tune_parser.add_argument("--labels", type=Path, default=Path(DEFAULT_LABELS_PATH))
    tune_parser.add_argument("--model-id", default=None)
    tune_parser.add_argument("--target-far", type=float, default=0.01)

    return parser


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "label":
        try:
            since = _parse_iso(args.since)
            until = _parse_iso(args.until)
        except ValueError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        summary = label_sessions(
            args.sessions,
            args.labels,
            since=since,
            until=until,
            speaker_id=args.speaker_id,
            other=args.other,
            overwrite=args.overwrite,
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0

    try:
        result = tune_threshold(
            args.sessions, args.labels, model_id=args.model_id, target_far=args.target_far
        )
    except TuningError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
