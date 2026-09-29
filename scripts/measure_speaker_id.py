#!/usr/bin/env python3
"""Measures D-16's own acceptance number over real edge turns: `speaker_id_ms`
after `vad_end` must have p95 under 100 ms (11-CONTEXT.md D-16). Reads every
session's `timing.json` under `--sessions`, the exact field
`speaker_id.turn_gate.evaluate_turn_speaker` already writes there
(11-04-SUMMARY.md) -- this script measures nothing itself and takes no
second pass over audio.

`scripts/measure_turns.py` (VOICE-02) is the model this follows: a stage's
`p50`/`p95`/`max` are `None` -- never a fabricated zero -- when no turn in
the run reached it, and the number this prints is what an acceptance claim
is made from, never one taken on synthetic or fixture audio.

`atlas` is not an installed package -- like every script here, this needs
`PYTHONPATH=src`. Run inside the application container, against the real
session data:

    docker compose exec app env PYTHONPATH=src .venv/bin/python \\
        scripts/measure_speaker_id.py --sessions /data/sessions
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

# `atlas` is not on `sys.path` without `PYTHONPATH=src` -- every `atlas.*`
# import stays inside the function that needs it, the same discipline
# `scripts/tune_speaker_threshold.py` uses, so `--help` needs no
# `PYTHONPATH` at all.

_DEFAULT_BUDGET_MS = 100.0
_DEFAULT_MIN_TURNS = 20


def _parse_iso(value: str) -> datetime:
    parsed = datetime.fromisoformat(value)
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def iter_sessions(root: "Path | str", since: "datetime | None" = None) -> "list[Path]":
    """The same directory rule `scripts/tune_speaker_threshold.py` uses --
    copied, not imported, since neither script is a package
    (11-09-PLAN.md's own read_first note)."""
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
        sessions.append(entry)
    return sessions


def _read_timing(directory: "Path | str") -> "dict[str, Any] | None":
    from atlas.session.recorder import TIMING_FILENAME

    timing_path = Path(directory) / TIMING_FILENAME
    try:
        return json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def read_speaker_id_ms(directory: "Path | str") -> "float | None":
    """`speaker_id_ms` from one session's `timing.json`, or `None` for a
    missing/unreadable file or an unmeasured turn -- never a fabricated
    zero."""
    payload = _read_timing(directory)
    if payload is None:
        return None
    value = payload.get("speaker_id_ms")
    return float(value) if value is not None else None


def read_speaker_gate_wait_ms(directory: "Path | str") -> "float | None":
    """`speaker_gate_wait_ms` from one session's `timing.json`, or `None`
    for a missing/unreadable file or an unmeasured turn."""
    payload = _read_timing(directory)
    if payload is None:
        return None
    value = payload.get("speaker_gate_wait_ms")
    return float(value) if value is not None else None


def measure_speaker_id(
    sessions_root: "Path | str",
    *,
    since: "datetime | None" = None,
    budget_ms: float = _DEFAULT_BUDGET_MS,
    min_turns: int = _DEFAULT_MIN_TURNS,
) -> "dict[str, Any]":
    """The report `main` prints. Raises nothing: an insufficient-turns
    condition is reported in `turns`/`verdict` (`None` when short), and
    `main` is what turns that into exit code 2."""
    from atlas.speaker_id.evaluation import percentile

    sessions = iter_sessions(sessions_root, since)
    speaker_id_values = [v for v in (read_speaker_id_ms(d) for d in sessions) if v is not None]
    gate_wait_values = [v for v in (read_speaker_gate_wait_ms(d) for d in sessions) if v is not None]

    turns = len(speaker_id_values)
    if turns == 0:
        p50_ms = p95_ms = max_ms = None
    else:
        p50_ms = percentile(speaker_id_values, 50)
        p95_ms = percentile(speaker_id_values, 95)
        max_ms = max(speaker_id_values)
    gate_wait_p95_ms = percentile(gate_wait_values, 95) if gate_wait_values else None

    verdict = None
    if turns >= min_turns and p95_ms is not None:
        verdict = "PASS" if p95_ms < budget_ms else "FAIL"

    return {
        "turns": turns,
        "p50_ms": p50_ms,
        "p95_ms": p95_ms,
        "max_ms": max_ms,
        "gate_wait_p95_ms": gate_wait_p95_ms,
        "budget_ms": budget_ms,
        "verdict": verdict,
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure speaker_id_ms over real edge turns against the D-16 100ms p95 budget."
        )
    )
    parser.add_argument("--sessions", type=Path, required=True, help="session directory root")
    parser.add_argument("--since", default=None, help="ISO timestamp; older sessions are excluded")
    parser.add_argument("--budget-ms", type=float, default=_DEFAULT_BUDGET_MS)
    parser.add_argument("--min-turns", type=int, default=_DEFAULT_MIN_TURNS)
    return parser


def main(argv: "Sequence[str] | None" = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    since = _parse_iso(args.since) if args.since else None
    report = measure_speaker_id(
        args.sessions, since=since, budget_ms=args.budget_ms, min_turns=args.min_turns
    )
    print(json.dumps(report, indent=2, sort_keys=True))

    if report["turns"] < args.min_turns:
        print(
            f"error: only {report['turns']} turn(s) carry a speaker_id_ms measurement, "
            f"need at least {args.min_turns}",
            file=sys.stderr,
        )
        return 2
    return 0 if report["verdict"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
