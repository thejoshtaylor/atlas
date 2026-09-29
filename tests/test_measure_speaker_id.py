"""Tests for `scripts/measure_speaker_id.py` (11-09-PLAN.md, D-16).

Loaded by file path, the same pattern `tests/test_score_wake_engines.py`
uses (lines 1-40): `scripts/` carries no package `__init__.py` and is not on
`pythonpath`.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_speaker_id.py"
_spec = importlib.util.spec_from_file_location("measure_speaker_id", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
measure_speaker_id_module = importlib.util.module_from_spec(_spec)
sys.modules["measure_speaker_id"] = measure_speaker_id_module
_spec.loader.exec_module(measure_speaker_id_module)

from atlas.session.recorder import TIMING_FILENAME


def _stamp(dt: datetime) -> str:
    return dt.strftime("%Y%m%dT%H%M%S%f") + "Z"


def _make_session(
    root: Path,
    *,
    turn_id: str,
    when: "datetime | None" = None,
    speaker_id_ms: "float | None" = None,
    speaker_gate_wait_ms: "float | None" = None,
    write_timing: bool = True,
) -> Path:
    """A session directory shaped exactly like `session/recorder.py`'s own
    naming (`<UTC stamp>-<turn_id>`), with a `timing.json` carrying
    `speaker_id_ms` -- never through a real `SessionRecorder`, since this
    script reads `timing.json` directly and a test needs full control over
    both the directory's timestamp and the field's value."""
    when = when or datetime.now(timezone.utc)
    directory = root / f"{_stamp(when)}-{turn_id}"
    directory.mkdir(parents=True)
    if write_timing:
        payload = {"speaker_id_ms": speaker_id_ms, "speaker_gate_wait_ms": speaker_gate_wait_ms}
        (directory / TIMING_FILENAME).write_text(json.dumps(payload), encoding="utf-8")
    return directory


class TestMeasureSpeakerId:
    def test_p95_by_linear_interpolation_all_under_budget_passes(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        values = list(np.linspace(10, 95, 20))
        for i, value in enumerate(values):
            _make_session(sessions_root, turn_id=f"t{i}", speaker_id_ms=float(value))

        report = measure_speaker_id_module.measure_speaker_id(sessions_root, budget_ms=100.0, min_turns=20)

        expected_p95 = float(np.percentile(np.asarray(values, dtype=np.float64), 95))
        assert report["turns"] == 20
        assert report["p95_ms"] == pytest.approx(expected_p95)
        assert report["verdict"] == "PASS"

    def test_p95_at_or_above_budget_fails(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        values = list(np.linspace(10, 105, 20))
        for i, value in enumerate(values):
            _make_session(sessions_root, turn_id=f"t{i}", speaker_id_ms=float(value))

        report = measure_speaker_id_module.measure_speaker_id(sessions_root, budget_ms=100.0, min_turns=20)

        expected_p95 = float(np.percentile(np.asarray(values, dtype=np.float64), 95))
        assert report["p95_ms"] == pytest.approx(expected_p95)
        assert expected_p95 >= 100.0
        assert report["verdict"] == "FAIL"

    def test_nineteen_turns_with_min_turns_20_reports_short(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        for i in range(19):
            _make_session(sessions_root, turn_id=f"t{i}", speaker_id_ms=50.0)

        report = measure_speaker_id_module.measure_speaker_id(sessions_root, budget_ms=100.0, min_turns=20)

        assert report["turns"] == 19
        assert report["verdict"] is None

    def test_null_and_unreadable_timing_are_not_counted(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        for i in range(20):
            _make_session(sessions_root, turn_id=f"good{i}", speaker_id_ms=50.0)
        # A turn with no measurement at all -- never a fabricated zero.
        _make_session(sessions_root, turn_id="null-turn", speaker_id_ms=None)
        # A directory with no timing.json at all -- unreadable.
        (sessions_root / f"{_stamp(datetime.now(timezone.utc))}-unreadable").mkdir()
        # A directory with a corrupt timing.json.
        corrupt = _make_session(sessions_root, turn_id="corrupt", speaker_id_ms=50.0)
        (corrupt / TIMING_FILENAME).write_text("not json", encoding="utf-8")

        report = measure_speaker_id_module.measure_speaker_id(sessions_root, budget_ms=100.0, min_turns=20)

        assert report["turns"] == 20

    def test_since_excludes_older_sessions(self, tmp_path: Path) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        now = datetime.now(timezone.utc)
        _make_session(sessions_root, turn_id="old", when=now - timedelta(days=2), speaker_id_ms=50.0)
        _make_session(sessions_root, turn_id="new", when=now, speaker_id_ms=60.0)

        report = measure_speaker_id_module.measure_speaker_id(
            sessions_root, since=now - timedelta(hours=1), budget_ms=100.0, min_turns=1
        )

        assert report["turns"] == 1
        assert report["p50_ms"] == pytest.approx(60.0)


class TestCli:
    def test_help_exits_0_and_lists_budget_ms(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            measure_speaker_id_module.main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "--budget-ms" in out

    def test_main_exits_2_when_short_of_min_turns(self, tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        for i in range(19):
            _make_session(sessions_root, turn_id=f"t{i}", speaker_id_ms=50.0)

        exit_code = measure_speaker_id_module.main(["--sessions", str(sessions_root), "--min-turns", "20"])

        assert exit_code == 2
        assert "19" in capsys.readouterr().err

    def test_main_exits_0_on_pass_and_1_on_fail(self, tmp_path: Path) -> None:
        passing_root = tmp_path / "passing"
        passing_root.mkdir()
        for i in range(20):
            _make_session(passing_root, turn_id=f"t{i}", speaker_id_ms=10.0)

        exit_code = measure_speaker_id_module.main(
            ["--sessions", str(passing_root), "--min-turns", "20", "--budget-ms", "100"]
        )
        assert exit_code == 0

        failing_root = tmp_path / "failing"
        failing_root.mkdir()
        for i in range(20):
            _make_session(failing_root, turn_id=f"f{i}", speaker_id_ms=500.0)

        exit_code = measure_speaker_id_module.main(
            ["--sessions", str(failing_root), "--min-turns", "20", "--budget-ms", "100"]
        )
        assert exit_code == 1
