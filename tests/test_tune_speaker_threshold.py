"""Tests for `scripts/tune_speaker_threshold.py` (11-09-PLAN.md, D-06).

Loaded by file path, the same pattern `tests/test_score_wake_engines.py`
uses (lines 1-40): `scripts/` carries no package `__init__.py` and is not on
`pythonpath` (only `src`/`mcp`/`edge/src` are, per `pyproject.toml`).

Sessions are written by a real `SessionRecorder` (11-04-SUMMARY.md's own
`speaker.result` event shape), never a hand-rolled directory -- proving this
script reads the exact artifact plan 11-04 writes, not an assumption about
its shape.
"""

from __future__ import annotations

import importlib.util
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "tune_speaker_threshold.py"
_spec = importlib.util.spec_from_file_location("tune_speaker_threshold", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
tune_speaker_threshold = importlib.util.module_from_spec(_spec)
sys.modules["tune_speaker_threshold"] = tune_speaker_threshold
_spec.loader.exec_module(tune_speaker_threshold)

from atlas.config import SessionConfig
from atlas.session.recorder import SessionRecorder
from atlas.timing import TurnTimings

_MODEL_ID = "3dspeaker_speech_campplus_sv_en_voxceleb_16k"
_OTHER_MODEL_ID = "nemo_en_titanet_small"


def _write_session(
    root: Path, *, scores: "dict[str, float]", model_id: str = _MODEL_ID, with_scores: bool = True
) -> str:
    """Write one real session directory via `SessionRecorder`, carrying a
    `speaker.result` event -- `with_scores=False` omits `scores` entirely,
    the shape a mode-off/no-tracker turn's own event has."""
    timings = TurnTimings()
    recorder = SessionRecorder(SessionConfig(dir=str(root), record_audio=False), timings)
    event: dict[str, object] = {
        "type": "speaker.result",
        "status": "unknown",
        "best_speaker_id": None,
        "model_id": model_id,
        "effective_mode": "record",
    }
    if with_scores:
        event["scores"] = scores
    recorder.record_event(event)
    recorder.close(timings)
    return recorder.directory.name


@pytest.fixture
def now() -> datetime:
    return datetime.now(timezone.utc)


class TestLabelSessions:
    def test_labels_sessions_in_window_with_scores(self, tmp_path: Path, now: datetime) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        labels_path = tmp_path / "labels.json"

        session_id = _write_session(sessions_root, scores={"1": 0.8, "2": 0.2})

        summary = tune_speaker_threshold.label_sessions(
            sessions_root,
            labels_path,
            since=now - timedelta(hours=1),
            until=now + timedelta(hours=1),
            speaker_id=1,
        )

        assert summary["labeled"] == [session_id]
        assert summary["skipped_no_scores"] == []
        labels = tune_speaker_threshold._load_labels(labels_path)
        assert labels[session_id] == 1

    def test_skips_a_session_with_no_scores(self, tmp_path: Path, now: datetime) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        labels_path = tmp_path / "labels.json"

        session_id = _write_session(sessions_root, scores={}, with_scores=False)

        summary = tune_speaker_threshold.label_sessions(
            sessions_root,
            labels_path,
            since=now - timedelta(hours=1),
            until=now + timedelta(hours=1),
            speaker_id=1,
        )

        assert summary["labeled"] == []
        assert summary["skipped_no_scores"] == [session_id]

    def test_keeps_an_existing_label_unless_overwrite(self, tmp_path: Path, now: datetime) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        labels_path = tmp_path / "labels.json"
        session_id = _write_session(sessions_root, scores={"1": 0.8, "2": 0.2})

        tune_speaker_threshold.label_sessions(
            sessions_root, labels_path, since=now - timedelta(hours=1), until=now + timedelta(hours=1), speaker_id=1
        )
        # Relabel as "other" without --overwrite: the original label 1 stays.
        tune_speaker_threshold.label_sessions(
            sessions_root, labels_path, since=now - timedelta(hours=1), until=now + timedelta(hours=1), other=True
        )
        labels = tune_speaker_threshold._load_labels(labels_path)
        assert labels[session_id] == 1

        # With --overwrite, the new label replaces the old one.
        tune_speaker_threshold.label_sessions(
            sessions_root,
            labels_path,
            since=now - timedelta(hours=1),
            until=now + timedelta(hours=1),
            other=True,
            overwrite=True,
        )
        labels = tune_speaker_threshold._load_labels(labels_path)
        assert labels[session_id] == "other"

    def test_a_directory_whose_name_does_not_parse_is_never_read(self, tmp_path: Path, now: datetime) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        labels_path = tmp_path / "labels.json"

        bogus = sessions_root / "not-a-session-directory"
        bogus.mkdir()
        # No events.jsonl at all inside it -- if this script ever tried to
        # read it, `read_speaker_result` would raise or return None loudly;
        # either way it must never appear in any summary bucket.
        session_id = _write_session(sessions_root, scores={"1": 0.8, "2": 0.2})

        summary = tune_speaker_threshold.label_sessions(
            sessions_root, labels_path, since=now - timedelta(hours=1), until=now + timedelta(hours=1), speaker_id=1
        )

        assert summary["labeled"] == [session_id]
        assert "not-a-session-directory" not in summary["labeled"]
        assert "not-a-session-directory" not in summary["skipped_no_scores"]


class TestTuneThreshold:
    def test_returns_trial_counts_and_a_threshold_between_the_groups(self, tmp_path: Path, now: datetime) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        labels_path = tmp_path / "labels.json"

        labels: dict[str, object] = {}
        # Member 1's genuine scores near 0.8, cross scores against member 2
        # near 0.2.
        for _ in range(12):
            sid = _write_session(sessions_root, scores={"1": 0.8, "2": 0.2})
            labels[sid] = 1
        # Member 2's own genuine sessions, near 0.8 for member 2 and 0.2 for
        # member 1 -- gives member 1 enough impostor trials too.
        for _ in range(12):
            sid = _write_session(sessions_root, scores={"1": 0.2, "2": 0.8})
            labels[sid] = 2
        # "other" sessions: every score is an impostor trial, near 0.1.
        for _ in range(12):
            sid = _write_session(sessions_root, scores={"1": 0.1, "2": 0.1})
            labels[sid] = "other"

        tune_speaker_threshold._save_labels(labels_path, labels)

        result = tune_speaker_threshold.tune_threshold(sessions_root, labels_path)

        assert result["genuine_trials"] == 24
        assert result["impostor_trials"] == 48
        assert result["eer"] == pytest.approx(0.0)
        assert 0.2 < result["recommended_threshold"] < 0.8
        assert result["model_id"] == _MODEL_ID
        assert result["sessions_used"] == 36

    def test_two_model_ids_with_no_model_id_flag_exits_naming_both(self, tmp_path: Path, now: datetime) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        labels_path = tmp_path / "labels.json"

        labels: dict[str, object] = {}
        sid_a = _write_session(sessions_root, scores={"1": 0.8, "2": 0.2}, model_id=_MODEL_ID)
        labels[sid_a] = 1
        sid_b = _write_session(sessions_root, scores={"1": 0.8, "2": 0.2}, model_id=_OTHER_MODEL_ID)
        labels[sid_b] = 1
        tune_speaker_threshold._save_labels(labels_path, labels)

        with pytest.raises(tune_speaker_threshold.TuningError) as exc_info:
            tune_speaker_threshold.tune_threshold(sessions_root, labels_path)

        assert _MODEL_ID in str(exc_info.value)
        assert _OTHER_MODEL_ID in str(exc_info.value)

    def test_nine_genuine_trials_exits_naming_the_shortfall(self, tmp_path: Path, now: datetime) -> None:
        sessions_root = tmp_path / "sessions"
        sessions_root.mkdir()
        labels_path = tmp_path / "labels.json"

        labels: dict[str, object] = {}
        for _ in range(9):
            sid = _write_session(sessions_root, scores={"1": 0.8, "2": 0.2})
            labels[sid] = 1
        for _ in range(12):
            sid = _write_session(sessions_root, scores={"1": 0.1, "2": 0.1})
            labels[sid] = "other"
        tune_speaker_threshold._save_labels(labels_path, labels)

        with pytest.raises(tune_speaker_threshold.TuningError) as exc_info:
            tune_speaker_threshold.tune_threshold(sessions_root, labels_path)

        assert "9" in str(exc_info.value)
        assert "genuine" in str(exc_info.value)


class TestCli:
    def test_help_exits_0_and_lists_label_and_tune(self, capsys: pytest.CaptureFixture[str]) -> None:
        with pytest.raises(SystemExit) as exc_info:
            tune_speaker_threshold.main(["--help"])
        assert exc_info.value.code == 0
        out = capsys.readouterr().out
        assert "label" in out
        assert "tune" in out
