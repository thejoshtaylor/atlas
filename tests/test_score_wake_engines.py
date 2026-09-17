"""Unit tests for the offline wake-engine scoring harness's pure parts
(DBG-04): the corpus-floor check, the scoring arithmetic, the
unavailable-engine outcome, and the byte-identical-input guarantee.

Deliberately narrow, following `tests/test_measure_harness.py`'s own
stated posture: no real corpus exists in this environment (no camera, no
`/models`, no Vosk model directory) -- Wave 0's own scaffold and
RESEARCH.md's Environment Availability table both say so. Every test here
is against the harness's arithmetic and wiring, over a synthetic labelled
set of *scores* or fixed byte strings, never against fabricated audio
scored as if it were a real room recording -- that prohibition is this
plan's own (`must_haves.prohibitions`), and it is why no test in this file
calls `decode_corpus`/`_decode_one` at all.

`scripts/score_wake_engines.py` is loaded by file path rather than package
import: `scripts/` is not on `pythonpath` (only `src`/`mcp` are, per
`pyproject.toml`), the same reason `test_measure_harness.py` loads
`measure_turns.py` this way.
"""

from __future__ import annotations

import importlib.util
import json
import os
import re
import sys
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "score_wake_engines.py"
_spec = importlib.util.spec_from_file_location("score_wake_engines", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
score_wake_engines = importlib.util.module_from_spec(_spec)
sys.modules["score_wake_engines"] = score_wake_engines
_spec.loader.exec_module(score_wake_engines)

from spire_voice.config import OpenWakeWordConfig
from spire_voice.wake.base import WakeError, WakeHit
from spire_voice.wake.openwakeword_engine import OpenWakeWordDetector


# --- fixtures --------------------------------------------------------------


def _recording(
    label: str,
    *,
    distance: str | None = None,
    speaker: str | None = None,
    duration_s: float | None = None,
    file: str = "some.alaw",
) -> "score_wake_engines.CorpusRecording":
    return score_wake_engines.CorpusRecording(
        file=file,
        label=label,
        distance=distance,
        speaker=speaker,
        position_index=None,
        global_index=0,
        timestamp="2026-09-17T00:00:00Z",
        duration_s=duration_s,
        source="test-camera",
    )


class _FakeDetector:
    """Records every chunk handed to `process()` -- the evidence the
    byte-identical-input test compares between two engines. Fires a
    `WakeHit` at `score` whenever `trigger` (a substring) appears in the
    chunk; `None` otherwise, so a test controls detections directly
    without needing real audio."""

    def __init__(self, *, score: float = 1.0, trigger: bytes | None = None) -> None:
        self.score = score
        self.trigger = trigger
        self.received: list[bytes] = []
        self.closed = False

    def process(self, chunk: bytes) -> WakeHit | None:
        self.received.append(chunk)
        if self.trigger is not None and self.trigger in chunk:
            return WakeHit(score=self.score)
        return None

    def reset(self) -> None:
        # Stateless by design (module docstring) -- nothing to clear.
        pass

    def close(self) -> None:
        self.closed = True


# --- check_corpus_floor (D-07) ----------------------------------------------


def test_check_corpus_floor_reports_short_for_a_corpus_below_twenty_positives():
    recordings = [_recording("positive", distance="close", speaker="a") for _ in range(5)]
    recordings.append(_recording("negative", duration_s=60.0))

    floor = score_wake_engines.check_corpus_floor(recordings)

    assert floor.positive_count == 5
    assert floor.meets_floor is False
    assert any("5 positive" in reason for reason in floor.reasons)


def test_check_corpus_floor_reports_met_for_a_corpus_at_the_floor():
    recordings = [
        _recording("positive", distance=("close" if i % 2 == 0 else "far"), speaker=("a" if i % 2 == 0 else "b"))
        for i in range(20)
    ]
    recordings.append(_recording("negative", duration_s=120.0))

    floor = score_wake_engines.check_corpus_floor(recordings)

    assert floor.positive_count == 20
    assert floor.distances == frozenset({"close", "far"})
    assert floor.speakers == frozenset({"a", "b"})
    assert floor.has_negative_run is True
    assert floor.meets_floor is True
    assert floor.reasons == ()


def test_check_corpus_floor_flags_a_missing_negative_run():
    recordings = [_recording("positive", distance="close", speaker="a") for _ in range(20)]
    recordings += [_recording("positive", distance="far", speaker="b") for _ in range(20)]

    floor = score_wake_engines.check_corpus_floor(recordings)

    assert floor.has_negative_run is False
    assert floor.meets_floor is False
    assert any("no negative run" in reason for reason in floor.reasons)


def test_check_corpus_floor_flags_a_single_distance_or_speaker():
    recordings = [_recording("positive", distance="close", speaker="a") for _ in range(25)]
    recordings.append(_recording("negative", duration_s=60.0))

    floor = score_wake_engines.check_corpus_floor(recordings)

    assert floor.meets_floor is False
    assert any("1 distance" in reason for reason in floor.reasons)
    assert any("1 speaker" in reason for reason in floor.reasons)


# --- scoring arithmetic (against scores, never audio) -----------------------


def test_detections_at_threshold_counts_scores_at_or_above():
    scores = [0.9, 0.5, None, 0.2, 0.6]

    assert score_wake_engines.detections_at_threshold(scores, 0.5) == 3
    assert score_wake_engines.detections_at_threshold(scores, 0.7) == 1
    assert score_wake_engines.detections_at_threshold(scores, 1.0) == 0


def test_false_accepts_per_hour_is_none_when_negative_duration_is_zero():
    assert score_wake_engines.false_accepts_per_hour([0.9, 0.9], 0.5, 0.0) is None


def test_false_accepts_per_hour_scales_by_negative_duration():
    # 2 false accepts over 30 minutes (1800s) -> 4/hour
    rate = score_wake_engines.false_accepts_per_hour([0.9, 0.9, 0.1], 0.5, 1800.0)

    assert rate == pytest.approx(4.0)


def test_threshold_sweep_produces_one_row_per_threshold():
    positive_scores = [0.9, 0.4, None]
    negative_scores = [0.2, 0.6]

    rows = score_wake_engines.threshold_sweep(positive_scores, negative_scores, 3600.0, thresholds=(0.3, 0.5, 0.8))

    assert [row.threshold for row in rows] == [0.3, 0.5, 0.8]
    assert rows[0].positive_detections == 2  # 0.9 and 0.4 both clear 0.3
    assert rows[1].positive_detections == 1  # only 0.9 clears 0.5
    assert rows[2].positive_detections == 1  # only 0.9 clears 0.8
    assert rows[0].positive_total == 3
    assert rows[1].false_accepts_per_hour == pytest.approx(1.0)  # one negative (0.6) clears 0.5, over 1 hour


# --- byte-identical input across engines (D-06) -----------------------------


def test_score_all_engines_hands_byte_identical_input_to_every_engine():
    recording = _recording("positive", distance="close", speaker="a")
    pcm16 = b"\x01\x02" * 4000  # arbitrary fixed bytes -- never claimed to be real speech
    fake_vosk = _FakeDetector(trigger=b"\x01\x02")
    fake_oww = _FakeDetector(trigger=b"\x01\x02")

    reports = score_wake_engines.score_all_engines(
        {"vosk": fake_vosk, "openwakeword": fake_oww}, [(recording, pcm16)]
    )

    assert fake_vosk.received == fake_oww.received
    assert b"".join(fake_vosk.received) == pcm16
    assert reports["vosk"].positive_scores == reports["openwakeword"].positive_scores


def test_score_engine_over_corpus_separates_positive_and_negative_scores():
    positive = _recording("positive", distance="close", speaker="a")
    negative = _recording("negative", duration_s=60.0)
    detector = _FakeDetector(score=0.8, trigger=b"HIT")
    pairs = [(positive, b"no-trigger-here"), (negative, b"contains-HIT-here")]

    report = score_wake_engines.score_engine_over_corpus(detector, "vosk", pairs)

    assert report.available is True
    assert report.positive_scores == (None,)
    assert report.negative_scores == (0.8,)


class _StatefulPendingHitDetector:
    """A detector whose one call to `process()` fires a *delayed* hit --
    the call after the one that saw the trigger byte, not the same call --
    unless `reset()` clears that pending state first.

    This is the shape `KaldiRecognizer.AcceptWaveform` actually has: an
    endpoint can land one call after the audio that produced it. A detector
    reused across recordings with no `reset()` between them can fire a hit
    attributed to the *next* recording's first bytes, from state the
    *previous* recording's trailing audio produced (WR-02's own report)."""

    def __init__(self) -> None:
        self._pending = False

    def process(self, chunk: bytes) -> WakeHit | None:
        if self._pending:
            self._pending = False
            return WakeHit(score=1.0)
        if b"TRIGGER" in chunk:
            self._pending = True
        return None

    def reset(self) -> None:
        self._pending = False

    def close(self) -> None:
        pass


def test_score_engine_over_corpus_resets_the_detector_between_recordings():
    """WR-02 fix: without `reset()` between recordings, the first
    recording's trailing `_pending` state (its own phrase-final endpoint
    landing just past the file boundary -- a missed positive) would carry
    into the second recording and fire on its very first chunk, registering
    a false accept against audio where the phrase was never spoken."""
    trigger_recording = _recording("positive", file="a.alaw")
    innocent_recording = _recording("negative", file="b.alaw", duration_s=10.0)
    pairs = [
        (trigger_recording, b"TRIGGER-with-no-endpoint-before-the-file-ends"),
        (innocent_recording, b"nothing-of-interest-in-this-recording"),
    ]

    report = score_wake_engines.score_engine_over_corpus(_StatefulPendingHitDetector(), "vosk", pairs)

    # The positive's own trailing state must not carry into the negative
    # that follows it: the positive itself is not detected within its own
    # bytes (this fake deliberately mimics an endpoint landing one call
    # later), and the negative must not inherit a hit from it.
    assert report.positive_scores == (None,)
    assert report.negative_scores == (None,)


# --- the unavailable-engine outcome (D-08) ----------------------------------


def test_build_engine_returns_engine_unavailable_with_named_cause_when_factory_raises_wake_error():
    def _boom():
        raise WakeError("model not present at /models/hey_spire.onnx")

    result = score_wake_engines.build_engine(_boom)

    assert isinstance(result, score_wake_engines.EngineUnavailable)
    assert "hey_spire.onnx" in result.reason


def test_build_engine_returns_the_constructed_detector_when_the_factory_succeeds():
    fake = _FakeDetector()

    result = score_wake_engines.build_engine(lambda: fake)

    assert result is fake


def test_score_engine_over_corpus_reports_an_unavailable_engine_without_dropping_it():
    unavailable = score_wake_engines.EngineUnavailable(reason="model not present at /models/hey_spire.onnx")

    report = score_wake_engines.score_engine_over_corpus(unavailable, "openwakeword", [])

    assert report.name == "openwakeword"
    assert report.available is False
    assert "hey_spire.onnx" in report.unavailable_reason


def test_score_all_engines_keeps_an_unavailable_engine_in_the_report():
    recording = _recording("positive", distance="close", speaker="a")
    available = _FakeDetector(trigger=b"HIT")
    unavailable = score_wake_engines.EngineUnavailable(reason="model not present")

    reports = score_wake_engines.score_all_engines(
        {"vosk": available, "openwakeword": unavailable}, [(recording, b"no-trigger")]
    )

    assert set(reports.keys()) == {"vosk", "openwakeword"}
    assert reports["openwakeword"].available is False
    assert reports["openwakeword"].unavailable_reason == "model not present"


# --- openWakeWord satisfies the same protocol Vosk does ---------------------


def test_openwakeword_detector_raises_wake_error_naming_the_missing_model_path(tmp_path):
    missing = tmp_path / "does_not_exist.onnx"
    config = OpenWakeWordConfig(model_path=str(missing))

    with pytest.raises(WakeError) as excinfo:
        OpenWakeWordDetector(config)

    assert str(missing) in str(excinfo.value)


def test_openwakeword_detector_satisfies_the_wake_detector_protocol(tmp_path, monkeypatch):
    import openwakeword

    class _FakeModel:
        def __init__(self, wakeword_models, inference_framework):
            assert inference_framework == "onnx"
            self.model_name = wakeword_models[0].split("/")[-1].rsplit(".", 1)[0]

        def predict(self, samples):
            return {self.model_name: 0.9}

    monkeypatch.setattr(openwakeword, "Model", _FakeModel)
    model_path = tmp_path / "hey_spire.onnx"
    model_path.write_bytes(b"not a real model -- only the path needs to exist for this test")
    config = OpenWakeWordConfig(model_path=str(model_path), threshold=0.5, trigger_frames=1)

    detector = OpenWakeWordDetector(config)
    assert hasattr(detector, "process")
    assert hasattr(detector, "close")

    frame = b"\x00\x00" * 1280  # one 80ms frame's worth of PCM16 samples -- exercises the wiring only
    hit = detector.process(frame)

    assert hit is not None
    assert hit.score == pytest.approx(0.9)
    detector.close()  # must not raise


# --- CLI --------------------------------------------------------------


def test_cli_help_lists_every_argument(capsys):
    parser = score_wake_engines.build_arg_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--config", "--corpus-dir", "--session-dir"):
        assert flag in help_text


def test_load_manifest_returns_empty_list_when_no_corpus_exists_yet(tmp_path):
    recordings = score_wake_engines.load_manifest(tmp_path / "no_such_corpus")

    assert recordings == []


# --- the durable report (plan 02-13, Task 1) --------------------------------

_REPO_ROOT = Path(__file__).resolve().parent.parent

# config.example.yaml's ${...} placeholders load_config must expand -- these
# tests drive main() against the real example config, so they need the same
# fictional values tests/test_config.py's own end-to-end case sets.
_CONFIG_ENV_VARS = ("XAI_API_KEY", "TAPO_USER", "TAPO_PASSWORD", "SPEAKER_ENSURE_URL", "HA_URL", "HA_TOKEN")


def _set_config_env_vars(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in _CONFIG_ENV_VARS:
        monkeypatch.setenv(name, "test-value")


def _engine_report(
    *, available: bool = True, unavailable_reason: str | None = None, sweep: tuple = ()
) -> "score_wake_engines.EngineReport":
    return score_wake_engines.EngineReport(
        name="test-engine",
        available=available,
        unavailable_reason=unavailable_reason,
        positive_scores=(0.9,) if available else (),
        negative_scores=(0.1,) if available else (),
        threshold_sweep=sweep,
    )


def _floor(*, meets_floor: bool = True, reasons: tuple[str, ...] = ()) -> "score_wake_engines.FloorReport":
    return score_wake_engines.FloorReport(
        positive_count=25,
        distances=frozenset({"close", "far"}),
        speakers=frozenset({"a", "b"}),
        has_negative_run=True,
        negative_duration_s=180.0,
        meets_floor=meets_floor,
        reasons=reasons,
    )


def test_validate_report_path_is_a_no_op_when_report_out_is_none():
    score_wake_engines.validate_report_path(None)  # must not raise


def test_validate_report_path_creates_the_parent_and_leaves_no_probe_file(tmp_path):
    report_out = tmp_path / "reports" / "run.json"

    score_wake_engines.validate_report_path(report_out)

    assert report_out.parent.is_dir()
    assert list(report_out.parent.iterdir()) == []


def test_validate_report_path_raises_report_path_error_when_the_parent_is_a_file(tmp_path):
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("x")

    with pytest.raises(score_wake_engines.ReportPathError):
        score_wake_engines.validate_report_path(not_a_dir / "run.json")


def test_classify_recommendation_below_floor_corpus_is_provisional():
    floor = _floor(meets_floor=False, reasons=("only 5 positive(s) recorded, floor is 20",))
    sweep = score_wake_engines.threshold_sweep([0.9], [0.1], 180.0)
    reports = {"vosk": _engine_report(sweep=sweep), "openwakeword": _engine_report(sweep=sweep)}

    classification, recommendation, reasons = score_wake_engines.classify_recommendation(floor, reports, "vosk")

    assert classification == "provisional"
    assert recommendation == "vosk"
    assert any("floor is 20" in reason for reason in reasons)


def test_classify_recommendation_single_engine_run_is_provisional():
    floor = _floor(meets_floor=True)
    reports = {
        "vosk": _engine_report(sweep=score_wake_engines.threshold_sweep([0.9], [0.1], 180.0)),
        "openwakeword": _engine_report(available=False, unavailable_reason="model not present at /models/x.onnx"),
    }

    classification, recommendation, reasons = score_wake_engines.classify_recommendation(floor, reports, "vosk")

    assert classification == "provisional"
    assert recommendation == "vosk"
    assert any("only one engine could run" in reason for reason in reasons)


def test_classify_recommendation_identical_sweeps_is_provisional():
    floor = _floor(meets_floor=True)
    sweep = score_wake_engines.threshold_sweep([0.9, 0.4], [0.1], 180.0)
    reports = {"vosk": _engine_report(sweep=sweep), "openwakeword": _engine_report(sweep=sweep)}

    classification, recommendation, reasons = score_wake_engines.classify_recommendation(floor, reports, "vosk")

    assert classification == "provisional"
    assert any("no separation" in reason for reason in reasons)


def test_classify_recommendation_two_engines_meeting_floor_with_separation_is_measured():
    floor = _floor(meets_floor=True)
    strong_sweep = score_wake_engines.threshold_sweep([0.9, 0.9, 0.9], [0.0, 0.0], 180.0)
    weak_sweep = score_wake_engines.threshold_sweep([0.9, None, None], [0.9, 0.9], 180.0)
    reports = {"vosk": _engine_report(sweep=strong_sweep), "openwakeword": _engine_report(sweep=weak_sweep)}

    classification, recommendation, reasons = score_wake_engines.classify_recommendation(floor, reports, "vosk")

    assert classification == "measured"
    assert recommendation == "vosk"
    assert reasons == ()


def test_build_report_includes_the_corpus_fingerprint(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    manifest = corpus_dir / score_wake_engines._MANIFEST_FILENAME
    manifest.write_text('{"file": "a.alaw", "label": "positive"}\n')
    floor = _floor(meets_floor=False, reasons=("only 1 positive(s) recorded, floor is 20",))

    report = score_wake_engines.build_report(
        corpus_dir=corpus_dir, floor=floor, reports={}, current_default="vosk"
    )

    assert report["corpus_dir"] == str(corpus_dir)
    assert report["positive_count"] == floor.positive_count
    assert report["negative_duration_s"] == floor.negative_duration_s
    assert report["manifest_digest"] == score_wake_engines._manifest_digest(corpus_dir)
    assert report["manifest_digest"] is not None
    assert report["classification"] == "provisional"


def test_build_report_manifest_digest_is_none_for_an_empty_corpus(tmp_path):
    corpus_dir = tmp_path / "empty_corpus"
    corpus_dir.mkdir()

    report = score_wake_engines.build_report(
        corpus_dir=corpus_dir, floor=_floor(meets_floor=False), reports={}, current_default="vosk"
    )

    assert report["manifest_digest"] is None


def test_build_report_unavailable_engine_is_a_named_row_not_an_absent_section(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    reports = {
        "vosk": _engine_report(sweep=score_wake_engines.threshold_sweep([0.9], [0.1], 180.0)),
        "openwakeword": _engine_report(available=False, unavailable_reason="model not present at /models/x.onnx"),
    }

    report = score_wake_engines.build_report(
        corpus_dir=corpus_dir, floor=_floor(meets_floor=True), reports=reports, current_default="vosk"
    )

    assert set(report["engines"]) == {"vosk", "openwakeword"}
    assert report["engines"]["openwakeword"]["available"] is False
    assert "/models/x.onnx" in report["engines"]["openwakeword"]["unavailable_reason"]


def test_written_report_contains_no_transcript_audio_or_rtsp_derived_content(tmp_path):
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    reports = {
        "vosk": _engine_report(sweep=score_wake_engines.threshold_sweep([0.9], [0.1], 180.0)),
        "openwakeword": _engine_report(available=False, unavailable_reason="model not present at /models/x.onnx"),
    }
    report_out = tmp_path / "report.json"

    score_wake_engines._write_report(
        report_out,
        score_wake_engines.build_report(
            corpus_dir=corpus_dir, floor=_floor(meets_floor=True), reports=reports, current_default="vosk"
        ),
    )
    text = report_out.read_text()

    assert "rtsp://" not in text
    assert "TAPO_USER" not in text
    assert "TAPO_PASSWORD" not in text
    assert str(corpus_dir) in text
    assert "/models/x.onnx" in text  # the one deliberately-named exception (D-08)


def test_main_without_report_out_behaves_as_before(tmp_path, capsys, monkeypatch):
    _set_config_env_vars(monkeypatch)
    empty_corpus = tmp_path / "empty_corpus"

    exit_code = score_wake_engines.main(["--corpus-dir", str(empty_corpus)])

    assert exit_code == 0
    assert "no corpus recorded yet" in capsys.readouterr().out


def test_main_with_report_out_writes_exactly_one_file_and_the_same_exit_code(tmp_path, monkeypatch):
    _set_config_env_vars(monkeypatch)
    empty_corpus = tmp_path / "empty_corpus"
    report_path = tmp_path / "out" / "report.json"

    without_flag = score_wake_engines.main(["--corpus-dir", str(empty_corpus)])
    with_flag = score_wake_engines.main(
        ["--corpus-dir", str(empty_corpus), "--report-out", str(report_path)]
    )

    assert without_flag == with_flag == 0
    assert report_path.exists()
    assert list(report_path.parent.iterdir()) == [report_path]
    written = json.loads(report_path.read_text())
    assert written["positive_count"] == 0


def test_main_with_an_unwritable_report_out_fails_before_any_engine_loads(tmp_path, capsys, monkeypatch):
    _set_config_env_vars(monkeypatch)
    corpus_dir = tmp_path / "corpus"
    corpus_dir.mkdir()
    # Points at a file that does not exist, so if validation were skipped
    # this would instead fail deep inside decode_corpus with a raw,
    # unhandled file-not-found error -- proving, by its absence, that
    # validation really does run first.
    (corpus_dir / score_wake_engines._MANIFEST_FILENAME).write_text(
        json.dumps(
            {
                "file": "does-not-exist.alaw",
                "label": "positive",
                "distance": "close",
                "speaker": "a",
                "position_index": 1,
                "global_index": 0,
                "timestamp": "2026-09-17T00:00:00Z",
                "duration_s": 2.0,
                "source": "test",
            }
        )
        + "\n"
    )
    not_a_dir = tmp_path / "not_a_dir"
    not_a_dir.write_text("x")
    report_path = not_a_dir / "report.json"

    exit_code = score_wake_engines.main(
        ["--corpus-dir", str(corpus_dir), "--report-out", str(report_path)]
    )

    assert exit_code == 1
    assert "error:" in capsys.readouterr().err


# --- the wake-default-evidence line (plan 02-13, Task 2) --------------------

_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"
_RUNBOOK_PATH = _REPO_ROOT / "docs" / "runbooks" / "wake-engine-corpus.md"


def test_real_config_has_exactly_one_wake_default_evidence_line():
    evidence = score_wake_engines.parse_wake_default_evidence(_CONFIG_PATH.read_text())

    assert evidence.classification in ("provisional", "measured")


def test_parse_wake_default_evidence_provisional_requires_nothing_further():
    evidence = score_wake_engines.parse_wake_default_evidence(
        "wake:\n  # wake-default-evidence: provisional\n"
    )

    assert evidence.classification == "provisional"
    assert evidence.date is None
    assert evidence.positives is None
    assert evidence.negative_duration_s is None


def test_parse_wake_default_evidence_measured_requires_date_positives_and_duration():
    text = (
        "wake:\n  # wake-default-evidence: measured date=2026-09-20 "
        "positives=24 negative_duration_s=185.3\n"
    )

    evidence = score_wake_engines.parse_wake_default_evidence(text)

    assert evidence.classification == "measured"
    assert evidence.date == "2026-09-20"
    assert evidence.positives == 24
    assert evidence.negative_duration_s == pytest.approx(185.3)


def test_parse_wake_default_evidence_measured_missing_fields_raises():
    with pytest.raises(ValueError):
        score_wake_engines.parse_wake_default_evidence("wake:\n  # wake-default-evidence: measured\n")


def test_parse_wake_default_evidence_rejects_zero_or_multiple_lines():
    with pytest.raises(ValueError):
        score_wake_engines.parse_wake_default_evidence("wake:\n  engine: vosk\n")
    with pytest.raises(ValueError):
        score_wake_engines.parse_wake_default_evidence(
            "wake:\n  # wake-default-evidence: provisional\n  # wake-default-evidence: provisional\n"
        )


def test_runbook_names_only_scripts_that_exist():
    text = _RUNBOOK_PATH.read_text()
    referenced = set(re.findall(r"scripts/[\w.\-]+\.(?:py|sh)", text))

    assert referenced, "expected the runbook to name at least one script"
    for rel_path in referenced:
        path = _REPO_ROOT / rel_path
        assert path.exists(), f"{rel_path} named in the runbook does not exist"
        if rel_path.endswith(".sh"):
            assert os.access(path, os.X_OK), f"{rel_path} is not executable"


def test_runbook_states_the_same_floor_score_wake_engines_enforces():
    text = _RUNBOOK_PATH.read_text()
    match = re.search(r"floor is \*\*(\d+)\*\*", text)

    assert match is not None, "expected the runbook to state the floor as **N** somewhere"
    assert int(match.group(1)) == score_wake_engines._FLOOR_POSITIVES


def test_runbook_names_the_two_closing_edits_and_the_coupling_test():
    text = _RUNBOOK_PATH.read_text()

    assert "wake-default-evidence" in text
    assert "DBG-04" in text
    assert "fails" in text.lower()
