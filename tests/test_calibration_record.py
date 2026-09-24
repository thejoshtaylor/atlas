"""`calibration/record.py` (plan 02-10, Task 3): round-trip, the three
tolerated bad states `load` never raises for, the staleness boundary, the
exact serialized key set, and the URL-rejecting placement note.
"""

from __future__ import annotations

import dataclasses
import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from atlas.calibration.record import (
    SCHEMA_VERSION,
    CalibrationError,
    EchoCalibration,
)


def _make_record(**overrides) -> EchoCalibration:
    fields = dict(
        schema_version=SCHEMA_VERSION,
        probe_format_version=1,
        probe_seed=20260917,
        source="camera",
        delay_s=0.045,
        confidence=0.94,
        echo_level=0.12,
        gain=0.6,
        agc_verdict="absent",
        segment_levels=(0.11, 0.115, 0.12, 0.118),
        encoding="alaw",
        sample_rate=8000,
        channels=1,
        placement_note="camera on the kitchen shelf, room quiet",
        taken_at=datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc),
    )
    fields.update(overrides)
    return EchoCalibration(**fields)


def test_record_round_trips_through_save_and_load_with_every_field_equal(tmp_path):
    record = _make_record()
    path = tmp_path / "calibration" / "echo_path.json"
    record.save(path)

    loaded = EchoCalibration.load(path)

    assert loaded is not None
    assert dataclasses.asdict(loaded) == dataclasses.asdict(record)


def test_load_of_missing_file_returns_none(tmp_path):
    assert EchoCalibration.load(tmp_path / "does-not-exist.json") is None


def test_load_of_schema_version_mismatch_returns_none(tmp_path):
    record = _make_record()
    path = tmp_path / "echo_path.json"
    payload = record._to_json_dict()
    payload["schema_version"] = SCHEMA_VERSION + 1
    path.write_text(json.dumps(payload), encoding="utf-8")

    assert EchoCalibration.load(path) is None


def test_load_of_malformed_json_returns_none(tmp_path):
    path = tmp_path / "echo_path.json"
    path.write_text("{not valid json", encoding="utf-8")

    assert EchoCalibration.load(path) is None


def test_is_stale_true_past_the_window_false_inside_it():
    taken_at = datetime(2026, 9, 1, tzinfo=timezone.utc)
    record = _make_record(taken_at=taken_at)

    just_inside = taken_at + timedelta(days=7) - timedelta(seconds=1)
    exactly_at_boundary = taken_at + timedelta(days=7)
    just_past = taken_at + timedelta(days=7, seconds=1)

    assert record.is_stale(just_inside, max_age_days=7) is False
    assert record.is_stale(exactly_at_boundary, max_age_days=7) is False
    assert record.is_stale(just_past, max_age_days=7) is True


def test_is_stale_reads_the_records_own_timestamp_not_file_mtime(tmp_path):
    old_taken_at = datetime(2020, 1, 1, tzinfo=timezone.utc)
    record = _make_record(taken_at=old_taken_at)
    path = tmp_path / "echo_path.json"
    record.save(path)  # file mtime is "now", far newer than taken_at

    loaded = EchoCalibration.load(path)
    assert loaded.is_stale(datetime(2020, 1, 8, 0, 0, 1, tzinfo=timezone.utc), max_age_days=7) is True


def test_serialized_key_set_is_exactly_the_declared_field_set():
    record = _make_record()
    payload = record._to_json_dict()
    expected_keys = {f.name for f in dataclasses.fields(EchoCalibration)}
    assert set(payload.keys()) == expected_keys


def test_load_of_a_v1_file_with_no_echo_cancelled_key_defaults_it_to_false(tmp_path):
    record = _make_record()
    path = tmp_path / "echo_path.json"
    payload = record._to_json_dict()
    del payload["echo_cancelled"]  # exactly the shape a file written before this field existed has
    path.write_text(json.dumps(payload), encoding="utf-8")

    loaded = EchoCalibration.load(path)

    assert loaded is not None
    assert loaded.echo_cancelled is False


def test_placement_note_containing_a_url_scheme_is_rejected():
    with pytest.raises(CalibrationError):
        _make_record(placement_note="see rtsp://camera.invalid:554/stream1")


def test_placement_note_containing_userinfo_is_rejected():
    with pytest.raises(CalibrationError):
        _make_record(placement_note="credentials are admin:hunter2@camera.invalid")
