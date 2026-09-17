"""Unit tests for the measurement harness's pure parts: the WAV chunker,
the format validator, the pacing schedule, and the per-turn aggregator.

Deliberately narrow. The action text for this file is explicit that the
network-driving parts (the WebSocket and WebRTC turn drivers) stay out of
these tests entirely -- they need a live server, which is plan 01.1-08's
job, not this suite's. Every WAV here is synthesized into `tmp_path` with
the stdlib `wave` module, so these tests need neither the real recorded
fixture nor a network, and every assertion below is against the computed
schedule or the arithmetic itself -- never against real elapsed time, which
would be flaky on a loaded machine and prove less than the arithmetic does.

`scripts/measure_turns.py` is loaded by file path rather than package
import: `scripts/` is not on `pythonpath` (only `src`/`mcp` are, per
`pyproject.toml`), and adding it there is out of this plan's stated file
scope.
"""

from __future__ import annotations

import importlib.util
import sys
import wave
from pathlib import Path

import pytest

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_turns.py"
_spec = importlib.util.spec_from_file_location("measure_turns", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
measure_turns = importlib.util.module_from_spec(_spec)
sys.modules["measure_turns"] = measure_turns
_spec.loader.exec_module(measure_turns)


def _write_wav(
    path: Path,
    *,
    nchannels: int = 1,
    sampwidth: int = 2,
    framerate: int = 16000,
    nframes: int = 300,
) -> None:
    """Write `nframes` frames of silence at the given format -- the content
    is never inspected by anything under test here, only the shape is.
    """
    frame = b"\x00" * sampwidth * nchannels
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(nchannels)
        wav_file.setsampwidth(sampwidth)
        wav_file.setframerate(framerate)
        wav_file.writeframes(frame * nframes)


# --- read_wav_chunks ---------------------------------------------------


def test_read_wav_chunks_yields_128_sample_chunks_with_final_partial(tmp_path):
    path = tmp_path / "one.wav"
    _write_wav(path, nframes=300)  # 2 full chunks of 128 + a 44-frame remainder

    chunks = measure_turns.read_wav_chunks(path)

    assert len(chunks) == 3
    assert len(chunks[0]) == 128 * 2  # 256 bytes: 128 frames, 16-bit mono
    assert len(chunks[1]) == 128 * 2
    assert len(chunks[2]) == (300 - 256) * 2  # the final, shorter chunk


def test_read_wav_chunks_exact_multiple_has_no_partial_chunk(tmp_path):
    path = tmp_path / "exact.wav"
    _write_wav(path, nframes=256)  # exactly two 128-frame chunks

    chunks = measure_turns.read_wav_chunks(path)

    assert len(chunks) == 2
    assert all(len(chunk) == 256 for chunk in chunks)


def test_read_wav_chunks_rejects_wrong_format_before_any_chunk_is_read(tmp_path):
    path = tmp_path / "wrong-rate.wav"
    _write_wav(path, framerate=8000, nframes=128)

    with pytest.raises(measure_turns.WavFormatError):
        measure_turns.read_wav_chunks(path)


def test_read_wav_chunks_raises_a_clear_error_for_a_missing_fixture(tmp_path):
    missing = tmp_path / "does_not_exist.wav"

    with pytest.raises(FileNotFoundError):
        measure_turns.read_wav_chunks(missing)


# --- validate_wav_format -------------------------------------------------


def test_validate_wav_format_accepts_the_expected_shape():
    measure_turns.validate_wav_format(1, 2, 16000)  # must not raise


def test_validate_wav_format_names_found_and_expected_on_wrong_framerate():
    with pytest.raises(measure_turns.WavFormatError) as excinfo:
        measure_turns.validate_wav_format(1, 2, 8000)

    message = str(excinfo.value)
    assert "8000" in message
    assert "16000" in message


def test_validate_wav_format_rejects_stereo():
    with pytest.raises(measure_turns.WavFormatError) as excinfo:
        measure_turns.validate_wav_format(2, 2, 16000)

    message = str(excinfo.value)
    assert "2 channel" in message
    assert "mono" in message


def test_validate_wav_format_rejects_wrong_sample_width():
    with pytest.raises(measure_turns.WavFormatError) as excinfo:
        measure_turns.validate_wav_format(1, 1, 16000)

    message = str(excinfo.value)
    assert "1-byte" in message
    assert "16-bit" in message


def test_validate_wav_format_names_every_problem_at_once():
    with pytest.raises(measure_turns.WavFormatError) as excinfo:
        measure_turns.validate_wav_format(2, 1, 8000)

    message = str(excinfo.value)
    assert "channel" in message
    assert "byte" in message
    assert "Hz" in message


# --- pacing_schedule ------------------------------------------------------


def test_pacing_schedule_sums_to_the_recordings_real_duration(tmp_path):
    path = tmp_path / "one.wav"
    _write_wav(path, nframes=300)
    chunks = measure_turns.read_wav_chunks(path)

    schedule = measure_turns.pacing_schedule(chunks)

    assert len(schedule) == len(chunks)
    assert sum(schedule) == pytest.approx(300 / 16000, rel=1e-9)


def test_pacing_schedule_gives_each_full_chunk_the_same_duration(tmp_path):
    path = tmp_path / "exact.wav"
    _write_wav(path, nframes=256)
    chunks = measure_turns.read_wav_chunks(path)

    schedule = measure_turns.pacing_schedule(chunks)

    assert schedule[0] == pytest.approx(128 / 16000, rel=1e-9)
    assert schedule[0] == pytest.approx(schedule[1], rel=1e-9)


# --- aggregate_turn_events --------------------------------------------------


def _event(stt_final_ms: float, first_audio_ms: float | None, answer_audio_ms: float | None) -> dict:
    return {
        "stage_durations_ms": {"stt_final_at": stt_final_ms, "answer_audio_at": answer_audio_ms},
        "end_of_speech_to_first_audio_ms": first_audio_ms,
        "end_of_speech_to_answer_audio_ms": answer_audio_ms,
    }


def test_aggregate_turn_events_returns_median_per_stage():
    events = [
        _event(100.0, 500.0, 900.0),
        _event(200.0, 700.0, 1100.0),
        _event(300.0, 600.0, 1000.0),
    ]

    result = measure_turns.aggregate_turn_events(events)

    assert result["stt_final_at"].median == 200.0
    assert result["end_of_speech_to_first_audio_ms"].median == 600.0
    assert result["end_of_speech_to_answer_audio_ms"].median == 1000.0


def test_aggregate_turn_events_reports_min_max_and_reached_count():
    events = [_event(100.0, 500.0, 900.0), _event(200.0, 700.0, 1100.0), _event(300.0, 600.0, 1000.0)]

    result = measure_turns.aggregate_turn_events(events)

    agg = result["end_of_speech_to_first_audio_ms"]
    assert agg.minimum == 500.0
    assert agg.maximum == 700.0
    assert agg.reached == 3
    assert agg.total == 3


def test_aggregate_turn_events_returns_none_for_a_stage_no_turn_reached():
    events = [_event(100.0, 500.0, None), _event(200.0, 600.0, None)]

    result = measure_turns.aggregate_turn_events(events)

    agg = result["end_of_speech_to_answer_audio_ms"]
    assert agg.median is None
    assert agg.minimum is None
    assert agg.maximum is None
    assert agg.reached == 0
    assert agg.total == 2


def test_aggregate_turn_events_medians_only_over_turns_that_reached_a_stage():
    events = [_event(100.0, 500.0, 900.0), _event(200.0, 600.0, None), _event(300.0, 550.0, 950.0)]

    result = measure_turns.aggregate_turn_events(events)

    agg = result["end_of_speech_to_answer_audio_ms"]
    assert agg.median == 925.0  # median of [900.0, 950.0] -- the 600ms-only turn excluded, not zeroed
    assert agg.reached == 2
    assert agg.total == 3


def test_aggregate_turn_events_of_an_empty_run_is_empty():
    assert measure_turns.aggregate_turn_events([]) == {}


# --- CLI argument validation -------------------------------------------------


def test_cli_help_exits_zero_and_lists_all_four_arguments(capsys):
    parser = measure_turns.build_arg_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--help"])

    assert excinfo.value.code == 0
    help_text = capsys.readouterr().out
    for flag in ("--transport", "--turns", "--wav", "--url"):
        assert flag in help_text


def test_cli_rejects_an_unrecognized_transport(capsys):
    parser = measure_turns.build_arg_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--transport", "carrier-pigeon"])

    assert excinfo.value.code != 0
    stderr = capsys.readouterr().err
    assert "websocket" in stderr
    assert "webrtc" in stderr


def test_cli_rejects_fewer_than_five_turns(capsys):
    parser = measure_turns.build_arg_parser()

    with pytest.raises(SystemExit) as excinfo:
        parser.parse_args(["--transport", "websocket", "--turns", "2"])

    assert excinfo.value.code != 0
    stderr = capsys.readouterr().err
    assert "5" in stderr or "five" in stderr.lower()


def test_cli_accepts_the_five_turn_floor_exactly():
    parser = measure_turns.build_arg_parser()

    args = parser.parse_args(["--transport", "websocket", "--turns", "5"])

    assert args.turns == 5


def test_cli_defaults_turns_to_five_when_omitted():
    parser = measure_turns.build_arg_parser()

    args = parser.parse_args(["--transport", "webrtc"])

    assert args.turns == 5
