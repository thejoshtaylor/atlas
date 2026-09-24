"""`calibration/runner.py`'s `run_echo_calibration` (plan 02-11 Task 1).
Tasks 2 and 3 extend this file with the import-seam assertions their
callers -- `scripts/calibrate_echo_path.py` and `app.py`'s two routes --
must both satisfy, per this plan's own instruction to keep those
assertions here rather than in a new file, since the seam is what this
file is about.

Every case here is checkable with no camera: a loopback fake plays the role
of both the audio source and the speaker sink, echoing back whatever was
written to it with a delay and a gain the test itself injects, the same
discipline `tests/test_echo_path.py` already applies one layer down. That
the real camera's echo path is measurable at all remains this plan's own
live human check, not this file's.
"""

from __future__ import annotations

import ast
import asyncio
import dataclasses
import importlib.util
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import numpy as np
import pytest
from fastapi import HTTPException

import atlas.app as app_module

from atlas.audio.alaw import alaw_to_pcm16, pcm16_to_alaw
from atlas.calibration.record import EchoCalibration
from atlas.calibration.runner import (
    MAX_RECORD_FILE_SIZE_BYTES,
    CalibrationRunResult,
    CalibrationRunnerError,
    find_latest_calibration,
    run_echo_calibration,
)
from atlas.config import CalibrationConfig, CameraConfig, SpeakerConfig
from atlas.providers.tts_xai import SinkFormat
from atlas.providers.tts_piper import _resample_pcm16
from atlas.transports.base import SourceFormat

# `scripts/` is not on `pythonpath` (only `src`/`mcp` are, per
# `pyproject.toml`) -- loaded by file path, the same mechanism
# `tests/test_score_wake_engines.py` already uses for
# `score_wake_engines.py`.
_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "calibrate_echo_path.py"
_spec = importlib.util.spec_from_file_location("calibrate_echo_path", _SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
calibrate_echo_path = importlib.util.module_from_spec(_spec)
sys.modules["calibrate_echo_path"] = calibrate_echo_path
_spec.loader.exec_module(calibrate_echo_path)

SAMPLE_RATE = 8000
_CHUNK_BYTES = 64


def _make_camera_config() -> CameraConfig:
    # A realistic, credential-shaped -- but fictional -- URL, so the
    # no-URL-in-response/no-URL-in-log assertions in this file are
    # checking against something that could actually leak, not an empty
    # string that would pass by accident.
    return CameraConfig(
        rtsp_url="rtsp://redacted@camera.invalid/stream1",
        encoding="alaw",
        sample_rate=SAMPLE_RATE,
        channels=1,
    )


def _make_calibration_config(tmp_path: Path, **overrides: object) -> CalibrationConfig:
    fields = dict(
        dir=str(tmp_path),
        probe_duration_s=1.2,
        settle_s=0.05,
        tail_s=0.3,
        max_age_days=30,
        route_enabled=False,
    )
    fields.update(overrides)
    return CalibrationConfig(**fields)


class _LoopbackFake:
    """Both the audio source and the speaker sink `run_echo_calibration`
    is handed. `frames()` yields room tone until `write()` is called, then
    yields the written probe decoded, delayed by `delay_samples`, and
    scaled by whatever `gain_curve` returns -- the same injected-delay-and-
    gain shape `tests/test_echo_path.py`'s own `_build_recording` uses one
    layer down, driven here end to end through the runner instead of
    handed directly to `measure_echo_path`.

    `events` records `("sleep", seconds)` and `("write", byte_count)` in
    the order they actually happened, when this fake's own `write` and an
    injected sleep both append to it -- the mechanism the settle/window
    ordering test uses instead of measuring real wall-clock time.
    """

    def __init__(
        self,
        *,
        sample_rate: int = SAMPLE_RATE,
        encoding: str = "alaw",
        delay_samples: int = 200,
        gain_curve: Callable[[int], np.ndarray] | None = None,
        scale: float = 1.0,
        silence: bool = False,
        uncorrelated_noise_rms: float = 0.0,
        sink: SinkFormat | None = None,
    ) -> None:
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._delay_samples = delay_samples
        self._gain_curve = gain_curve
        self._scale = scale
        self._silence = silence
        self._uncorrelated_noise_rms = uncorrelated_noise_rms
        # 260923-pyj: the sink the probe was written in, when the test sets
        # one -- `sink_format()` is what `app.py`'s route reads (duck-typed,
        # like the real camera source) to decide whether to convert the
        # probe before writing it.
        self._sink = sink
        self.written: bytes | None = None
        self.events: list[tuple[str, object]] = []

    def source_format(self) -> SourceFormat:
        return SourceFormat(self._encoding, self._sample_rate)

    def sink_format(self) -> SinkFormat | None:
        return self._sink

    async def write(self, chunk: bytes) -> None:
        self.events.append(("write", len(chunk)))
        self.written = chunk

    async def frames(self):
        if self.written is None:
            tone = bytes(_CHUNK_BYTES)
            while True:
                yield tone
                await asyncio.sleep(0)
            return

        recording = self._build_recording()
        encoded = pcm16_to_alaw(recording.tobytes()) if self._encoding == "alaw" else recording.tobytes()
        for start in range(0, len(encoded), _CHUNK_BYTES):
            yield encoded[start : start + _CHUNK_BYTES]
            await asyncio.sleep(0)

    def _build_recording(self) -> np.ndarray:
        """Decode `self.written` -- the probe as it was actually written,
        in `self._sink`'s format when one is set -- back to PCM16 at this
        fake's own source rate, so the delay/gain injection below still
        operates in one consistent rate regardless of what format the
        probe travelled in."""
        assert self.written is not None
        if self._sink is None or self._sink.codec == "alaw":
            pcm16_bytes = alaw_to_pcm16(self.written)
        else:
            pcm16_bytes = self.written
        if self._sink is not None and self._sink.sample_rate != self._sample_rate:
            pcm16_bytes = _resample_pcm16(pcm16_bytes, self._sink.sample_rate, self._sample_rate)
        reference = np.frombuffer(pcm16_bytes, dtype="<i2").astype(np.float64)
        n = len(reference)
        total = self._delay_samples + n + self._delay_samples
        recording = np.zeros(total, dtype=np.float64)
        if self._uncorrelated_noise_rms > 0:
            # Independent of the written probe entirely -- room noise the
            # reference was never found in, not the reference itself plus
            # noise (that recovers a delay just fine, see
            # `test_measure_echo_path_with_moderate_noise_still_recovers_delay`
            # one layer down). A fixed seed keeps this deterministic.
            rng = np.random.default_rng(20260922)
            recording = rng.standard_normal(total) * self._uncorrelated_noise_rms
        elif not self._silence:
            gains = self._gain_curve(n) if self._gain_curve is not None else np.full(n, self._scale)
            recording[self._delay_samples : self._delay_samples + n] = reference * gains
        return np.clip(np.round(recording), -32768, 32767).astype(np.int16)


def _make_traced_sleep(fake: _LoopbackFake, *, pumps: int = 200) -> Callable[[float], object]:
    """A `sleep` that never really waits -- it appends to the same
    `events` log `_LoopbackFake.write` uses, then yields control to the
    event loop `pumps` times so a concurrently running drain task can
    consume everything the fake source is willing to produce right now,
    with no real delay."""

    async def _sleep(seconds: float) -> None:
        fake.events.append(("sleep", seconds))
        for _ in range(pumps):
            await asyncio.sleep(0)

    return _sleep


def _fixed_clock(value: datetime) -> Callable[[], datetime]:
    return lambda: value


# ---------------------------------------------------------------------------
# Task 1: run_echo_calibration
# ---------------------------------------------------------------------------


async def test_recovers_injected_delay_and_gain_end_to_end(tmp_path):
    fake = _LoopbackFake(delay_samples=300, scale=0.55)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)
    taken_at = datetime(2026, 9, 17, 12, 0, 0, tzinfo=timezone.utc)

    result = await run_echo_calibration(
        fake,
        fake,
        camera_config,
        calibration_config,
        "camera on the kitchen shelf",
        now=_fixed_clock(taken_at),
        sleep=_make_traced_sleep(fake),
    )

    assert result.failure_reason is None
    assert result.calibration is not None
    expected_delay_s = 300 / SAMPLE_RATE
    assert abs(result.calibration.delay_s - expected_delay_s) <= 2.0 / SAMPLE_RATE
    assert abs(result.calibration.gain - 0.55) < 0.05
    assert result.calibration.taken_at == taken_at
    assert result.calibration.placement_note == "camera on the kitchen shelf"


async def test_agc_present_for_a_monotonic_ramp_and_absent_for_constant_gain(tmp_path):
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    constant_fake = _LoopbackFake(delay_samples=150, scale=0.7)
    constant_result = await run_echo_calibration(
        constant_fake, constant_fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(constant_fake),
    )
    assert constant_result.failure_reason is None
    assert constant_result.calibration.agc_verdict == "absent"

    ramp_fake = _LoopbackFake(delay_samples=150, gain_curve=lambda n: np.linspace(0.3, 1.4, n))
    ramp_result = await run_echo_calibration(
        ramp_fake, ramp_fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(ramp_fake),
    )
    assert ramp_result.failure_reason is None
    assert ramp_result.calibration.agc_verdict == "present"


async def test_silence_returns_a_named_failure_and_writes_no_record(tmp_path):
    fake = _LoopbackFake(silence=True)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )

    assert result.calibration is None
    assert result.failure_reason
    assert list(Path(calibration_config.dir).glob("*")) == []


async def test_uncorrelated_noise_saves_an_echo_cancelled_record_instead_of_a_failure(tmp_path):
    """A camera that removes its own speaker output from its microphone
    (`aec` mode) produces exactly this recording: something came back --
    room noise, not silence -- but none of it is the probe. T-260922-eca:
    this is the one case that changed meaning from the failure the same
    recording shape reported before this plan."""
    fake = _LoopbackFake(uncorrelated_noise_rms=400.0)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )

    assert result.failure_reason is None
    assert result.calibration is not None
    assert result.calibration.echo_cancelled is True
    assert result.calibration.delay_s == 0.0
    assert result.calibration.gain == 0.0
    assert result.calibration.agc_verdict == "absent"
    assert result.calibration.echo_level > 0
    assert find_latest_calibration(calibration_config.dir) is not None


async def test_near_floor_room_tone_still_counts_as_an_echo_cancelled_camera(tmp_path):
    """The live house camera gates its microphone to rms ~8-20 of 32768,
    right at the A-law floor, so an rms threshold called it silence. Any
    sample above the codec floor proves the microphone is live."""
    fake = _LoopbackFake(uncorrelated_noise_rms=12.0)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )

    assert result.failure_reason is None
    assert result.calibration is not None
    assert result.calibration.echo_cancelled is True


# --- 260923-sfi (D2): only tapo_talk is trusted to explain "no echo" as its own AEC ---


async def test_uncorrelated_noise_with_no_camera_aec_stays_a_failure(tmp_path):
    """tcp and go2rtc speak through a different device (or a failed
    backchannel) -- "no echo came back" there means the probe never
    reached the microphone, not that a camera cancelled its own echo.
    `speaker_has_aec=False` must keep this the plain failure it always
    was, and save no record."""
    fake = _LoopbackFake(uncorrelated_noise_rms=400.0)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
        speaker_has_aec=False,
    )

    assert result.calibration is None
    assert result.failure_reason
    assert list(Path(calibration_config.dir).glob("*")) == []


async def test_uncorrelated_noise_with_no_speaker_has_aec_argument_defaults_to_refusing(tmp_path):
    """The default must refuse -- proven by omitting the keyword entirely,
    not by passing False explicitly."""
    fake = _LoopbackFake(uncorrelated_noise_rms=400.0)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )

    assert result.calibration is None
    assert result.failure_reason
    assert list(Path(calibration_config.dir).glob("*")) == []


def test_speaker_config_cancels_own_echo_only_on_tapo_talk():
    assert SpeakerConfig(backend="tapo_talk").cancels_own_echo is True
    assert SpeakerConfig(backend="tcp", tcp_url="tcp://speaker.invalid:1").cancels_own_echo is False
    assert SpeakerConfig(backend="go2rtc").cancels_own_echo is False


async def test_settle_elapses_before_the_probe_is_written_and_window_covers_duration_plus_tail(tmp_path):
    fake = _LoopbackFake(delay_samples=100, scale=0.9)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, settle_s=0.4, probe_duration_s=1.2, tail_s=0.6)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )

    assert result.failure_reason is None
    # Three events, in order: the settle sleep, the probe write sandwiched
    # between it and a recording-window sleep sized to duration + tail --
    # never just duration, which is exactly what would let a delay near the
    # top of the plausible range fall outside the recorded window.
    assert len(fake.events) == 3
    assert fake.events[0] == ("sleep", 0.4)
    assert fake.events[1][0] == "write"
    assert fake.events[2] == ("sleep", 1.2 + 0.6)


async def test_every_written_file_holds_no_audio_and_matches_the_declared_field_set(tmp_path):
    fake = _LoopbackFake(delay_samples=120, scale=0.8)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )

    assert result.failure_reason is None
    written = list(Path(calibration_config.dir).glob("*.json"))
    assert len(written) == 1
    for path in written:
        assert path.stat().st_size <= MAX_RECORD_FILE_SIZE_BYTES
    loaded = EchoCalibration.load(written[0])
    assert loaded is not None
    payload_keys = {f.name for f in dataclasses.fields(EchoCalibration)}
    assert set(dataclasses.asdict(loaded).keys()) == payload_keys


async def test_a_second_run_replaces_the_current_record_while_the_first_stays_readable(tmp_path):
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    fake_1 = _LoopbackFake(delay_samples=100, scale=0.5)
    result_1 = await run_echo_calibration(
        fake_1, fake_1, camera_config, calibration_config, "first",
        now=_fixed_clock(datetime(2026, 9, 17, 10, 0, 0, tzinfo=timezone.utc)),
        sleep=_make_traced_sleep(fake_1),
    )
    assert result_1.failure_reason is None

    fake_2 = _LoopbackFake(delay_samples=100, scale=0.5)
    result_2 = await run_echo_calibration(
        fake_2, fake_2, camera_config, calibration_config, "second",
        now=_fixed_clock(datetime(2026, 9, 17, 11, 0, 0, tzinfo=timezone.utc)),
        sleep=_make_traced_sleep(fake_2),
    )
    assert result_2.failure_reason is None

    written = sorted(Path(calibration_config.dir).glob("*.json"))
    assert len(written) == 2

    current = find_latest_calibration(calibration_config.dir)
    assert current is not None
    assert current.placement_note == "second"

    first_reloaded = EchoCalibration.load(written[0])
    assert first_reloaded is not None
    assert first_reloaded.placement_note == "first"


async def test_find_latest_calibration_of_an_empty_or_missing_directory_is_none(tmp_path):
    assert find_latest_calibration(tmp_path / "does-not-exist") is None
    assert find_latest_calibration(tmp_path) is None


async def test_decoding_an_unsupported_encoding_is_a_named_error():
    from atlas.calibration.runner import _decode_native

    with pytest.raises(CalibrationRunnerError):
        _decode_native(b"\x00\x00", "opus")


def test_runner_module_contains_no_reference_to_the_camera_url_field():
    source = Path("src/atlas/calibration/runner.py").read_text(encoding="utf-8")
    hits = [line.strip() for line in source.splitlines() if not line.lstrip().startswith("#") and "rtsp_url" in line]
    assert hits == []


# ---------------------------------------------------------------------------
# 260923-pyj: the probe follows the speaker sink format (D4)
# ---------------------------------------------------------------------------


async def test_probe_matches_todays_bytes_when_no_sink_is_given(tmp_path):
    """No sink means the camera case of today: unconverted A-law at the
    source's own rate -- proven byte for byte against build_probe itself,
    not just "some bytes were written"."""
    from atlas.audio.probe import build_probe

    fake = _LoopbackFake(delay_samples=200, scale=0.6)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )

    assert result.failure_reason is None
    expected_alaw, _ = build_probe(SAMPLE_RATE, calibration_config.probe_duration_s)
    assert fake.written == expected_alaw


@pytest.mark.parametrize(
    "sink, expected_written_bytes",
    [
        (SinkFormat("alaw", 8000), 9600),
        (SinkFormat("alaw", 16000), 19200),
        (SinkFormat("pcm", 8000), 19200),
        (SinkFormat("pcm", 24000), 57600),
    ],
)
async def test_probe_is_written_in_the_sink_format_and_still_correlates_at_the_source_rate(
    tmp_path, sink, expected_written_bytes
):
    """The probe travels in the speaker's own format (D4) -- delay and
    gain must still recover against the 8 kHz microphone reference, and
    the saved record must still describe the microphone, never the sink."""
    fake = _LoopbackFake(delay_samples=300, scale=0.55, sink=sink)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
        sink=sink,
    )

    assert fake.written is not None
    assert len(fake.written) == expected_written_bytes
    assert result.failure_reason is None
    assert result.calibration is not None
    expected_delay_s = 300 / SAMPLE_RATE
    assert abs(result.calibration.delay_s - expected_delay_s) <= 2.0 / SAMPLE_RATE
    # A looser gain tolerance than the no-sink case (0.05): this fixture's
    # own loopback round-trips the probe through a second resample this
    # test's fake source performs (sink rate back down to the microphone
    # rate) to decode what it "recorded" -- a fixture artifact of testing
    # resampled sinks with no real speaker/microphone acoustics, not a
    # measurement of the correlator's own precision.
    assert abs(result.calibration.gain - 0.55) < 0.1
    assert result.calibration.encoding == "alaw"
    assert result.calibration.sample_rate == SAMPLE_RATE


async def test_an_unsupported_sink_codec_raises_before_anything_plays(tmp_path):
    fake = _LoopbackFake(delay_samples=200, scale=0.6)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)

    with pytest.raises(CalibrationRunnerError) as exc:
        await run_echo_calibration(
            fake, fake, camera_config, calibration_config, "note",
            sleep=_make_traced_sleep(fake),
            sink=SinkFormat("mulaw", 8000),
        )

    assert "mulaw" in str(exc.value)
    assert fake.written is None


# ---------------------------------------------------------------------------
# Task 2 seam: the command-line caller imports run_echo_calibration and
# nothing from atlas.audio.echo_path -- the mechanism that makes the
# "one implementation" claim checkable rather than aspirational (D-20).
# ---------------------------------------------------------------------------


def _imported_module_names(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            names.add(node.module)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                names.add(alias.name)
    return names


def test_calibrate_echo_path_script_imports_the_runner_and_never_the_measurement_module():
    imports = _imported_module_names(_SCRIPT_PATH)
    assert any(name.endswith("calibration.runner") for name in imports)
    assert not any("audio.echo_path" in name for name in imports)


def test_calibrate_echo_path_script_reads_no_environment_variable_directly():
    tree = ast.parse(_SCRIPT_PATH.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in {"environ", "getenv"}:
            pytest.fail(f"scripts/calibrate_echo_path.py touches os.{node.attr} directly")
        if isinstance(node, ast.Subscript) and isinstance(node.value, ast.Attribute):
            if node.value.attr == "environ":
                pytest.fail("scripts/calibrate_echo_path.py reads os.environ[...] directly")


@pytest.mark.parametrize(
    "backend, expected_has_aec",
    [
        ("tcp", False),
        ("tapo_talk", True),
        ("go2rtc", False),
    ],
)
async def test_run_passes_the_tts_sink_to_run_echo_calibration(tmp_path, monkeypatch, backend, expected_has_aec):
    """260923-pyj (D4): the script writes the probe into the same FIFO the
    pod's own egress reads, so it must convert the probe the same way --
    `_run` reads `config.tts` and passes it through as `run_echo_
    calibration`'s `sink` keyword. 260923-sfi (D1, D2): it must also pad
    that FifoWriter with sink-format silence -- the probe is short, so an
    unpadded FIFO would cut it the same way an unpadded reply is cut --
    and pass `speaker_has_aec` from `config.speaker.cancels_own_echo`."""
    from atlas.audio.cue import silence
    from atlas.config import TtsConfig

    recorded: dict[str, object] = {}
    fifo_kwargs: dict[str, object] = {}

    class _FakeCameraSource:
        def start(self) -> None:
            return None

        async def close(self) -> None:
            return None

    class _FakeFifoWriter:
        def __init__(self, path, **kwargs):
            self.path = path
            fifo_kwargs.update(kwargs)

        async def open(self) -> None:
            return None

        async def close(self) -> None:
            return None

    async def _fake_run_echo_calibration(source, speaker, camera_config, calibration_config, placement_note, **kwargs):
        recorded.update(kwargs)
        return CalibrationRunResult(calibration=None, failure_reason="stub")

    monkeypatch.setattr(calibrate_echo_path, "_open_camera", lambda camera_config: _FakeCameraSource())
    monkeypatch.setattr(calibrate_echo_path, "FifoWriter", _FakeFifoWriter)
    monkeypatch.setattr(calibrate_echo_path, "run_echo_calibration", _fake_run_echo_calibration)

    config = SimpleNamespace(
        camera=_make_camera_config(),
        speaker=SpeakerConfig(fifo_path=str(tmp_path / "speaker.fifo"), reopen_timeout_s=1.0, backend=backend),
        calibration=_make_calibration_config(tmp_path),
        tts=TtsConfig(codec="pcm", sample_rate=24000),
    )

    await calibrate_echo_path._run(config, "note")

    assert recorded["sink"] == SinkFormat(codec="pcm", sample_rate=24000)
    assert recorded["speaker_has_aec"] is expected_has_aec
    assert fifo_kwargs["pad_bytes"] == silence(SinkFormat("pcm", 24000), 0.2)
    assert fifo_kwargs["pad_idle_s"] == 0.15


def test_validate_environment_rejects_a_placement_note_shaped_like_a_url(tmp_path):
    with pytest.raises(calibrate_echo_path.CalibrationScriptError):
        calibrate_echo_path.validate_environment(tmp_path, "see rtsp://camera.invalid:554/stream1")
    with pytest.raises(calibrate_echo_path.CalibrationScriptError):
        calibrate_echo_path.validate_environment(tmp_path, "credentials are admin:hunter2@camera.invalid")


def test_validate_environment_rejects_an_unwritable_calibration_directory(tmp_path):
    unwritable = tmp_path / "locked"
    unwritable.mkdir()
    unwritable.chmod(0o400)
    try:
        with pytest.raises(calibrate_echo_path.CalibrationScriptError):
            calibrate_echo_path.validate_environment(unwritable / "calibration", "a fine note")
    finally:
        unwritable.chmod(0o700)  # tmp_path cleanup needs write+execute back


def test_validate_environment_accepts_a_writable_directory_and_a_plain_note(tmp_path):
    calibrate_echo_path.validate_environment(tmp_path / "calibration", "camera on the kitchen shelf")
    # validate_environment's own write-check cleans up after itself -- it
    # proves the directory is writable, it does not leave a marker behind.
    assert list((tmp_path / "calibration").glob("*")) == []


def test_handle_result_of_a_failed_run_exits_non_zero_and_prints_the_reason(tmp_path, capsys):
    result = CalibrationRunResult(calibration=None, failure_reason="no correlation peak above threshold")
    exit_code = calibrate_echo_path._handle_result(result, tmp_path)
    captured = capsys.readouterr()
    assert exit_code == 1
    assert "no correlation peak above threshold" in captured.err


async def test_handle_result_of_a_successful_run_exits_zero_and_names_every_measured_number(tmp_path, capsys):
    fake = _LoopbackFake(delay_samples=100, scale=0.6)
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path)
    run_result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "note",
        sleep=_make_traced_sleep(fake),
    )
    assert run_result.failure_reason is None

    exit_code = calibrate_echo_path._handle_result(run_result, Path(calibration_config.dir))
    captured = capsys.readouterr()

    assert exit_code == 0
    assert "delay" in captured.out
    assert "gain" in captured.out
    assert "correlation confidence" in captured.out
    assert "AGC verdict" in captured.out
    assert "record written" in captured.out
    assert "rtsp://" not in captured.out
    assert "://" not in captured.out


# ---------------------------------------------------------------------------
# Task 3 seam: app.py's two routes both call run_echo_calibration and
# never atlas.audio.echo_path directly.
# ---------------------------------------------------------------------------


def test_app_module_imports_the_runner_and_never_the_measurement_module_for_calibration():
    imports = _imported_module_names(Path("src/atlas/app.py"))
    assert any(name.endswith("calibration.runner") for name in imports)
    assert not any("audio.echo_path" in name for name in imports)


def test_calibration_routes_are_disabled_by_default_and_name_the_config_key(tmp_path, monkeypatch):
    """Full boot, real HTTP: the shipped default (no `calibration:` block
    in the fake config, matching `config.example.yaml`'s own
    `route_enabled: false`) must make both routes answer with a client
    error naming the key that would turn them on -- not a 404 that looks
    like a typo, and not a server error.

    Plan 03-05 moved both routes behind `require_role(Role.OPERATOR)`
    (T-03-32) -- this test now authenticates as the smoke fixture's
    pre-seeded admin first, so the assertions below still exercise the
    calibration-specific refusal rather than an auth 401 that would have
    masked it.
    """
    import test_startup_smoke as smoke
    from fastapi.testclient import TestClient

    from atlas.auth.tokens import issue_access_token
    from atlas.config import SecurityConfig

    # `test_startup_smoke.py`'s own autouse fixture only applies within
    # that module -- this file needs the same structurally-valid test key
    # (`validate_secret_key_strength`, plan 03-05) set explicitly.
    monkeypatch.setenv("ATLAS_SECRET_KEY", smoke._TEST_SECRET_KEY)
    monkeypatch.setattr(app_module, "CONFIG_PATH", str(smoke._write_fake_config(tmp_path)))
    monkeypatch.setattr(smoke.plugin_manager_module, "start_plugin_host", smoke._fake_start_plugin_host)
    monkeypatch.setattr(app_module, "precache_all", smoke._fake_precache_all)
    monkeypatch.setattr(app_module, "run_migrations", smoke._fake_run_migrations)
    monkeypatch.setattr(app_module, "build_engine", smoke._fake_build_engine)
    monkeypatch.setattr(app_module, "_build_repositories", smoke._fake_build_repositories)
    monkeypatch.setattr(app_module.brain_race, "build_tiers", smoke._fake_build_tiers)
    monkeypatch.setattr(app_module, "_build_wake_detector", smoke._fake_build_wake_detector)
    monkeypatch.setattr(app_module, "_build_ffmpeg_supervisor", smoke._fake_build_ffmpeg_supervisor)
    monkeypatch.setattr(app_module, "CameraAudioSource", smoke._FakeCameraSource)

    security = SecurityConfig()
    # id=1/role="admin" is exactly what `smoke._fake_build_repositories`
    # pre-seeds -- an admin outranks the operator role these routes require.
    token = issue_access_token(user_id=1, role="admin", security=security)

    with TestClient(app_module.app, cookies={security.cookie_name: token}) as client:
        read_response = client.get("/calibration/echo-path")
        assert read_response.status_code == 403
        assert "calibration.route_enabled" in read_response.json()["detail"]

        run_response = client.post("/calibration/echo-path/run", json={"placement_note": "note"})
        assert run_response.status_code == 403
        assert "calibration.route_enabled" in run_response.json()["detail"]


def _install_calibration_state(
    camera_config, calibration_config, camera_source, speaker_writer, *, speaker_backend: str = "go2rtc"
) -> None:
    """Sets exactly the `app.state` attributes the two calibration routes
    read, with no real lifespan boot -- the route handlers are plain
    coroutines FastAPI's `@app.get`/`@app.post` decorators register and
    return unchanged, so calling them directly here needs no ASGI
    machinery, no tool host, no brain tier, and no wake detector.

    260923-sfi (D2): `speaker_backend` builds the `SpeakerConfig` the route
    reads `cancels_own_echo` from -- `"go2rtc"` (default) matches every
    existing caller's behaviour before this field existed."""
    app_module.app.state.config = SimpleNamespace(
        calibration=calibration_config, camera=camera_config, speaker=SpeakerConfig(backend=speaker_backend)
    )
    app_module.app.state.camera_source = camera_source
    app_module.app.state.speaker_writer = speaker_writer
    app_module.app.state.calibration_in_progress = False


async def test_read_route_returns_not_found_when_no_calibration_exists(tmp_path):
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    _install_calibration_state(_make_camera_config(), calibration_config, None, None)

    with pytest.raises(HTTPException) as exc:
        await app_module.get_echo_path_calibration()
    assert exc.value.status_code == 404


async def test_read_route_returns_the_stored_record_and_its_age_with_no_url_in_the_response(tmp_path):
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    fake = _LoopbackFake(delay_samples=100, scale=0.5)
    run_result = await run_echo_calibration(
        fake, fake, camera_config, calibration_config, "kitchen shelf",
        now=_fixed_clock(datetime(2020, 1, 1, tzinfo=timezone.utc)),
        sleep=_make_traced_sleep(fake),
    )
    assert run_result.failure_reason is None
    _install_calibration_state(camera_config, calibration_config, None, None)

    response = await app_module.get_echo_path_calibration()

    assert response["agc_verdict"] == run_result.calibration.agc_verdict
    assert response["age_days"] > 0
    assert response["echo_cancelled"] is run_result.calibration.echo_cancelled
    body = json.dumps(response)
    assert "rtsp://" not in body
    assert "redacted" not in body
    assert "://" not in body


async def test_run_route_returns_measured_numbers_when_enabled_with_no_url_in_the_response(tmp_path):
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    fake = _LoopbackFake(delay_samples=120, scale=0.6)
    _install_calibration_state(camera_config, calibration_config, fake, fake)

    response = await app_module.run_echo_path_calibration(
        app_module.CalibrationRunRequest(placement_note="kitchen shelf")
    )

    assert response["gain"] == pytest.approx(0.6, abs=0.05)
    assert response["agc_verdict"] in {"absent", "present", "indeterminate"}
    assert response["echo_cancelled"] is False
    assert app_module.app.state.calibration_in_progress is False
    body = json.dumps(response)
    assert "rtsp://" not in body
    assert "redacted" not in body


async def test_run_route_reads_the_sink_from_the_camera_source_duck_typed(tmp_path):
    """260923-pyj (D4): the route reads `sink_format()` off the camera
    source it already holds -- never `config.tts`, which the route's own
    tests give as a `SimpleNamespace` with no `tts` attribute at all."""
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    fake = _LoopbackFake(delay_samples=120, scale=0.6, sink=SinkFormat("pcm", 24000))
    _install_calibration_state(camera_config, calibration_config, fake, fake)

    await app_module.run_echo_path_calibration(app_module.CalibrationRunRequest(placement_note="kitchen shelf"))

    assert len(fake.written) == 57600


async def test_run_route_returns_conflict_when_a_run_is_already_in_progress(tmp_path):
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    _install_calibration_state(camera_config, calibration_config, None, None)
    app_module.app.state.calibration_in_progress = True
    try:
        with pytest.raises(HTTPException) as exc:
            await app_module.run_echo_path_calibration(app_module.CalibrationRunRequest())
        assert exc.value.status_code == 409
    finally:
        app_module.app.state.calibration_in_progress = False


async def test_run_route_reports_a_failed_measurement_by_name_rather_than_a_partial_record(tmp_path):
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    fake = _LoopbackFake(silence=True)
    _install_calibration_state(camera_config, calibration_config, fake, fake)

    with pytest.raises(HTTPException) as exc:
        await app_module.run_echo_path_calibration(app_module.CalibrationRunRequest())

    assert exc.value.status_code == 422
    assert exc.value.detail
    assert list(Path(calibration_config.dir).glob("*")) == []


async def test_run_route_refuses_the_echo_cancelled_fallback_on_the_tcp_backend(tmp_path):
    """260923-sfi (D2): the route reads `speaker_has_aec` from
    `config.speaker.cancels_own_echo`, which is False on tcp -- an
    uncorrelated recording there must report the plain failure, never a
    saved echo_cancelled record."""
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    fake = _LoopbackFake(uncorrelated_noise_rms=400.0)
    _install_calibration_state(camera_config, calibration_config, fake, fake, speaker_backend="tcp")

    with pytest.raises(HTTPException) as exc:
        await app_module.run_echo_path_calibration(app_module.CalibrationRunRequest(placement_note="note"))

    assert exc.value.status_code == 422
    assert list(Path(calibration_config.dir).glob("*")) == []


async def test_run_route_allows_the_echo_cancelled_fallback_on_the_tapo_talk_backend(tmp_path):
    """The one backend this fallback is trusted for."""
    camera_config = _make_camera_config()
    calibration_config = _make_calibration_config(tmp_path, route_enabled=True)
    fake = _LoopbackFake(uncorrelated_noise_rms=400.0)
    _install_calibration_state(camera_config, calibration_config, fake, fake, speaker_backend="tapo_talk")

    response = await app_module.run_echo_path_calibration(
        app_module.CalibrationRunRequest(placement_note="note")
    )

    assert response["echo_cancelled"] is True
