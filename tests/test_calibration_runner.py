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

import asyncio
import dataclasses
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import pytest

from spire_voice.audio.alaw import alaw_to_pcm16, pcm16_to_alaw
from spire_voice.calibration.record import EchoCalibration
from spire_voice.calibration.runner import (
    MAX_RECORD_FILE_SIZE_BYTES,
    CalibrationRunnerError,
    find_latest_calibration,
    run_echo_calibration,
)
from spire_voice.config import CalibrationConfig, CameraConfig
from spire_voice.transports.base import SourceFormat

SAMPLE_RATE = 8000
_CHUNK_BYTES = 64


def _make_camera_config() -> CameraConfig:
    return CameraConfig(rtsp_url="", encoding="alaw", sample_rate=SAMPLE_RATE, channels=1)


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
    ) -> None:
        self._sample_rate = sample_rate
        self._encoding = encoding
        self._delay_samples = delay_samples
        self._gain_curve = gain_curve
        self._scale = scale
        self._silence = silence
        self.written: bytes | None = None
        self.events: list[tuple[str, object]] = []

    def source_format(self) -> SourceFormat:
        return SourceFormat(self._encoding, self._sample_rate)

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
        assert self.written is not None
        reference = np.frombuffer(alaw_to_pcm16(self.written), dtype="<i2").astype(np.float64)
        n = len(reference)
        total = self._delay_samples + n + self._delay_samples
        recording = np.zeros(total, dtype=np.float64)
        if not self._silence:
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
    from spire_voice.calibration.runner import _decode_native

    with pytest.raises(CalibrationRunnerError):
        _decode_native(b"\x00\x00", "opus")


def test_runner_module_contains_no_reference_to_the_camera_url_field():
    source = Path("src/spire_voice/calibration/runner.py").read_text(encoding="utf-8")
    hits = [line.strip() for line in source.splitlines() if not line.lstrip().startswith("#") and "rtsp_url" in line]
    assert hits == []
