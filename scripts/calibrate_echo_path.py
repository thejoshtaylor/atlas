#!/usr/bin/env python3
"""Run one echo-path calibration (Tier 1, VOICE-07) from the command line --
a thin caller of `run_echo_calibration` (`atlas.calibration.runner`),
and nothing else. All of the measuring lives in the runner; this script
parses arguments, validates what is checkable before anything opens, opens
the camera source and the speaker FIFO writer, calls `run_echo_calibration`,
prints the result, and closes both. A test parses this file's own imports
to prove it: `run_echo_calibration` is imported, `atlas.audio.
echo_path` never is (`tests/test_calibration_runner.py`).

Follows `scripts/capture_wake_corpus.py`'s own conventions: validate up
front with a named exception -- the calibration directory's writability,
the placement note's shape -- because discovering a problem after the
operator has already quieted the room and stood still for the probe is the
same lost evening that script's own validation exists to prevent. Default
paths resolve relative to the repository root. This module reads no
environment variable directly; it reaches the camera and the speaker FIFO
only through `atlas.config.load_config`, the one place `${TAPO_USER}`/
`${TAPO_PASSWORD}` are ever expanded from the environment -- run this
script through `scripts/dev-calibrate-echo.sh`, which sources `.env` the
way `dev-run.sh`/`dev-capture-corpus.sh` do.

The calibration writes into the same speaker FIFO the running assistant
already writes into (02-RESEARCH.md Open Question 2's own open question,
inherited rather than re-decided here) -- a reachable reader on that pipe
is expected to already exist, the same real-deployment assumption the
FIFO's own initial open makes (`speaker/fifo_writer.py`'s module
docstring): run this against a deployment where the assistant itself is
already running, never as a substitute for it.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from atlas.calibration.record import EchoCalibration
from atlas.calibration.runner import CalibrationRunResult, run_echo_calibration
from atlas.config import CameraConfig, Config, ConfigError, load_config
from atlas.providers.tts_xai import SinkFormat
from atlas.speaker.fifo_writer import FifoWriter, SpeakerError
from atlas.transports.camera import CameraAudioSource

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"

# Matches `calibration/record.py`'s own `_validate_placement_note` markers --
# this script's own copy exists so the operator learns of a bad note before
# the room has been quieted and the probe played, not after
# `EchoCalibration.__post_init__` raises the same rejection once it is too
# late to matter (`run_echo_calibration` still holds that check too; this
# is belt-and-suspenders on the one field an operator types free text into).
_URL_SCHEME_SEPARATOR = "://"
_USERINFO_MARKER = "@"

# The naming scheme `calibration/runner.py`'s own (private) filename
# helper writes -- duplicated here only for display, never for measurement:
# this script reads no calibration record it did not just receive back
# from `run_echo_calibration` directly, it only needs this pattern to tell
# the operator where that record landed on disk.
_RECORD_GLOB = "echo_path-*.json"


class CalibrationScriptError(ValueError):
    """Raised before anything is opened -- an unwritable calibration
    directory or a placement note shaped like a URL surfaces here, rather
    than after the operator has already quieted the room and played the
    probe (`scripts/capture_wake_corpus.py`'s own `validate_environment`
    exists for exactly this reason)."""


class _NullSpeaker:
    """`CameraAudioSource` takes a speaker sink for reply audio it never
    sends here -- `run_echo_calibration` writes the probe through its own
    separate `speaker` argument (the real `FifoWriter` below), never
    through `CameraAudioSource.send_audio`."""

    async def write(self, chunk: bytes) -> None:
        return None


def validate_environment(calibration_dir: Path, placement_note: str) -> None:
    """Validate everything checkable before opening the camera or the
    speaker FIFO: the placement note carries no URL scheme separator and
    no userinfo marker, and the calibration directory is actually
    writable. Raises `CalibrationScriptError` naming what failed.
    """
    if _URL_SCHEME_SEPARATOR in placement_note:
        raise CalibrationScriptError(
            f"--placement-note may not contain a URL scheme separator ({_URL_SCHEME_SEPARATOR!r}) "
            "-- this field is never allowed to carry the camera's RTSP URL"
        )
    if _USERINFO_MARKER in placement_note:
        raise CalibrationScriptError(
            f"--placement-note may not contain a userinfo marker ({_USERINFO_MARKER!r}) "
            "-- this field is never allowed to carry a credential"
        )
    probe_path = calibration_dir / ".write-check"
    try:
        calibration_dir.mkdir(parents=True, exist_ok=True)
        probe_path.write_text("")
        probe_path.unlink(missing_ok=True)
    except OSError as exc:
        raise CalibrationScriptError(
            f"calibration directory is not writable: {calibration_dir} ({type(exc).__name__})"
        ) from exc


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run one echo-path calibration (Tier 1, VOICE-07): play a known probe out of "
            "the camera speaker, record it back from the camera microphone, and measure "
            "the round-trip delay, the echo level, the gain, and whether the camera applies "
            "automatic gain control."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"the application config to read camera.*/speaker.*/calibration.* from (default: {_DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--placement-note",
        required=True,
        help="where the camera is and what the room sounds like -- never a URL or a credential",
    )
    return parser


def _open_camera(camera_config: CameraConfig) -> CameraAudioSource:
    return CameraAudioSource(camera_config, _NullSpeaker())


def _latest_record_path(calibration_dir: Path) -> Path | None:
    candidates = sorted(calibration_dir.glob(_RECORD_GLOB))
    return candidates[-1] if candidates else None


async def _run(config: Config, placement_note: str) -> CalibrationRunResult:
    source = _open_camera(config.camera)
    source.start()
    speaker = FifoWriter(config.speaker.fifo_path, reopen_timeout_s=config.speaker.reopen_timeout_s)
    await speaker.open()
    try:
        # The script writes into the same FIFO the running pod's own
        # egress reads, so the probe must be in that same TTS sink format.
        sink = SinkFormat(codec=config.tts.codec, sample_rate=config.tts.sample_rate)
        return await run_echo_calibration(
            source, speaker, config.camera, config.calibration, placement_note, sink=sink
        )
    finally:
        await speaker.close()
        await source.close()


def _print_summary(calibration: EchoCalibration, record_path: Path | None) -> None:
    """Names the measured delay, gain, AGC verdict, correlation confidence
    and record path -- never a URL, never audio."""
    print(f"delay: {calibration.delay_s * 1000:.1f} ms")
    print(f"gain: {calibration.gain:.3f}")
    print(f"correlation confidence: {calibration.confidence:.3f}")
    print(f"AGC verdict: {calibration.agc_verdict}")
    if calibration.agc_verdict != "absent":
        print(
            "the echo path is time-varying -- plan 02-12's correlation will track the "
            "level rather than assume a fixed gain"
        )
    print(f"record written: {record_path if record_path is not None else '(not found on disk)'}")


def _handle_result(result: CalibrationRunResult, calibration_dir: Path) -> int:
    """Print the outcome and return the process exit code -- separated
    from `main()`'s own I/O-heavy orchestration so a test can drive this
    against a `CalibrationRunResult` it built directly, with no camera and
    no speaker FIFO involved at all.

    A failed result exits non-zero and prints the measurement's own named
    reason (to stderr): a calibration that measured nothing must never be
    mistaken for one that did (T-02-52).
    """
    if result.failure_reason is not None:
        print(f"calibration measured nothing: {result.failure_reason}", file=sys.stderr)
        return 1

    assert result.calibration is not None
    _print_summary(result.calibration, _latest_record_path(calibration_dir))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except ConfigError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    calibration_dir = Path(config.calibration.dir)
    try:
        validate_environment(calibration_dir, args.placement_note)
    except CalibrationScriptError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        result = asyncio.run(_run(config, args.placement_note))
    except SpeakerError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    return _handle_result(result, calibration_dir)


if __name__ == "__main__":
    sys.exit(main())
