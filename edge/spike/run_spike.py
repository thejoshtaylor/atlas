#!/usr/bin/env python3
"""The D-18 spike CLI: one subcommand per open question, each writing a
machine-readable JSON result whose verdict comes from `analysis.py`'s
tested functions -- this file never computes a pass/fail rule itself
(D-19).

Every subcommand that touches real hardware (`sounddevice`, `sherpa_onnx`,
`usb`) imports it inside that subcommand's own function, so `--help` and
this repository's test suite both run on a dev host with no XVF3800 or
microphone attached -- no hardware import happens at module load.

Run on the Pi, inside `edge/`:
    uv run python spike/run_spike.py info
    uv run python spike/run_spike.py channels --position "standing close"
    uv run python spike/run_spike.py doa --position "standing close" --angle 90
    uv run python spike/run_spike.py aec --volume-note "half volume" --vad-model models/silero_vad.onnx
    uv run python spike/run_spike.py multibeam --talker-a-angle 45 --talker-b-angle 200
    uv run python spike/run_spike.py onset --vad-model models/silero_vad.onnx
    uv run python spike/run_spike.py report
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

import analysis
import xvf3800_control

RESULTS_DIR = Path(__file__).resolve().parent / "results"

# D-08's onset/offset measurement, and the AEC double-talk VAD count, both
# poll at the same 32ms chunk size the operator's own spike benchmark used
# (Open Question 2) -- not the sherpa-onnx example's coarser 100ms, which
# would fold its own poll granularity into the number being measured.
_VAD_WINDOW_S = 0.032


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _write_result(name: str, payload: dict[str, Any]) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{name}-{_timestamp()}.json"
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(json.dumps(payload, indent=2, default=str))
    return path


def _write_wav(path: Path, buffer: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_channels = buffer.shape[1] if buffer.ndim == 2 else 1
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.ascontiguousarray(buffer).astype(np.int16).tobytes())


def _count_vad_segments(samples: np.ndarray, sample_rate: int, vad_model_path: str) -> int:
    """`samples`: mono float32 in [-1, 1]. Returns the number of speech
    segments Silero VAD closed while consuming `samples` in 32ms windows."""
    import sherpa_onnx

    config = sherpa_onnx.VadModelConfig()
    config.silero_vad.model = vad_model_path
    config.sample_rate = sample_rate
    vad = sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    window = max(1, int(round(sample_rate * _VAD_WINDOW_S)))
    segments = 0
    for start in range(0, len(samples) - window + 1, window):
        vad.accept_waveform(samples[start : start + window])
        while not vad.empty():
            vad.pop()
            segments += 1
    return segments


def cmd_info(args: argparse.Namespace) -> int:
    import sounddevice as sd

    result: dict[str, Any] = {"command": "info"}
    try:
        device = xvf3800_control.find_device()
        version_bytes = xvf3800_control.read_parameter(
            device, xvf3800_control.PARAMETERS["VERSION"]
        )
        result["xvf_version"] = list(xvf3800_control.decode_version(version_bytes))
    except xvf3800_control.XvfNotFound as exc:
        result["xvf_error"] = str(exc)

    matches = []
    for index, device_info in enumerate(sd.query_devices()):
        if args.device_name.lower() in device_info["name"].lower():
            matches.append(
                {
                    "index": index,
                    "name": device_info["name"],
                    "max_input_channels": device_info["max_input_channels"],
                    "max_output_channels": device_info["max_output_channels"],
                }
            )
    result["sounddevice_matches"] = matches
    _write_result("info", result)
    return 0


def cmd_channels(args: argparse.Namespace) -> int:
    """Q1: which beam each capture channel carries."""
    import sounddevice as sd

    sample_rate = 16000
    safe_position = args.position.replace(" ", "_")

    print(f"[{args.position}] stay quiet for 5 seconds...")
    silence = sd.rec(int(5.0 * sample_rate), samplerate=sample_rate, channels=2, dtype="int16")
    sd.wait()

    print(f"[{args.position}] now say: {args.sentence!r}")
    time.sleep(1.0)
    speech = sd.rec(
        int(args.duration_s * sample_rate), samplerate=sample_rate, channels=2, dtype="int16"
    )
    sd.wait()

    buffer = np.concatenate([silence, speech], axis=0)
    stamp = _timestamp()
    stereo_path = RESULTS_DIR / f"channels-{safe_position}-{stamp}.wav"
    _write_wav(stereo_path, buffer, sample_rate)
    channel_paths = []
    for ch in range(buffer.shape[1]):
        ch_path = RESULTS_DIR / f"channels-{safe_position}-ch{ch}-{stamp}.wav"
        _write_wav(ch_path, buffer[:, ch : ch + 1], sample_rate)
        channel_paths.append(str(ch_path))

    snr = analysis.channel_snr_db(buffer.astype(np.float64), sample_rate)
    result: dict[str, Any] = {
        "command": "channels",
        "position": args.position,
        "snr_db": list(snr),
        "stereo_wav": str(stereo_path),
        "channel_wavs": channel_paths,
    }
    _write_result(f"channels-{safe_position}", result)
    return 0


def cmd_doa(args: argparse.Namespace) -> int:
    """Q2: read DoA through the vendor control interface."""
    device = xvf3800_control.find_device()
    readings: list[dict[str, Any]] = []
    end_time = time.monotonic() + 10.0
    while time.monotonic() < end_time:
        doa_bytes = xvf3800_control.read_parameter(
            device, xvf3800_control.PARAMETERS["DOA_VALUE"]
        )
        azimuth_bytes = xvf3800_control.read_parameter(
            device, xvf3800_control.PARAMETERS["AEC_AZIMUTH_VALUES"]
        )
        doa_deg, speech_detected = xvf3800_control.decode_doa_value(doa_bytes)
        azimuths_deg = xvf3800_control.decode_azimuth_values(azimuth_bytes)
        readings.append(
            {
                "doa_deg": doa_deg,
                "speech_detected": speech_detected,
                "azimuths_deg": list(azimuths_deg),
            }
        )
        time.sleep(0.1)

    result: dict[str, Any] = {
        "command": "doa",
        "position": args.position,
        "declared_angle_deg": args.angle,
        "readings": readings,
    }
    _write_result(f"doa-{args.position.replace(' ', '_')}", result)
    return 0


def cmd_aec(args: argparse.Namespace) -> int:
    """Q3: does playback through the aux output give working AEC."""
    import sounddevice as sd

    sample_rate = 16000
    print(f"Volume check: {args.volume_note}")
    print("Recording 20 seconds of you reading -- this becomes the playback reference.")
    reference = sd.rec(int(20.0 * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
    sd.wait()
    _write_wav(RESULTS_DIR / f"aec-reference-{_timestamp()}.wav", reference, sample_rate)

    def _run(label: str) -> int:
        print(f"{label} -- playing the reference back through the XVF3800...")
        capture = sd.playrec(reference, samplerate=sample_rate, channels=2, dtype="int16")
        sd.wait()
        asr_channel = capture[:, args.asr_channel].astype(np.float32) / 32768.0
        return _count_vad_segments(asr_channel, sample_rate, args.vad_model)

    playback_only_segments = _run("Stay quiet")
    input("Press enter, then talk over the playback this time...")
    doubletalk_segments = _run("Talk over the playback")

    verdict = analysis.aec_verdict(playback_only_segments, doubletalk_segments)
    result: dict[str, Any] = {"command": "aec", "volume_note": args.volume_note, **verdict}
    _write_result("aec", result)
    return 0


def cmd_multibeam(args: argparse.Namespace) -> int:
    """Q4: does the multi-beam output give one stream per talker."""
    device = xvf3800_control.find_device()

    def _poll(seconds: float, label: str) -> list[dict[str, Any]]:
        print(f"{label} for {seconds:.0f} seconds...")
        readings: list[dict[str, Any]] = []
        end_time = time.monotonic() + seconds
        while time.monotonic() < end_time:
            azimuth_bytes = xvf3800_control.read_parameter(
                device, xvf3800_control.PARAMETERS["AEC_AZIMUTH_VALUES"]
            )
            spenergy_bytes = xvf3800_control.read_parameter(
                device, xvf3800_control.PARAMETERS["AEC_SPENERGY_VALUES"]
            )
            readings.append(
                {
                    "azimuths_deg": list(xvf3800_control.decode_azimuth_values(azimuth_bytes)),
                    "spenergy": list(xvf3800_control.decode_spenergy(spenergy_bytes)),
                }
            )
            time.sleep(0.1)
        return readings

    alternating = _poll(20.0, "Talker A and B alternate")
    together = _poll(10.0, "Talker A and B speak together")

    result: dict[str, Any] = {
        "command": "multibeam",
        "talker_angles_deg": [args.talker_a_angle, args.talker_b_angle],
        "alternating": alternating,
        "together": together,
    }
    if together:
        latest_azimuths = together[-1]["azimuths_deg"]
        verdict = analysis.multibeam_verdict(
            tuple(latest_azimuths), (args.talker_a_angle, args.talker_b_angle)
        )
        result.update(verdict)
    _write_result("multibeam", result)
    return 0


def cmd_onset(args: argparse.Namespace) -> int:
    """Q5: the Silero onset delay, for the pre_roll_ms default."""
    import sounddevice as sd

    sample_rate = 16000
    onset_delays_ms: list[float] = []
    offset_lags_ms: list[float] = []
    for i in range(args.count):
        print(f"[{i + 1}/{args.count}] stay quiet for 1 second...")
        time.sleep(1.0)
        print(f"[{i + 1}/{args.count}] now say the wake phrase...")
        buffer = sd.rec(int(2.0 * sample_rate), samplerate=sample_rate, channels=1, dtype="int16")
        sd.wait()
        samples = buffer[:, 0].astype(np.float64)
        onset_delays_ms.append(analysis.energy_onset_ms(samples, sample_rate))
        offset_lags_ms.append(analysis.energy_offset_ms(samples, sample_rate))

    pre_roll_ms = analysis.derive_pre_roll_ms(onset_delays_ms)
    tail_ms = analysis.derive_tail_ms(offset_lags_ms, args.endpointing_ms)

    result: dict[str, Any] = {
        "command": "onset",
        "onset_delays_ms": onset_delays_ms,
        "offset_lags_ms": offset_lags_ms,
        "endpointing_ms": args.endpointing_ms,
        "derived_pre_roll_ms": pre_roll_ms,
        "derived_tail_ms": tail_ms,
    }
    print(f"derived_pre_roll_ms={pre_roll_ms} derived_tail_ms={tail_ms}")
    _write_result("onset", result)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """Merge the newest JSON per subcommand into results/report.json."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    latest: dict[str, dict[str, Any]] = {}
    for path in sorted(RESULTS_DIR.glob("*.json")):
        if path.name == "report.json":
            continue
        prefix = path.stem.rsplit("-", 1)[0]  # strip the trailing UTC timestamp
        latest[prefix] = json.loads(path.read_text())

    report_path = RESULTS_DIR / "report.json"
    report_path.write_text(json.dumps(latest, indent=2))
    print(json.dumps(latest, indent=2))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D-18 spike CLI: one subcommand per open question about the XVF3800/Pi microphone."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    info_p = subparsers.add_parser(
        "info", help="Show the USB device, VERSION, and matching sounddevice devices."
    )
    info_p.add_argument("--device-name", default="reSpeaker")
    info_p.set_defaults(func=cmd_info)

    channels_p = subparsers.add_parser(
        "channels", help="Q1: which beam each capture channel carries."
    )
    channels_p.add_argument("--position", required=True)
    channels_p.add_argument(
        "--sentence", default="The quick brown fox jumps over the lazy dog."
    )
    channels_p.add_argument("--duration-s", type=float, default=5.0)
    channels_p.set_defaults(func=cmd_channels)

    doa_p = subparsers.add_parser("doa", help="Q2: read DoA through the vendor control interface.")
    doa_p.add_argument("--position", required=True)
    doa_p.add_argument("--angle", type=float, required=True)
    doa_p.set_defaults(func=cmd_doa)

    aec_p = subparsers.add_parser(
        "aec", help="Q3: whether playback through the aux output gives working AEC."
    )
    aec_p.add_argument("--volume-note", required=True)
    aec_p.add_argument("--vad-model", required=True)
    aec_p.add_argument(
        "--asr-channel", type=int, default=1, help="Capture channel index to run VAD against."
    )
    aec_p.set_defaults(func=cmd_aec)

    multibeam_p = subparsers.add_parser(
        "multibeam", help="Q4: whether the multi-beam output gives one stream per talker."
    )
    multibeam_p.add_argument("--talker-a-angle", type=float, required=True)
    multibeam_p.add_argument("--talker-b-angle", type=float, required=True)
    multibeam_p.set_defaults(func=cmd_multibeam)

    onset_p = subparsers.add_parser(
        "onset", help="Q5: the Silero onset delay, for the pre_roll_ms default."
    )
    onset_p.add_argument("--count", type=int, default=20)
    onset_p.add_argument("--vad-model", required=True)
    onset_p.add_argument(
        "--endpointing-ms",
        type=float,
        default=200.0,
        help="stt.endpointing_ms in config/config.example.yaml (line 427 when this plan was written).",
    )
    onset_p.set_defaults(func=cmd_onset)

    report_p = subparsers.add_parser(
        "report", help="Merge the newest JSON per subcommand into results/report.json."
    )
    report_p.set_defaults(func=cmd_report)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
