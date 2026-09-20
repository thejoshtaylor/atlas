#!/usr/bin/env python3
"""Measures the fully local provider set's own speed on this host (PROV-06,
D-12): the provisioned `faster-whisper` speech recognizer over a fixed
audio input, and the provisioned Piper synthesizer over a fixed sentence.

D-12 is explicit that this figure is published, not met: the reply-time
budget is already about seven times over on the cloud path, and this
project's host has no GPU. What is owed is the honest number, in the same
register `scripts/measure_turns.py` already established for VOICE-02 --
per-stage median, minimum, and maximum over several repetitions, and the
host it was taken on, because a number with no host attached is not a
published figure.

The fixed audio input is not a recorded human utterance -- no camera, no
microphone exists in every environment this script might run in, the same
limitation `docs/runbooks/wake-engine-corpus.md` states for the wake-word
corpus. Instead, this script's own fixed sentence is first synthesized by
the provisioned Piper voice, and that real synthesized speech (not
silence, not a tone) is what the provisioned `faster-whisper` model then
transcribes. Both stages run for real, against real model weights, on
this host -- only the "microphone" is Piper's own voice rather than a
recorded person's.

Both providers must already be provisioned (`scripts/fetch_models.py` /
`scripts/dev-fetch-models.sh`) before this script can measure anything --
it never downloads a model itself, and a missing model file is reported by
name and this script exits non-zero rather than substituting a smaller or
synthetic model.
"""

from __future__ import annotations

import argparse
import platform
import statistics
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import AsyncIterator

from spire_voice.config import ConfigError, load_config
from spire_voice.providers.boot import ProviderUnavailable
from spire_voice.providers.stt_faster_whisper import FasterWhisperStt
from spire_voice.providers.tts_piper import PiperTts
from spire_voice.providers.tts_xai import SinkFormat
from spire_voice.transports.base import SourceFormat

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"

_DEFAULT_REPETITIONS = 5
_MIN_REPETITIONS = 3

# A realistic assistant reply -- the shape of sentence Piper actually
# synthesizes in production, not a short greeting that would under-count
# real synthesis and decode work.
_FIXED_TEXT = (
    "Sure, I have turned on the kitchen lights. Tomorrow looks sunny with a "
    "high near seventy degrees, so you will not need a jacket."
)

# faster-whisper's own feature extractor expects 16 kHz mono PCM16 --
# `FasterWhisperStt.stream` applies no resample at all for `encoding="pcm"`
# (it assumes the source already is this rate, matching every browser
# transport in this codebase). The audio fed to it here is resampled to
# this exact rate by Piper's own `sink` parameter before it is ever
# handed to the recognizer, so the recognizer sees real-rate audio, not
# audio nominally mislabeled at the wrong rate.
_STT_SOURCE_FORMAT = SourceFormat(encoding="pcm", sample_rate=16000)
_STT_SINK_FORMAT = SinkFormat(codec="pcm", sample_rate=_STT_SOURCE_FORMAT.sample_rate)


class MeasurementError(Exception):
    """Raised before any repetition runs -- a provider that could not be
    constructed at all (a missing model file, most likely). Never raised
    partway through a repetition."""


@dataclass(frozen=True)
class StageResult:
    name: str
    durations_ms: "list[float]"

    @property
    def median_ms(self) -> "float | None":
        return statistics.median(self.durations_ms) if self.durations_ms else None

    @property
    def min_ms(self) -> "float | None":
        return min(self.durations_ms) if self.durations_ms else None

    @property
    def max_ms(self) -> "float | None":
        return max(self.durations_ms) if self.durations_ms else None


def cpu_brand() -> str:
    """A best-effort human-readable CPU name -- never required for the
    report to print, since `os.cpu_count()` (the number D-12 actually
    calls for) works regardless of whether this succeeds."""
    system = platform.system()
    try:
        if system == "Darwin":
            result = subprocess.run(
                ["sysctl", "-n", "machdep.cpu.brand_string"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            brand = result.stdout.strip()
            if brand:
                return brand
        elif system == "Linux":
            with open("/proc/cpuinfo", encoding="utf-8") as fh:
                for line in fh:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
    except (OSError, subprocess.SubprocessError):
        pass
    return platform.processor() or platform.machine() or "unknown CPU"


def host_description(cpu_count: "int | None") -> str:
    """D-12's own requirement: core count and whether an accelerator was
    used, because a number without the host it was taken on is not a
    published figure. Both providers measured here run `device="cpu"`
    (`stt_faster_whisper.py`) with no GPU code path at all, so "no
    accelerator" is a fact about the code, not an assumption about the
    host."""
    cores = cpu_count if cpu_count else "an unknown number of"
    return f"{cores}-core {cpu_brand()}, CPU only (no GPU/accelerator used)"


async def _frames_once(chunk: bytes) -> AsyncIterator[bytes]:
    yield chunk


async def measure_tts(tts: PiperTts, text: str, repetitions: int) -> StageResult:
    """Time `repetitions` full synthesis calls against the default
    (browser) sink -- the same sink shape a real turn actually uses,
    never a special-cased rate that would make this figure incomparable
    to what an operator's own turn experiences."""
    durations_ms = []
    for _ in range(repetitions):
        start = time.monotonic()
        await tts.synthesize_once(text)
        durations_ms.append((time.monotonic() - start) * 1000)
    return StageResult(name="text-to-speech (Piper)", durations_ms=durations_ms)


async def measure_stt(stt: FasterWhisperStt, audio: bytes, repetitions: int) -> StageResult:
    """Time `repetitions` full transcriptions of the same fixed audio --
    `stream()`'s own async generator is drained to completion each time,
    matching how a real turn consumes it."""
    durations_ms = []
    for _ in range(repetitions):
        start = time.monotonic()
        async for _event in stt.stream(_frames_once(audio), _STT_SOURCE_FORMAT):
            pass
        durations_ms.append((time.monotonic() - start) * 1000)
    return StageResult(name="speech-to-text (faster-whisper)", durations_ms=durations_ms)


async def run_measurement(
    stt_config, tts_config, *, repetitions: int, text: str = _FIXED_TEXT
) -> "list[StageResult]":
    """Construct both providers for real (a missing model raises
    `ProviderUnavailable`, translated to `MeasurementError` by the caller),
    synthesize the fixed sentence once at the recognizer's own rate to
    build the fixed audio input, then measure both stages."""
    tts = PiperTts(tts_config)
    stt = FasterWhisperStt(stt_config)

    tts_result = await measure_tts(tts, text, repetitions)
    audio_for_stt = await tts.synthesize_once(text, sink=_STT_SINK_FORMAT)
    if not audio_for_stt:
        raise MeasurementError(
            "Piper synthesized no audio for the fixed sentence -- cannot measure "
            "speech-to-text with no input to feed it"
        )
    stt_result = await measure_stt(stt, audio_for_stt, repetitions)
    return [stt_result, tts_result]


def _print_report(results: "list[StageResult]", cpu_count: "int | None") -> None:
    print(f"\nhost: {host_description(cpu_count)}")
    for result in results:
        if not result.durations_ms:
            print(f"  {result.name}: no repetitions completed")
            continue
        print(
            f"  {result.name}: median={result.median_ms:.0f}ms "
            f"min={result.min_ms:.0f}ms max={result.max_ms:.0f}ms "
            f"({len(result.durations_ms)} repetitions)"
        )


def _repetitions_type(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--repetitions must be an integer, got {raw!r}") from exc
    if value < _MIN_REPETITIONS:
        raise argparse.ArgumentTypeError(f"--repetitions must be at least {_MIN_REPETITIONS}, got {value}")
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the fully local provider set's own speed on this host (D-12) -- "
            "the provisioned faster-whisper model over synthesized speech, and the "
            "provisioned Piper voice over a fixed sentence. Both models must already be "
            "provisioned (scripts/fetch_models.py); this script never downloads one."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"the application config to read stt.*/tts.* from (default: {_DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--repetitions",
        type=_repetitions_type,
        default=_DEFAULT_REPETITIONS,
        help=f"repetitions per stage, at least {_MIN_REPETITIONS} (default: {_DEFAULT_REPETITIONS})",
    )
    return parser


def main(argv: "list[str] | None" = None) -> int:
    import asyncio
    import os

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except (ConfigError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        results = asyncio.run(
            run_measurement(config.stt, config.tts, repetitions=args.repetitions)
        )
    except ProviderUnavailable as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    except MeasurementError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_report(results, os.cpu_count())
    return 0


if __name__ == "__main__":
    sys.exit(main())
