#!/usr/bin/env python3
"""Scores both wake engines offline, over identical corpus audio (DBG-04).

Running both engines live would give each a different acoustic moment and
prove nothing about which is better (D-06). This harness decodes every
corpus recording's raw A-law bytes to 16 kHz mono PCM16 exactly once,
through `transports.camera.CameraAudioSource.decode_for_detector` -- the
same detector-only decode path the live pipeline uses, never a second,
independently derived resample -- and hands that identical decoded copy to
every engine under test.

**openWakeWord has no "hey spire" model in this environment** (RESEARCH.md
Pitfall 3): its adapter raises `WakeError` naming the missing path, which
this harness reports as a row with its cause named, never an absent row
and never a reason to drop it from the report. If only one engine can
actually run, the report says so plainly -- a one-horse race, not a
comparison.

**The corpus floor (D-07) is checked and reported as its own line**: at
least twenty positives across more than one distance and more than one
speaker, plus a negative run. Below the floor, this harness never prints a
recommendation the corpus does not support -- a number reported without
its caveat is how Phase 01 came to tick a requirement it had not measured
(STATE.md's own carried correction).

Reads `manifest.jsonl`, written by `scripts/capture_wake_corpus.py` under
the same shared filename convention (both scripts spell it out as a
literal, not a shared import -- `scripts/` is not a package). Optionally
also scores recorded session directories as additional negative material
(`--session-dir`, repeatable): real microphone bytes from an actual turn,
scored the same way corpus negatives are.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import av

from spire_voice.config import CameraConfig, VoskWakeConfig, OpenWakeWordConfig, load_config
from spire_voice.transports.camera import CameraAudioSource
from spire_voice.wake.base import WakeDetector, WakeError, WakeHit
from spire_voice.wake.openwakeword_engine import OpenWakeWordDetector
from spire_voice.wake.vosk_engine import VoskWakeDetector

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"
_DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "wake_corpus"

_MANIFEST_FILENAME = "manifest.jsonl"  # capture_wake_corpus.py writes this same filename

# D-07's stated floor: at least twenty positives, across more than one
# distance and more than one speaker, plus a negative run. Stated, not
# judged -- a corpus that falls short is reported as provisional, in the
# report's own words, never rounded up.
_FLOOR_POSITIVES = 20

# An arbitrary, documented chunk size (100ms at 16kHz mono 16-bit) -- both
# engines see byte-identical input regardless of where this boundary
# falls, since `pcm16` is decoded once and the same bytes object is sliced
# for each engine in turn.
_DETECTOR_CHUNK_BYTES = 3200

_THRESHOLD_SWEEP: tuple[float, ...] = (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)


# --- corpus manifest -------------------------------------------------------


@dataclass(frozen=True)
class CorpusRecording:
    """One recorded audio file plus its metadata -- `file` is always an
    absolute path by the time this dataclass is constructed, whether it
    came from `manifest.jsonl` (resolved against the corpus directory) or
    from a session directory (already absolute)."""

    file: str
    label: str
    distance: str | None
    speaker: str | None
    position_index: int | None
    global_index: int
    timestamp: str
    duration_s: float | None
    source: str | None
    note: str | None = None


def load_manifest(corpus_dir: Path) -> list[CorpusRecording]:
    """Read `manifest.jsonl` from `corpus_dir`. An absent manifest (no
    corpus recorded yet in this environment) is an empty corpus, not an
    error -- the floor check reports that plainly rather than this
    function raising."""
    manifest_path = corpus_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return []
    recordings: list[CorpusRecording] = []
    with manifest_path.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            recordings.append(
                CorpusRecording(
                    file=str((corpus_dir / raw["file"]).resolve()),
                    label=raw["label"],
                    distance=raw.get("distance"),
                    speaker=raw.get("speaker"),
                    position_index=raw.get("position_index"),
                    global_index=raw.get("global_index", 0),
                    timestamp=raw.get("timestamp", ""),
                    duration_s=raw.get("duration_s"),
                    source=raw.get("source"),
                    note=raw.get("note"),
                )
            )
    return recordings


def load_session_recordings(session_dirs: Sequence[Path]) -> list[CorpusRecording]:
    """Each session directory's own raw audio (`session/recorder.py`'s
    `audio.<encoding>`) becomes one additional negative recording -- real
    microphone bytes from an actual turn, scored the same way corpus
    negatives are. A session missing its audio file (no audio was
    recorded, or `session.record_audio` was off) is skipped, not an
    error."""
    recordings: list[CorpusRecording] = []
    for session_dir in session_dirs:
        audio_files = sorted(session_dir.glob("audio.*"))
        if not audio_files:
            continue
        recordings.append(
            CorpusRecording(
                file=str(audio_files[0].resolve()),
                label="negative",
                distance=None,
                speaker=None,
                position_index=None,
                global_index=-1,
                timestamp="",
                duration_s=_read_session_duration(session_dir),
                source="session-recording",
                note=f"turn recording from {session_dir.name}",
            )
        )
    return recordings


def _read_session_duration(session_dir: Path) -> float | None:
    """The session's own `timing.json` (`session/recorder.py`'s
    `TIMING_FILENAME`), summed across its stage durations -- this harness
    measures nothing itself; `timing.py` already did. Returns `None`,
    never a fabricated zero, if the file is absent or unreadable."""
    timing_path = session_dir / "timing.json"
    if not timing_path.exists():
        return None
    try:
        payload = json.loads(timing_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    stage_durations = payload.get("stage_durations_ms") or {}
    total_ms = sum(v for v in stage_durations.values() if isinstance(v, (int, float)))
    return (total_ms / 1000.0) if total_ms else None


# --- the corpus floor (D-07) ------------------------------------------------


@dataclass(frozen=True)
class FloorReport:
    positive_count: int
    distances: frozenset[str]
    speakers: frozenset[str]
    has_negative_run: bool
    negative_duration_s: float
    meets_floor: bool
    reasons: tuple[str, ...]


def check_corpus_floor(recordings: Sequence[CorpusRecording]) -> FloorReport:
    """D-07, checked explicitly and reported as a line of its own: the
    positive count against twenty, whether positives span more than one
    distance and more than one speaker, and whether a negative run
    exists. Below the floor, `meets_floor` is `False` and `reasons` names
    every shortfall -- never rounded up into a claim."""
    positives = [r for r in recordings if r.label == "positive"]
    negatives = [r for r in recordings if r.label == "negative"]
    distances = frozenset(r.distance for r in positives if r.distance)
    speakers = frozenset(r.speaker for r in positives if r.speaker)
    negative_duration = sum((r.duration_s or 0.0) for r in negatives)
    has_negative_run = negative_duration > 0.0

    reasons: list[str] = []
    if len(positives) < _FLOOR_POSITIVES:
        reasons.append(f"only {len(positives)} positive(s) recorded, floor is {_FLOOR_POSITIVES}")
    if len(distances) < 2:
        reasons.append(f"positives span {len(distances)} distance(s), floor requires more than one")
    if len(speakers) < 2:
        reasons.append(f"positives span {len(speakers)} speaker(s), floor requires more than one")
    if not has_negative_run:
        reasons.append("no negative run recorded")

    return FloorReport(
        positive_count=len(positives),
        distances=distances,
        speakers=speakers,
        has_negative_run=has_negative_run,
        negative_duration_s=negative_duration,
        meets_floor=not reasons,
        reasons=tuple(reasons),
    )


# --- engine construction, including the unavailable outcome ----------------


class EngineUnavailable:
    """Recorded in place of a constructed `WakeDetector` when its factory
    raises `WakeError` -- the engine still appears in the report, with its
    cause named, never dropped (D-08's own reporting discipline: an
    engine that cannot run is a reported outcome with a named cause,
    never an absent row)."""

    def __init__(self, reason: str) -> None:
        self.reason = reason


def build_engine(factory: Callable[[], WakeDetector]) -> WakeDetector | EngineUnavailable:
    try:
        return factory()
    except WakeError as exc:
        return EngineUnavailable(reason=str(exc))


# --- scoring arithmetic (pure, against scores -- never fabricated audio) ---


@dataclass(frozen=True)
class ThresholdRow:
    threshold: float
    positive_detections: int
    positive_total: int
    false_accepts_per_hour: float | None  # None when negative duration is zero -- unmeasurable, never fabricated


def detections_at_threshold(scores: Sequence[float | None], threshold: float) -> int:
    return sum(1 for score in scores if score is not None and score >= threshold)


def false_accepts_per_hour(
    negative_scores: Sequence[float | None], threshold: float, negative_duration_s: float
) -> float | None:
    if negative_duration_s <= 0:
        return None
    count = detections_at_threshold(negative_scores, threshold)
    return count / (negative_duration_s / 3600.0)


def threshold_sweep(
    positive_scores: Sequence[float | None],
    negative_scores: Sequence[float | None],
    negative_duration_s: float,
    thresholds: Sequence[float] = _THRESHOLD_SWEEP,
) -> tuple[ThresholdRow, ...]:
    """A sweep across candidate thresholds rather than a single verdict at
    the configured one -- a single operating point hides whether an
    engine is close to a better one."""
    return tuple(
        ThresholdRow(
            threshold=t,
            positive_detections=detections_at_threshold(positive_scores, t),
            positive_total=len(positive_scores),
            false_accepts_per_hour=false_accepts_per_hour(negative_scores, t, negative_duration_s),
        )
        for t in thresholds
    )


@dataclass(frozen=True)
class EngineReport:
    name: str
    available: bool
    unavailable_reason: str | None
    positive_scores: tuple[float | None, ...]
    negative_scores: tuple[float | None, ...]
    threshold_sweep: tuple[ThresholdRow, ...]


def score_detector_over_recording(
    detector: WakeDetector, pcm16: bytes, chunk_bytes: int = _DETECTOR_CHUNK_BYTES
) -> float | None:
    """Feed `pcm16` to `detector.process()` in fixed-size chunks, returning
    the highest `WakeHit.score` observed across the whole recording, or
    `None` if it never fired."""
    best: float | None = None
    for start in range(0, len(pcm16), chunk_bytes):
        hit: WakeHit | None = detector.process(pcm16[start : start + chunk_bytes])
        if hit is not None and (best is None or hit.score > best):
            best = hit.score
    return best


def score_engine_over_corpus(
    engine: WakeDetector | EngineUnavailable,
    name: str,
    recordings_pcm16: Sequence[tuple[CorpusRecording, bytes]],
) -> EngineReport:
    if isinstance(engine, EngineUnavailable):
        return EngineReport(
            name=name,
            available=False,
            unavailable_reason=engine.reason,
            positive_scores=(),
            negative_scores=(),
            threshold_sweep=(),
        )
    positive_scores: list[float | None] = []
    negative_scores: list[float | None] = []
    negative_duration = 0.0
    for recording, pcm16 in recordings_pcm16:
        # WR-02 fix (code review): every recording is an independent
        # utterance. Without this, a detector that accumulates state across
        # `process()` calls (Vosk's `KaldiRecognizer`, openWakeWord's own
        # frame buffer) could carry a still-open decode from one
        # recording's trailing audio into the next recording's very first
        # bytes -- attributing a hit, or a miss, to the wrong file, which
        # corrupts the threshold-sweep evidence DBG-04 exists to produce.
        engine.reset()
        score = score_detector_over_recording(engine, pcm16)
        if recording.label == "positive":
            positive_scores.append(score)
        elif recording.label == "negative":
            negative_scores.append(score)
            negative_duration += recording.duration_s or 0.0
    return EngineReport(
        name=name,
        available=True,
        unavailable_reason=None,
        positive_scores=tuple(positive_scores),
        negative_scores=tuple(negative_scores),
        threshold_sweep=threshold_sweep(positive_scores, negative_scores, negative_duration),
    )


def score_all_engines(
    engines: dict[str, WakeDetector | EngineUnavailable],
    recordings_pcm16: Sequence[tuple[CorpusRecording, bytes]],
) -> dict[str, EngineReport]:
    """Both (all) engines are scored over the exact same `recordings_pcm16`
    sequence -- decoded once by the caller, never re-decoded per engine
    (D-06's identical-bytes requirement)."""
    return {name: score_engine_over_corpus(engine, name, recordings_pcm16) for name, engine in engines.items()}


# --- decoding: raw A-law on disk -> 16kHz mono PCM16, exactly once each ----


class _NullSpeaker:
    async def write(self, chunk: bytes) -> None:
        return None


def _alaw_file_opener(path: Path) -> Callable[[CameraConfig], Any]:
    def _open(_config: CameraConfig) -> Any:
        return av.open(str(path), format="alaw", options={"sample_rate": "8000", "ar": "8000", "ac": "1"})

    return _open


async def _decode_one(path: Path) -> bytes:
    """Decode one recording's raw A-law file to 16 kHz mono PCM16 through
    `CameraAudioSource.decode_for_detector` -- the exact detector-only
    decode path the live pipeline uses, never a second, independently
    derived resample."""
    config = CameraConfig(rtsp_url="", encoding="alaw", sample_rate=8000)
    source = CameraAudioSource(config, _NullSpeaker(), open_container=_alaw_file_opener(path))
    source.start()
    out = bytearray()
    try:
        async for raw_chunk in source.frames():
            out += source.decode_for_detector(raw_chunk)
    finally:
        await source.close()
    return bytes(out)


async def decode_corpus(recordings: Sequence[CorpusRecording]) -> list[tuple[CorpusRecording, bytes]]:
    """Decode every recording exactly once -- the caller hands this same
    list to every engine (D-06)."""
    decoded: list[tuple[CorpusRecording, bytes]] = []
    for recording in recordings:
        pcm16 = await _decode_one(Path(recording.file))
        decoded.append((recording, pcm16))
    return decoded


# --- report printing ---------------------------------------------------


def _print_floor(floor: FloorReport) -> None:
    print(
        f"corpus floor (D-07): {floor.positive_count}/{_FLOOR_POSITIVES} positives, "
        f"{len(floor.distances)} distance(s), {len(floor.speakers)} speaker(s), "
        f"negative run: {'yes' if floor.has_negative_run else 'no'} "
        f"({floor.negative_duration_s:.1f}s)"
    )
    if floor.meets_floor:
        print("floor met.")
    else:
        print("floor NOT met -- this result is PROVISIONAL, not measured:")
        for reason in floor.reasons:
            print(f"  - {reason}")


def _print_reports(reports: dict[str, EngineReport]) -> None:
    for name, report in reports.items():
        print(f"\n--- {name} ---")
        if not report.available:
            print(f"could not run: {report.unavailable_reason}")
            continue
        detected = sum(1 for s in report.positive_scores if s is not None)
        print(f"positives detected (any threshold): {detected}/{len(report.positive_scores)}")
        for row in report.threshold_sweep:
            fa = "n/a" if row.false_accepts_per_hour is None else f"{row.false_accepts_per_hour:.2f}/hr"
            print(
                f"  threshold {row.threshold:.2f}: {row.positive_detections}/{row.positive_total} "
                f"positives, false accepts {fa}"
            )

    available = [r for r in reports.values() if r.available]
    if len(available) < 2:
        print(
            "\nonly one engine could run -- this is a one-horse race, not a comparison; "
            "no default is chosen from a score here."
        )


# --- CLI --------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Score both wake engines offline, over identical corpus audio (DBG-04) -- "
            "never live, which would give each engine a different acoustic moment."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"the application config to read wake.* from (default: {_DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_DEFAULT_CORPUS_DIR,
        help=f"the corpus directory capture_wake_corpus.py wrote (default: {_DEFAULT_CORPUS_DIR})",
    )
    parser.add_argument(
        "--session-dir",
        type=Path,
        action="append",
        default=[],
        dest="session_dirs",
        help="an additional recorded session directory to score as negative material; repeatable",
    )
    return parser


def _build_engines(wake_vosk: VoskWakeConfig, phrase: str, wake_oww: OpenWakeWordConfig) -> dict[str, Any]:
    return {
        "vosk": build_engine(lambda: VoskWakeDetector(wake_vosk, phrase)),
        "openwakeword": build_engine(lambda: OpenWakeWordDetector(wake_oww)),
    }


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    config = load_config(args.config)

    recordings = load_manifest(args.corpus_dir)
    recordings += load_session_recordings(args.session_dirs)

    floor = check_corpus_floor(recordings)
    _print_floor(floor)

    if not recordings:
        print("\nno corpus recorded yet -- nothing to score. Run capture_wake_corpus.py first.")
        return 0

    engines = _build_engines(config.wake.vosk, config.wake.phrase, config.wake.openwakeword)
    decoded = asyncio.run(decode_corpus(recordings))
    reports = score_all_engines(engines, decoded)

    _print_reports(reports)

    if not floor.meets_floor:
        print("\ncorpus is below its stated floor -- no tuned default is claimed from this run.")

    return 0


if __name__ == "__main__":
    sys.exit(main())
