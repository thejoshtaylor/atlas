#!/usr/bin/env python3
"""Bootstraps the wake-word evaluation corpus: RTSP in, raw A-law bytes to
disk, timestamped -- **no wake engine anywhere near it**.

That is the whole point (D-05). A threshold is needed to *detect*, not to
*record* -- a capture path with a detector in it would only record what
some threshold already accepted, which is the bootstrap problem restated
rather than solved. This module imports nothing from `atlas.wake`,
and never will.

**The recorded metadata shape (D-05, this plan's Task 1 checkpoint) is the
`manifest` option the operator chose:** the operator declares a distance
and a speaker once per position ("standing close", "me"), then repeats the
wake phrase; every recording under that position carries the declared
distance and speaker, a per-position index, a global index, and a
timestamp, written as one line to `manifest.jsonl` in the corpus
directory. Moving to a new position is an explicit re-declaration, never
inferred.

`manifest`'s own named failure mode is moving without re-declaring, after
which every subsequent recording is silently mislabeled and nothing in the
audio reveals it. This module's defense is not a debug flag: before every
single capture, positive or negative, it prints the position label (or
"negative run") it is about to record under, and for positive captures the
running count already recorded under that exact position -- both always
on, so the floor (D-07: at least twenty positives across distances and
speakers) is something the operator can watch while recording, not
something discovered short afterward.

Follows `scripts/measure_turns.py`'s conventions: this script reads no
environment variable directly and prints no credential value, length, or
prefix. Camera connectivity comes from `atlas.config.load_config`,
the one place `${TAPO_USER}`/`${TAPO_PASSWORD}` are ever expanded from the
environment -- run this script through `scripts/dev-capture-corpus.sh`,
which sources `.env` the way `dev-run.sh`/`dev-measure.sh` do. The RTSP URL
itself carries the camera's credentials inline (`transports/camera.py`'s
own stated invariant): it is never printed, logged, or written to an
exception message or to the corpus metadata. A `--source-name` (e.g.
"back-camera") is recorded in its place.

The packet-level read is `transports.camera.CameraAudioSource`, reused
directly rather than re-derived: a corpus captured through a second,
subtly different RTSP path would score the engines on audio the running
system would never actually see.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from atlas.config import CameraConfig, ConfigError, load_config
from atlas.transports.camera import CameraAudioSource

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG_PATH = _REPO_ROOT / "config" / "config.example.yaml"
_DEFAULT_CORPUS_DIR = _REPO_ROOT / "data" / "wake_corpus"

_MANIFEST_FILENAME = "manifest.jsonl"  # score_wake_engines.py reads this same filename

# D-07's stated floor -- printed in the running count and the end-of-run
# summary so the operator can watch progress against it, never enforced or
# judged here (that is score_wake_engines.py's job, against the recorded
# manifest, not this script's).
_FLOOR_POSITIVES = 20

_DEFAULT_UTTERANCE_SECONDS = 2.5
_DEFAULT_NEGATIVE_SECONDS = 60.0


class CorpusCaptureError(ValueError):
    """Raised before any recording starts -- an unwritable corpus
    directory or a blank source name names itself here, rather than
    surfacing after the operator has already said the phrase ten times."""


class _NullSpeaker:
    """`CameraAudioSource` takes a speaker sink for reply audio it never
    sends here -- this capture mode has no reply, only a microphone."""

    async def write(self, chunk: bytes) -> None:
        return None


def validate_environment(corpus_dir: Path, source_name: str) -> None:
    """Validate everything checkable before opening the RTSP stream:
    the source name is non-blank (it stands in for the URL in every
    recorded metadata line), the corpus directory is coverable by the
    repository's ignore rules, and it is actually writable. Raises
    `CorpusCaptureError` naming what failed -- discovering an unwritable
    directory after the operator has already recorded ten utterances is
    the same lost evening this whole plan exists to prevent.
    """
    if not source_name.strip():
        raise CorpusCaptureError(
            "--source-name must be non-blank -- it is recorded in the corpus "
            "metadata in place of the RTSP URL, which carries this camera's "
            "credentials inline and must never be written to disk"
        )
    _ensure_gitignored(corpus_dir)
    probe = corpus_dir / ".write-check"
    try:
        corpus_dir.mkdir(parents=True, exist_ok=True)
        probe.write_text("")
        probe.unlink(missing_ok=True)
    except OSError as exc:
        raise CorpusCaptureError(
            f"corpus directory is not writable: {corpus_dir} ({type(exc).__name__})"
        ) from exc


def _ensure_gitignored(corpus_dir: Path) -> None:
    """Confirm `corpus_dir` is covered by the repository's ignore rules
    before the first write, extending `.gitignore` with a comment if it is
    not -- the same ordering plan 01.1-07 used for its WAV fixture and
    plan 02-01 used for the session store. This is audio of a real home
    and whoever was speaking in it, in a public repository.

    The default corpus directory (`data/wake_corpus`) already falls under
    the existing `data/` rule; this only ever writes a new line when
    `--corpus-dir` points somewhere that rule does not reach.
    """
    resolved = corpus_dir.resolve()
    try:
        rel = resolved.relative_to(_REPO_ROOT)
    except ValueError:
        return  # outside the repository entirely -- nothing for git to track or ignore

    try:
        result = subprocess.run(
            ["git", "check-ignore", "--quiet", str(rel)],
            cwd=_REPO_ROOT,
            timeout=10,
        )
        already_ignored = result.returncode == 0
    except (OSError, subprocess.SubprocessError):
        already_ignored = False  # cannot confirm -- extend defensively rather than risk a tracked write

    if already_ignored:
        return

    gitignore_path = _REPO_ROOT / ".gitignore"
    with gitignore_path.open("a", encoding="utf-8") as fh:
        fh.write(
            "\n# Wake-word corpus (plan 02-09, added automatically by "
            "capture_wake_corpus.py): audio of a real home and whoever was\n"
            "# speaking in it, recorded to bootstrap the wake-engine score. "
            "Never committed -- this repository is public.\n"
        )
        fh.write(f"{rel.as_posix()}/\n")


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _append_manifest(corpus_dir: Path, entry: dict[str, Any]) -> None:
    with (corpus_dir / _MANIFEST_FILENAME).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")


def _next_global_index(corpus_dir: Path) -> int:
    manifest_path = corpus_dir / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return 0
    with manifest_path.open(encoding="utf-8") as fh:
        return sum(1 for line in fh if line.strip())


async def _drain_for(frames_iter: Any, duration_s: float) -> tuple[bytes, float]:
    """Collect raw packet bytes from `frames_iter` for up to `duration_s`
    wall-clock seconds -- timed externally, never by decoding audio
    content, since this script holds no codec and no wake engine. Returns
    the bytes and the actual elapsed time, which may be shorter than
    `duration_s` if the source ends first or the operator interrupts.
    """
    buf = bytearray()
    start = time.monotonic()
    deadline = start + duration_s
    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                chunk = await asyncio.wait_for(frames_iter.__anext__(), timeout=remaining)
            except asyncio.TimeoutError:
                break
            except StopAsyncIteration:
                break
            buf += chunk
    except KeyboardInterrupt:
        pass  # the operator ended this recording early -- keep what was captured, not nothing
    return bytes(buf), time.monotonic() - start


async def run_positive_session(
    frames_iter: Any,
    corpus_dir: Path,
    source_name: str,
    utterance_seconds: float,
    global_index: int,
) -> int:
    """Interactive loop: declare a position once, then repeat utterances
    under it until 'n' (new position) or 'q' (finish). Prints the position
    label and its running count before every single capture, always --
    never a debug flag -- because a recording made after the operator
    moved without re-declaring is silently mislabeled otherwise, and
    nothing in the audio reveals it (manifest's own named failure mode).
    """
    position_counts: dict[tuple[str, str], int] = {}
    while True:
        distance = input(
            "\nNew position -- distance (e.g. 'close', 'across the room'), or 'q' to finish: "
        ).strip()
        if not distance or distance.lower() == "q":
            return global_index
        speaker = input("Speaker for this position (e.g. a name): ").strip()
        position = (distance, speaker)
        while True:
            recorded_so_far = position_counts.get(position, 0)
            print(f"position: {distance} / {speaker}")
            print(f"{distance}/{speaker}: {recorded_so_far} recorded")
            action = input("Enter to record, 'n' for a new position, 'q' to finish: ").strip().lower()
            if action == "n":
                break
            if action == "q":
                return global_index
            print(f"recording for {utterance_seconds:.1f}s -- say the phrase now")
            audio, elapsed = await _drain_for(frames_iter, utterance_seconds)
            if not audio:
                print("no audio captured -- check the camera connection; not recorded", file=sys.stderr)
                continue
            filename = f"positive_{global_index:04d}_{_timestamp()}.alaw"
            (corpus_dir / filename).write_bytes(audio)
            position_counts[position] = recorded_so_far + 1
            _append_manifest(
                corpus_dir,
                {
                    "file": filename,
                    "label": "positive",
                    "distance": distance,
                    "speaker": speaker,
                    "position_index": recorded_so_far + 1,
                    "global_index": global_index,
                    "timestamp": _timestamp(),
                    "duration_s": elapsed,
                    "source": source_name,
                },
            )
            global_index += 1


async def run_negative_session(
    frames_iter: Any,
    corpus_dir: Path,
    source_name: str,
    seconds: float,
    note: str,
    global_index: int,
) -> int:
    """One continuous negative run: ordinary room sound, the television,
    and speech that is not the wake phrase. Ctrl-C ends it early and
    records what was actually captured, never nothing.
    """
    print("recording a continuous negative run -- no wake phrase, no engine, just ambient sound")
    print(f"negative run (up to {seconds:.0f}s) -- Ctrl-C ends it early and keeps what was captured")
    audio, elapsed = await _drain_for(frames_iter, seconds)
    if not audio:
        print("no audio captured -- nothing recorded", file=sys.stderr)
        return global_index
    filename = f"negative_{global_index:04d}_{_timestamp()}.alaw"
    (corpus_dir / filename).write_bytes(audio)
    _append_manifest(
        corpus_dir,
        {
            "file": filename,
            "label": "negative",
            "distance": None,
            "speaker": None,
            "position_index": None,
            "global_index": global_index,
            "timestamp": _timestamp(),
            "duration_s": elapsed,
            "source": source_name,
            "note": note,
        },
    )
    return global_index + 1


def _print_summary(corpus_dir: Path) -> None:
    """Name the corpus directory, the positive count, and the total
    negative duration -- so the operator can see whether D-07's floor has
    been reached without opening the directory. Never prints audio content
    or a transcript.
    """
    manifest_path = corpus_dir / _MANIFEST_FILENAME
    positives = 0
    negative_duration = 0.0
    if manifest_path.exists():
        with manifest_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)
                if entry.get("label") == "positive":
                    positives += 1
                elif entry.get("label") == "negative":
                    negative_duration += float(entry.get("duration_s") or 0.0)
    print(f"\ncorpus directory: {corpus_dir}")
    print(f"positives recorded: {positives} (floor: {_FLOOR_POSITIVES})")
    print(f"negative duration recorded: {negative_duration:.1f}s")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Record the wake-word evaluation corpus with no wake engine in the loop at "
            "all (D-05) -- RTSP in, raw A-law to disk, timestamped."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG_PATH,
        help=f"the application config to read camera.* from (default: {_DEFAULT_CONFIG_PATH})",
    )
    parser.add_argument(
        "--corpus-dir",
        type=Path,
        default=_DEFAULT_CORPUS_DIR,
        help=f"where recordings and manifest.jsonl are written (default: {_DEFAULT_CORPUS_DIR})",
    )
    parser.add_argument(
        "--source-name",
        default="camera",
        help="a name for the audio source, recorded in the corpus metadata -- never the RTSP URL",
    )
    subparsers = parser.add_subparsers(dest="mode", required=True)

    positive = subparsers.add_parser(
        "positive", help="record wake-phrase utterances under a declared distance and speaker"
    )
    positive.add_argument(
        "--utterance-seconds",
        type=float,
        default=_DEFAULT_UTTERANCE_SECONDS,
        help=f"recording window per utterance (default: {_DEFAULT_UTTERANCE_SECONDS})",
    )

    negative = subparsers.add_parser(
        "negative", help="record one continuous negative run (ambient sound, the television)"
    )
    negative.add_argument(
        "--seconds",
        type=float,
        default=_DEFAULT_NEGATIVE_SECONDS,
        help=f"requested duration; Ctrl-C ends it early (default: {_DEFAULT_NEGATIVE_SECONDS})",
    )
    negative.add_argument(
        "--note",
        default="",
        help="free-text note describing this run's content (e.g. 'television + ambient talk')",
    )

    return parser


def _open_camera(config: CameraConfig, speaker: _NullSpeaker) -> CameraAudioSource:
    return CameraAudioSource(config, speaker)


async def _run(args: argparse.Namespace, camera_config: CameraConfig) -> None:
    source = _open_camera(camera_config, _NullSpeaker())
    source.start()
    frames_iter = source.frames()
    try:
        global_index = _next_global_index(args.corpus_dir)
        if args.mode == "positive":
            await run_positive_session(
                frames_iter, args.corpus_dir, args.source_name, args.utterance_seconds, global_index
            )
        else:
            await run_negative_session(
                frames_iter, args.corpus_dir, args.source_name, args.seconds, args.note, global_index
            )
    finally:
        await source.close()


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        validate_environment(args.corpus_dir, args.source_name)
        config = load_config(args.config)
    except (CorpusCaptureError, ConfigError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    try:
        asyncio.run(_run(args, config.camera))
    except KeyboardInterrupt:
        print("\ninterrupted -- recording what was captured so far", file=sys.stderr)

    _print_summary(args.corpus_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
