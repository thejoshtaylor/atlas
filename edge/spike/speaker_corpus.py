#!/usr/bin/env python3
"""The D-05 spike's Pi-side corpus recorder: one `record` subcommand that
prompts an enrollment, command, reply, or "other" phrase, captures it
through the real XVF3800, and appends it to a labeled corpus
(`11-05-PLAN.md`).

Run on the Pi, inside `edge/`, after stopping the `atlas-edge` service (the
array opens once, for this process's own life):

    PYTHONPATH=src:spike uv run python spike/speaker_corpus.py \\
        --label member-a --kind enrollment --vad-model models/silero_vad.onnx

`--label` is a token such as `member-a`, matching `^[a-z0-9-]{1,32}$` --
never a real name (T-11-16). The label `"other"` marks a non-household
voice (a television, a radio, a podcast).

`sounddevice`, `atlas_edge.vad`, and `run_spike` (this directory's own
hardware-capture helpers) are imported inside `cmd_record` only, never at
module load -- `--help` and this file's own test suite both run on a dev
host with no XVF3800 attached. This module never imports the server's
`atlas` package (only `atlas_edge`, the Pi's own package, is allowed) --
`tests/test_speaker_spike.py`'s contract test pins `ENROLLMENT_PROMPTS`
below to `atlas.speaker_id.phrases.ENROLLMENT_PHRASES` from the server
side, so the two lists can never quietly drift apart.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
import wave
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# 256 samples at 16 kHz is 16 ms -- the same frame size `atlas_edge.capture`
# feeds the live VAD gate in production (FRAME_SAMPLES, capture.py), so a
# recorded clip's speech_intervals_ms lands on the exact frame grid the
# server-side scorer expects.
FRAME_SAMPLES = 256
SAMPLE_RATE = 16000

_LABEL_RE = re.compile(r"^[a-z0-9-]{1,32}$")

# A restatement of atlas.speaker_id.phrases.ENROLLMENT_PHRASES -- this
# module never imports atlas (the Pi never does), so the two lists are
# pinned together by a contract test on the server side instead of a
# shared import.
ENROLLMENT_PROMPTS: "tuple[str, ...]" = (
    "The morning light comes through the kitchen window.",
    "A quiet street holds more sound than it first seems.",
    "Every good recipe starts with a clean counter.",
    "The garden needs water at least twice this week.",
    "Reading a little before bed makes the house feel calm.",
)

# At least twelve harmless commands (11-05-PLAN.md), each starting "hey
# atlas," and naming no real entity -- a generic room or device name only.
COMMAND_PROMPTS: "tuple[str, ...]" = (
    "hey atlas, what time is it",
    "hey atlas, turn on the kitchen light",
    "hey atlas, turn off the kitchen light",
    "hey atlas, what's the weather today",
    "hey atlas, set a timer for five minutes",
    "hey atlas, play some music",
    "hey atlas, pause the music",
    "hey atlas, what's on my calendar today",
    "hey atlas, turn down the volume",
    "hey atlas, turn up the volume",
    "hey atlas, is the front door locked",
    "hey atlas, how warm is it outside",
)

REPLY_PROMPTS: "tuple[str, ...]" = ("yes", "no", "cancel", "yes please", "that's right")


def validate_label(label: str) -> str:
    """Return `label` unchanged if it matches `^[a-z0-9-]{1,32}$`. Raises
    `ValueError` otherwise -- a label is a token, never a real name."""
    if not _LABEL_RE.match(label):
        raise ValueError(
            f"invalid label {label!r} -- must match {_LABEL_RE.pattern!r} "
            "(a lowercase token such as 'member-a', never a real name)"
        )
    return label


def prompts_for(kind: str, count: "int | None") -> "tuple[Any, ...]":
    """The prompts one `record` run reads: the fixed five enrollment
    phrases for `"enrollment"`, the fixed command list for `"command"`,
    the fixed reply list for `"reply"`, or `count` `None` placeholders for
    `"other"` (a non-scripted voice, one clip per placeholder)."""
    if kind == "enrollment":
        return ENROLLMENT_PROMPTS
    if kind == "command":
        return COMMAND_PROMPTS
    if kind == "reply":
        return REPLY_PROMPTS
    if kind == "other":
        return tuple([None] * (count or 0))
    raise ValueError(f"prompts_for: unknown kind {kind!r}")


def _write_wav(path: Path, buffer: "np.ndarray", sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n_channels = buffer.shape[1] if buffer.ndim == 2 else 1
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(n_channels)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(np.ascontiguousarray(buffer).astype(np.int16).tobytes())


def _speech_intervals_ms(speech_flags: "Sequence[bool]", frame_ms: float) -> "list[list[float]]":
    """The runs of `True` in `speech_flags`, each expressed in ms from the
    clip's start -- one flag per `frame_ms`-long frame."""
    intervals: "list[list[float]]" = []
    start: "int | None" = None
    for i, flag in enumerate(speech_flags):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            intervals.append([start * frame_ms, i * frame_ms])
            start = None
    if start is not None:
        intervals.append([start * frame_ms, len(speech_flags) * frame_ms])
    return intervals


def write_clip(
    root: Path,
    label: str,
    kind: str,
    prompt: "str | None",
    samples: "np.ndarray",
    speech_flags: "Sequence[bool]",
    sample_rate: int,
) -> Path:
    """Write `samples` (shape `(n, channels)`, int16) to
    `<root>/<label>/<kind>-<nn>.wav` at the next free index, and append one
    manifest line to `<root>/manifest.jsonl` naming its speech intervals
    (16 ms per frame, per `speech_flags`)."""
    validate_label(label)
    label_dir = root / label
    label_dir.mkdir(parents=True, exist_ok=True)

    next_index = 1
    for existing in label_dir.glob(f"{kind}-*.wav"):
        suffix = existing.stem.rsplit("-", 1)[-1]
        if suffix.isdigit():
            next_index = max(next_index, int(suffix) + 1)
    clip_path = label_dir / f"{kind}-{next_index:02d}.wav"
    _write_wav(clip_path, samples, sample_rate)

    frame_ms = FRAME_SAMPLES * 1000.0 / sample_rate
    n_samples = samples.shape[0] if samples.ndim else len(samples)
    manifest_entry = {
        "label": label,
        "kind": kind,
        "prompt": prompt,
        "file": str(clip_path.relative_to(root)),
        "seconds": n_samples / sample_rate,
        "speech_intervals_ms": _speech_intervals_ms(speech_flags, frame_ms),
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    with (root / "manifest.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest_entry) + "\n")
    return clip_path


def cmd_record(args: argparse.Namespace) -> int:
    """Prompt, capture, and write every clip `prompts_for` names for this
    `--label`/`--kind`. Every hardware import lives here, never at module
    scope."""
    import run_spike  # this directory's own capture helpers (D-04)
    from atlas_edge.vad import SileroGate  # noqa: PLC0415 -- lazy, see module docstring

    validate_label(args.label)
    sd = run_spike._array_sounddevice(args.device_name)
    root = Path(args.out)
    seconds = args.seconds if args.seconds is not None else (8.0 if args.kind == "other" else 6.0)
    prompts = prompts_for(args.kind, args.count)

    for i, prompt in enumerate(prompts, start=1):
        if prompt is not None:
            print(f"[{i}/{len(prompts)}] say: {prompt!r}")
        else:
            print(f"[{i}/{len(prompts)}] speak now (other voice)")
        time.sleep(1.0)
        buffer = run_spike._capture(sd, seconds, SAMPLE_RATE)

        gate = SileroGate(args.vad_model)
        mono = buffer[:, 0]
        speech_flags: "list[bool]" = []
        for start in range(0, len(mono) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
            frame = np.ascontiguousarray(mono[start : start + FRAME_SAMPLES]).astype("<i2").tobytes()
            speech_flags.append(bool(gate.push(frame)))

        clip_path = write_clip(root, args.label, args.kind, prompt, buffer, speech_flags, SAMPLE_RATE)
        print(f"  wrote {clip_path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="D-05 spike CLI: record a labeled speaker corpus on the real edge microphone."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    record_p = subparsers.add_parser("record", help="Record one label/kind's clips into the corpus.")
    record_p.add_argument("--label", required=True)
    record_p.add_argument("--kind", required=True, choices=("enrollment", "command", "reply", "other"))
    record_p.add_argument("--count", type=int, default=1, help="Clip count for --kind other; ignored otherwise.")
    record_p.add_argument("--seconds", type=float, default=None, help="Default: 6, or 8 for --kind other.")
    record_p.add_argument("--vad-model", required=True, dest="vad_model")
    record_p.add_argument("--out", default="spike/results/speakers")
    record_p.add_argument("--device-name", default="reSpeaker", dest="device_name")
    record_p.set_defaults(func=cmd_record)

    return parser


def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    sys.exit(main())
