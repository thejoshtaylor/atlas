"""Behavior tests for the D-05 spike's Pi-side corpus recorder
(11-05-PLAN.md, Task 1). Importing `speaker_corpus` here at all is itself
part of the proof: no `sounddevice`, `sherpa_onnx`, or `atlas_edge.vad`
import happens at module load, so this file runs with no XVF3800 attached
and no hardware library installed.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

import speaker_corpus


def test_validate_label_accepts_lowercase_tokens():
    assert speaker_corpus.validate_label("member-a") == "member-a"
    assert speaker_corpus.validate_label("other") == "other"


def test_validate_label_refuses_a_real_name():
    with pytest.raises(ValueError):
        speaker_corpus.validate_label("Josh")


def test_validate_label_refuses_empty_label():
    with pytest.raises(ValueError):
        speaker_corpus.validate_label("")


def test_validate_label_refuses_a_33_character_label():
    with pytest.raises(ValueError):
        speaker_corpus.validate_label("a" * 33)


def test_validate_label_accepts_a_32_character_label():
    assert speaker_corpus.validate_label("a" * 32) == "a" * 32


def test_prompts_for_enrollment_returns_the_five_phrases():
    prompts = speaker_corpus.prompts_for("enrollment", count=1)
    assert prompts == speaker_corpus.ENROLLMENT_PROMPTS
    assert len(prompts) == 5


def test_prompts_for_command_starts_every_prompt_with_hey_atlas():
    prompts = speaker_corpus.prompts_for("command", count=1)
    assert len(prompts) >= 12
    assert all(p.startswith("hey atlas,") for p in prompts)


def test_prompts_for_reply_returns_the_reply_list():
    prompts = speaker_corpus.prompts_for("reply", count=1)
    assert prompts == speaker_corpus.REPLY_PROMPTS


def test_prompts_for_other_returns_count_none_prompts():
    prompts = speaker_corpus.prompts_for("other", count=3)
    assert prompts == (None, None, None)


def test_prompts_for_unknown_kind_raises():
    with pytest.raises(ValueError):
        speaker_corpus.prompts_for("bogus", count=1)


def _stereo_clip(seconds: float, sample_rate: int = 16000) -> "np.ndarray":
    n = int(seconds * sample_rate)
    return np.zeros((n, 2), dtype=np.int16)


def test_write_clip_writes_wav_and_manifest_line(tmp_path):
    root = tmp_path / "speakers"
    samples = _stereo_clip(1.0)
    # 62 frames of 256 samples = 15872 samples < 16000; flag every other frame speech.
    speech_flags = [i % 2 == 0 for i in range(62)]

    clip_path = speaker_corpus.write_clip(
        root, "member-a", "command", "hey atlas, what time is it", samples, speech_flags, 16000
    )

    assert clip_path == root / "member-a" / "command-01.wav"
    assert clip_path.is_file()

    manifest_path = root / "manifest.jsonl"
    lines = manifest_path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["label"] == "member-a"
    assert entry["kind"] == "command"
    assert entry["prompt"] == "hey atlas, what time is it"
    assert entry["file"] == "member-a/command-01.wav"
    assert entry["seconds"] == pytest.approx(1.0)
    # Each True/False pair is one 16ms frame; runs alternate starting True.
    assert entry["speech_intervals_ms"][0] == [0.0, 16.0]
    assert entry["speech_intervals_ms"][1] == [32.0, 48.0]


def test_write_clip_uses_the_next_free_index_per_label_and_kind(tmp_path):
    root = tmp_path / "speakers"
    samples = _stereo_clip(0.5)
    flags = [True] * 10

    first = speaker_corpus.write_clip(root, "member-a", "command", "p1", samples, flags, 16000)
    second = speaker_corpus.write_clip(root, "member-a", "command", "p2", samples, flags, 16000)
    other_kind = speaker_corpus.write_clip(root, "member-a", "reply", "yes", samples, flags, 16000)

    assert first.name == "command-01.wav"
    assert second.name == "command-02.wav"
    assert other_kind.name == "reply-01.wav"

    lines = (root / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3


def test_write_clip_rejects_an_invalid_label(tmp_path):
    root = tmp_path / "speakers"
    samples = _stereo_clip(0.5)
    with pytest.raises(ValueError):
        speaker_corpus.write_clip(root, "Josh", "command", "p", samples, [True], 16000)


def test_build_parser_lists_record_subcommand():
    parser = speaker_corpus.build_parser()
    args = parser.parse_args(["record", "--label", "member-a", "--kind", "enrollment", "--vad-model", "x.onnx"])
    assert args.command == "record"
    assert args.label == "member-a"
    assert args.kind == "enrollment"
    assert args.seconds is None
    assert args.out == "spike/results/speakers"
    assert args.device_name == "reSpeaker"


def test_build_parser_requires_a_command():
    parser = speaker_corpus.build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args([])


def test_no_hardware_name_bound_at_module_scope():
    # `import speaker_corpus` above already ran with no XVF3800 attached --
    # the real proof. This also checks the module namespace directly: a
    # hardware import at module scope would bind these names here.
    assert "sounddevice" not in vars(speaker_corpus)
    assert "sherpa_onnx" not in vars(speaker_corpus)
    assert not hasattr(speaker_corpus, "SileroGate")
