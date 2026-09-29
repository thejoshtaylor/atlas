#!/usr/bin/env python3
"""The D-05/D-06/D-08/D-09/D-12 spike scorer: turns a recorded speaker
corpus into one JSON report with a coded pass/fail rule per question
(11-05-PLAN.md). Every `question_q*` function takes already-computed
inputs -- no audio, no model -- so each rule is tested in isolation.
Order matters (`<interfaces>`, 11-05-PLAN.md): Q1 picks the channel and Q6
the noise floor before the embedding questions (Q2 onward), since both
feed the shared windowing/matching pipeline.

Run on the dev host after `edge/spike/speaker_corpus.py record` builds a
corpus at `edge/spike/results/speakers/` (gitignored). `atlas` is not an
installed package -- like every script here, this needs `PYTHONPATH=src`:

    PYTHONPATH=src .venv/bin/python scripts/speaker_spike.py score \\
        --corpus edge/spike/results/speakers --model-dir models/speaker-id
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from atlas.audio.energy import rms_amplitude
from atlas.config import SPEAKER_MODEL_FILES
from atlas.speaker_id import evaluation
from atlas.speaker_id.change_point import find_change_points
from atlas.speaker_id.matching import l2_normalize, mean_embedding
from atlas.speaker_id.windows import SpeechWindowAccumulator

FRAME_SAMPLES = 256  # matches edge/spike/speaker_corpus.py's own frame size
SAMPLE_RATE = 16000
_MANIFEST_FILENAME = "manifest.jsonl"
_BLOCKING_QUESTIONS = ("q1", "q2", "q5", "q6", "q7")
_DEFAULT_WINDOW_OPTIONS: "tuple[int, ...]" = (500, 750, 1000)
_DEFAULT_FLOOR_GRID: "tuple[float, ...]" = tuple(round(0.05 * i, 2) for i in range(20))

# Q6 noise-frame margins (operator diagnosis, 2026-09-29, real 11-08 house corpus:
# unfiltered noise frames gave a 10x-too-high floor, 0.051 vs. the corrected 0.0033/0.0052).
_NOISE_INTERVAL_MARGIN_MS = 500.0  # excludes frames this close to a VAD interval's edge -- Silero's own onset/offset lag leaks real speech energy just outside the interval it flagged.
_NOISE_CLIP_ONSET_MS = 500.0  # excludes each clip's own first 500ms outright -- the ~300ms Silero onset lag means real speech can precede the first flagged interval.

@dataclass(frozen=True)
class CorpusClip:
    label: str
    kind: str
    prompt: "str | None"
    file: Path
    seconds: float
    speech_intervals_ms: "list[list[float]]"

def load_corpus(root: Path) -> "list[CorpusClip]":
    """Read `<root>/manifest.jsonl`. An absent manifest is an empty corpus."""
    manifest_path = root / _MANIFEST_FILENAME
    if not manifest_path.exists():
        return []
    clips: "list[CorpusClip]" = []
    for line in manifest_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        raw = json.loads(line)
        clips.append(CorpusClip(
            label=raw["label"], kind=raw["kind"], prompt=raw.get("prompt"), file=(root / raw["file"]).resolve(),
            seconds=raw["seconds"], speech_intervals_ms=raw.get("speech_intervals_ms", []),
        ))
    return clips

def _read_wav(path: Path) -> "np.ndarray":
    with wave.open(str(path), "rb") as wav_file:
        n_channels = wav_file.getnchannels()
        raw = wav_file.readframes(wav_file.getnframes())
    samples = np.frombuffer(raw, dtype="<i2")
    return samples.reshape(-1, n_channels) if n_channels > 1 else samples.reshape(-1, 1)

def _ceil_to(value: float, step: float) -> float:
    return round(math.ceil(round(value / step, 9)) * step, 6)

def _frame_in_intervals(start_ms: float, end_ms: float, intervals_ms: "Sequence[Sequence[float]]") -> bool:
    return any(iv[0] <= start_ms < iv[1] for iv in intervals_ms)

def _frame_is_room_noise(start_ms: float, end_ms: float, intervals_ms: "Sequence[Sequence[float]]") -> bool:
    """Q6's noise-frame filter: past the clip's own onset margin, and outside every interval's own margin (operator diagnosis, 2026-09-29)."""
    if start_ms < _NOISE_CLIP_ONSET_MS:
        return False
    return not any(iv[0] - _NOISE_INTERVAL_MARGIN_MS < end_ms and start_ms < iv[1] + _NOISE_INTERVAL_MARGIN_MS for iv in intervals_ms)

def _channel_pcm16(clip: CorpusClip, channel: int) -> bytes:
    samples = _read_wav(clip.file)
    col = channel if samples.shape[1] > channel else 0
    return np.ascontiguousarray(samples[:, col]).astype("<i2").tobytes()

def embed_windows(
    pcm16: bytes, intervals_ms: "Sequence[Sequence[float]]", embedder: Any, *, window_ms: int, speech_rms_floor: float, sample_rate: int = SAMPLE_RATE
) -> "list[np.ndarray]":
    """Window `pcm16`'s speech-interval frames and embed each closed window."""
    acc = SpeechWindowAccumulator(sample_rate=sample_rate, window_ms=window_ms, min_window_ms=min(250, window_ms), speech_rms_floor=speech_rms_floor)
    acc.start_segment(0)
    samples = np.frombuffer(pcm16, dtype="<i2")
    frame_ms = FRAME_SAMPLES * 1000.0 / sample_rate
    embeddings: "list[np.ndarray]" = []
    frame_index = 0
    for start in range(0, len(samples) - FRAME_SAMPLES + 1, FRAME_SAMPLES):
        frame_start_ms = frame_index * frame_ms
        if _frame_in_intervals(frame_start_ms, frame_start_ms + frame_ms, intervals_ms):
            frame_bytes = np.ascontiguousarray(samples[start : start + FRAME_SAMPLES]).astype("<i2").tobytes()
            window = acc.push(frame_bytes, frame_index=frame_index, at=frame_start_ms)
            if window is not None:
                embeddings.append(embedder.embed(window.pcm))
        frame_index += 1
    tail = acc.end_segment()
    if tail is not None:
        embeddings.append(embedder.embed(tail.pcm))
    return embeddings

def _trim_with_margin(pcm16: bytes, intervals_ms: "Sequence[Sequence[float]]", margin_ms: float, sample_rate: int) -> bytes:
    """The enrollment embedding input: the clip trimmed to its outer speech-interval bounds, widened by `margin_ms` on each side."""
    samples = np.frombuffer(pcm16, dtype="<i2")
    if not intervals_ms:
        return pcm16
    start_ms = min(iv[0] for iv in intervals_ms) - margin_ms
    end_ms = max(iv[1] for iv in intervals_ms) + margin_ms
    start_sample = max(0, int(start_ms * sample_rate / 1000.0))
    end_sample = min(len(samples), int(end_ms * sample_rate / 1000.0))
    if end_sample <= start_sample:
        return pcm16
    return np.ascontiguousarray(samples[start_sample:end_sample]).astype("<i2").tobytes()

def _classify(genuine: "list[float]", impostor: "list[float]", min_trials: int, fail_label: str) -> "dict[str, Any]":
    if len(genuine) >= min_trials and len(impostor) >= min_trials:
        eer, threshold = evaluation.equal_error_rate(genuine, impostor)
        return {"eer": eer, "threshold": threshold, "result": "PASS"}
    reason = f"FAIL (too few trials{fail_label}: {len(genuine)} genuine, {len(impostor)} impostor, need >= {min_trials} each)"
    return {"eer": None, "threshold": None, "result": reason}

def _pick_best(passing: "dict[Any, dict]", tie_better: "Callable[[Any, dict, Any, dict], bool]") -> "tuple[Any, dict]":
    """The lowest-`eer` candidate in `passing`, with ties within 0.5pp broken by `tie_better(candidate, result, best, best_result)` (Q2's/Q3's own tie rules, `<interfaces>`, 11-05-PLAN.md)."""
    ranked = sorted(passing.items(), key=lambda kv: kv[1]["eer"])
    best, best_result = ranked[0]
    for candidate, result in ranked[1:]:
        if result["eer"] - best_result["eer"] <= 0.005 and tie_better(candidate, result, best, best_result):
            best, best_result = candidate, result
    return best, best_result

def _score_model_window(
    *, embedder: Any, enrollment_by_label: "dict[str, list[tuple[bytes, list]]]", command_clips: "Sequence[tuple[str, bytes, list]]",
    other_clips: "Sequence[tuple[bytes, list]]", window_ms: int, speech_rms_floor: float, sample_rate: int = SAMPLE_RATE, enrollment_margin_ms: float = 200.0,
) -> "dict[str, Any]":
    """Score every command/other clip against every enrolled member's reference, for one embedder and one window length (D-06)."""
    references: "dict[str, np.ndarray]" = {}
    for label, clips in enrollment_by_label.items():
        vectors = [embedder.embed(_trim_with_margin(pcm16, intervals, enrollment_margin_ms, sample_rate)) for pcm16, intervals in clips]
        if vectors:
            references[label] = mean_embedding(vectors)

    genuine: "list[float]" = []
    impostor: "list[float]" = []
    embed_times_ms: "list[float]" = []

    def _turn_embedding(pcm16: bytes, intervals: "list") -> "np.ndarray | None":
        started = time.perf_counter()
        windows = embed_windows(pcm16, intervals, embedder, window_ms=window_ms, speech_rms_floor=speech_rms_floor, sample_rate=sample_rate)
        embed_times_ms.append((time.perf_counter() - started) * 1000.0)
        return mean_embedding(windows) if windows else None

    for label, pcm16, intervals in command_clips:
        embedding = _turn_embedding(pcm16, intervals)
        if embedding is None or label not in references:
            continue
        normalized = l2_normalize(embedding)
        for ref_label, reference in references.items():
            (genuine if ref_label == label else impostor).append(float(np.dot(normalized, reference)))

    for pcm16, intervals in other_clips:
        embedding = _turn_embedding(pcm16, intervals)
        if embedding is None:
            continue
        normalized = l2_normalize(embedding)
        impostor.extend(float(np.dot(normalized, reference)) for reference in references.values())

    return {
        "references": references,
        "genuine_scores": genuine,
        "impostor_scores": impostor,
        "embed_times_ms": embed_times_ms,
        "embed_ms_p95": evaluation.percentile(embed_times_ms, 95) if embed_times_ms else None,
    }

def question_q1(command_clips: "Sequence[tuple[str, bytes, bytes]]", transcriber: "Callable[[bytes], str]", *, min_clips: int = 10) -> "dict[str, Any]":
    """Q1 (D-08). `command_clips`: `(prompt, ch0_pcm16, ch1_pcm16)` per clip."""
    if len(command_clips) < min_clips:
        return {"result": f"FAIL (only {len(command_clips)} command clip(s), need at least {min_clips})", "asr_channel": None, "ch0_mean_wer": None, "ch1_mean_wer": None}
    ch0_mean = float(np.mean([evaluation.word_error_rate(prompt, transcriber(ch0)) for prompt, ch0, _ in command_clips]))
    ch1_mean = float(np.mean([evaluation.word_error_rate(prompt, transcriber(ch1)) for prompt, _, ch1 in command_clips]))
    asr_channel = 0 if (abs(ch0_mean - ch1_mean) < 0.01 or ch0_mean < ch1_mean) else 1
    return {"result": "PASS", "asr_channel": asr_channel, "ch0_mean_wer": ch0_mean, "ch1_mean_wer": ch1_mean}

def question_q6(noise_frames_rms: "Sequence[float]", *, min_frames: int = 100) -> "dict[str, Any]":
    """Q6: the speech-frame RMS floor."""
    if len(noise_frames_rms) < min_frames:
        return {"result": f"FAIL (only {len(noise_frames_rms)} noise frame(s), need at least {min_frames})", "speech_rms_floor": None, "p95_noise_rms": None}
    p95 = evaluation.percentile(noise_frames_rms, 95)
    return {"result": "PASS", "speech_rms_floor": max(_ceil_to(1.5 * p95, 0.001), 0.002), "p95_noise_rms": p95}

def question_q2(
    *, models: "Sequence[str]", embedder_factory: "Callable[[str], Any]", enrollment_by_label: "dict[str, list[tuple[bytes, list]]]",
    command_clips: "Sequence[tuple[str, bytes, list]]", other_clips: "Sequence[tuple[bytes, list]]", window_ms: int, speech_rms_floor: float,
    sample_rate: int = SAMPLE_RATE, min_trials: int = 20,
) -> "dict[str, Any]":
    """Q2 (D-05): which model wins, at `window_ms`."""
    per_model: "dict[str, Any]" = {}
    for model in models:
        scored = _score_model_window(
            embedder=embedder_factory(model), enrollment_by_label=enrollment_by_label, command_clips=command_clips, other_clips=other_clips,
            window_ms=window_ms, speech_rms_floor=speech_rms_floor, sample_rate=sample_rate,
        )
        entry = dict(scored)
        entry.update(genuine_trials=len(scored["genuine_scores"]), impostor_trials=len(scored["impostor_scores"]))
        entry.update(_classify(scored["genuine_scores"], scored["impostor_scores"], min_trials, f" for {model}"))
        per_model[model] = entry

    passing = {m: r for m, r in per_model.items() if r["result"] == "PASS"}
    if not passing:
        reason = next(iter(per_model.values()))["result"] if per_model else "no models scored"
        return {"result": f"FAIL ({reason})", "models": per_model, "winner": None}

    winner, _ = _pick_best(passing, lambda c, r, b, br: (r["embed_ms_p95"] or math.inf) < (br["embed_ms_p95"] or math.inf))
    return {"result": "PASS", "models": per_model, "winner": winner}

def question_q3(
    *, winning_model: str, embedder_factory: "Callable[[str], Any]", enrollment_by_label: "dict[str, list[tuple[bytes, list]]]",
    command_clips: "Sequence[tuple[str, bytes, list]]", other_clips: "Sequence[tuple[bytes, list]]", speech_rms_floor: float,
    sample_rate: int = SAMPLE_RATE, window_options: "Sequence[int]" = _DEFAULT_WINDOW_OPTIONS, min_trials: int = 20,
) -> "dict[str, Any]":
    """Q3 (D-09): which window length wins, on the winning model."""
    embedder = embedder_factory(winning_model)
    windows_result: "dict[int, Any]" = {}
    for window_ms in window_options:
        scored = _score_model_window(
            embedder=embedder, enrollment_by_label=enrollment_by_label, command_clips=command_clips, other_clips=other_clips,
            window_ms=window_ms, speech_rms_floor=speech_rms_floor, sample_rate=sample_rate,
        )
        entry = dict(scored)
        entry.update(genuine_trials=len(scored["genuine_scores"]), impostor_trials=len(scored["impostor_scores"]))
        entry.update(_classify(scored["genuine_scores"], scored["impostor_scores"], min_trials, f" at window_ms={window_ms}"))
        windows_result[window_ms] = entry

    passing = {w: r for w, r in windows_result.items() if r["result"] == "PASS"}
    if not passing:
        return {"result": "FAIL (no window length had enough trials)", "windows": windows_result, "winner_window_ms": None}

    winner_window_ms, _ = _pick_best(passing, lambda c, r, b, br: c < b)
    return {"result": "PASS", "windows": windows_result, "winner_window_ms": winner_window_ms}

def question_q4(*, genuine_scores: "Sequence[float]", impostor_scores: "Sequence[float]") -> "dict[str, Any]":
    """Q4 (D-06, starting value only): the EER threshold, rounded to 0.01."""
    if not genuine_scores or not impostor_scores:
        return {"result": "FAIL (no genuine/impostor scores for the winning model+window)", "threshold": None}
    _, threshold = evaluation.equal_error_rate(genuine_scores, impostor_scores)
    return {"result": "PASS", "threshold": round(threshold, 2)}

def question_q5(*, clip_windows: "Sequence[tuple[str, list[np.ndarray]]]", window_ms: int, floor_grid: "Sequence[float]" = _DEFAULT_FLOOR_GRID, min_pairs: int = 10) -> "dict[str, Any]":
    """Q5 (D-12): the change-point floor."""
    diff_pairs: "list[tuple[int, list]]" = []
    same_pairs: "list[tuple[int, list]]" = []
    for i, (label_i, windows_i) in enumerate(clip_windows):
        for j, (label_j, windows_j) in enumerate(clip_windows):
            if i == j or not windows_i or not windows_j:
                continue
            entry = (len(windows_i), list(windows_i) + list(windows_j))
            (same_pairs if label_i == label_j else diff_pairs).append(entry)

    if len(diff_pairs) < min_pairs or len(same_pairs) < min_pairs:
        return {"result": f"FAIL ({len(diff_pairs)} different-label pair(s), {len(same_pairs)} same-label pair(s), need at least {min_pairs} of each)", "floors": {}, "chosen_floor": None}

    floor_reports: "dict[float, Any]" = {}
    for floor in floor_grid:
        detections = 0
        lags_ms: "list[float]" = []
        for join_index, joined in diff_pairs:
            hit = next((cp for cp in find_change_points(joined, similarity_floor=floor) if abs(cp - join_index) <= 1), None)
            if hit is not None:
                detections += 1
                lags_ms.append((hit - join_index + 1) * window_ms)
        false_splits = sum(1 for _, joined in same_pairs if find_change_points(joined, similarity_floor=floor))
        floor_reports[floor] = {
            "detection_rate": detections / len(diff_pairs),
            "false_split_rate": false_splits / len(same_pairs),
            "mean_lag_ms": float(np.mean(lags_ms)) if lags_ms else None,
        }

    eligible = {f: r for f, r in floor_reports.items() if r["false_split_rate"] <= 0.10}
    if not eligible:
        return {"result": "FAIL (no floor keeps false splits at or below 10%)", "floors": floor_reports, "chosen_floor": None}

    chosen_floor = max(eligible, key=lambda f: (eligible[f]["detection_rate"], -f))
    chosen = eligible[chosen_floor]
    if chosen["detection_rate"] < 0.80:
        return {"result": f"FAIL (best floor {chosen_floor} detects only {chosen['detection_rate']:.0%}, need at least 80%)", "floors": floor_reports, "chosen_floor": chosen_floor}
    return {
        "result": "PASS", "floors": floor_reports, "chosen_floor": chosen_floor,
        "detection_rate": chosen["detection_rate"], "false_split_rate": chosen["false_split_rate"], "mean_lag_ms": chosen["mean_lag_ms"],
    }

def question_q7(enrollment_clips_intervals: "Sequence[list]", *, min_clips: int = 5) -> "dict[str, Any]":
    """Q7: enrollment capture -- single-interval share and the inter-phrase gap."""
    if len(enrollment_clips_intervals) < min_clips:
        return {"result": f"FAIL (only {len(enrollment_clips_intervals)} enrollment clip(s), need at least {min_clips})", "single_interval_share": None, "enrollment_gap_ms": None}
    single_share = sum(1 for iv in enrollment_clips_intervals if len(iv) == 1) / len(enrollment_clips_intervals)
    gaps_ms = [b[0] - a[1] for intervals in enrollment_clips_intervals for a, b in zip(sorted(intervals, key=lambda iv: iv[0]), sorted(intervals, key=lambda iv: iv[0])[1:])]
    p95_gap = evaluation.percentile(gaps_ms, 95) if gaps_ms else 0.0
    return {"result": "PASS", "single_interval_share": single_share, "p95_gap_ms": p95_gap, "enrollment_gap_ms": max(800, _ceil_to(p95_gap + 200, 50))}

def question_q8(
    *, reply_clips: "Sequence[tuple[str, bytes, list]]", references: "dict[str, np.ndarray]", embedder: Any, window_ms: int, speech_rms_floor: float,
    threshold: "float | None", sample_rate: int = SAMPLE_RATE, min_clips: int = 5,
) -> "dict[str, Any]":
    """Q8 (D-11 consequence): short-reply acceptance. Informational -- never blocks the overall verdict."""
    scored = 0
    accepted = 0
    for label, pcm16, intervals in reply_clips:
        windows = embed_windows(pcm16, intervals, embedder, window_ms=window_ms, speech_rms_floor=speech_rms_floor, sample_rate=sample_rate)
        if not windows or label not in references:
            continue
        scored += 1
        score = float(np.dot(l2_normalize(mean_embedding(windows)), references[label]))
        if threshold is not None and score >= threshold:
            accepted += 1
    result = "PASS" if len(reply_clips) >= min_clips else f"FAIL (only {len(reply_clips)} reply clip(s), need at least {min_clips} -- informational only)"
    return {"result": result, "scored_count": scored, "accepted_count": accepted, "acceptance_share": (accepted / scored) if scored else None}

def question_q9(*, embed_times_ms: "Sequence[float]") -> "dict[str, Any]":
    """Q9: embedding wall time. Informational -- never blocks the overall verdict."""
    if not embed_times_ms:
        return {"result": "FAIL (no embedding time measurements)", "p50_ms": None, "p95_ms": None}
    return {"result": "PASS", "p50_ms": evaluation.percentile(embed_times_ms, 50), "p95_ms": evaluation.percentile(embed_times_ms, 95)}

def overall_verdict(results: "dict[str, dict]") -> str:
    """`FAIL (Qn: reason; ...)` naming every failed question among Q1/Q2/Q5/Q6/Q7 -- Q3, Q4, Q8, Q9 never affect this verdict."""
    failures = [f"{name.upper()}: {results[name]['result']}" for name in _BLOCKING_QUESTIONS if not results.get(name, {}).get("result", "").startswith("PASS")]
    return f"FAIL ({'; '.join(failures)})" if failures else "PASS"

def _report_keys(q1, model, window_ms, q4, q5, q6, q7, q8, q9) -> "dict[str, Any]":
    return {
        "asr_channel": q1.get("asr_channel"),
        "model": model,
        "window_ms": window_ms,
        "threshold": q4.get("threshold") if q4 else None,
        "change_similarity_floor": q5.get("chosen_floor") if q5 else None,
        "speech_rms_floor": q6.get("speech_rms_floor"),
        "enrollment_gap_ms": q7.get("enrollment_gap_ms"),
        "short_reply_acceptance": q8.get("acceptance_share") if q8 else None,
        "embed_ms_p95": q9.get("p95_ms") if q9 else None,
    }

def score_corpus(
    root: Path, *, embedder_factory: "Callable[[str], Any]", transcriber: "Callable[[bytes], str]", models: "Sequence[str]" = tuple(SPEAKER_MODEL_FILES),
    window_options: "Sequence[int]" = _DEFAULT_WINDOW_OPTIONS, floor_grid: "Sequence[float]" = _DEFAULT_FLOOR_GRID, sample_rate: int = SAMPLE_RATE,
) -> "dict[str, Any]":
    """Run every question in order and return the full report, ending with `keys` and `verdict` (`<interfaces>`, 11-05-PLAN.md)."""
    clips = load_corpus(root)
    enrollment_clips = [c for c in clips if c.kind == "enrollment"]
    command_clips_raw = [c for c in clips if c.kind == "command"]
    reply_clips_raw = [c for c in clips if c.kind == "reply"]
    other_clips_raw = [c for c in clips if c.kind == "other"]

    q1 = question_q1([(c.prompt or "", _channel_pcm16(c, 0), _channel_pcm16(c, 1)) for c in command_clips_raw], transcriber)
    asr_channel = q1.get("asr_channel") if q1.get("asr_channel") is not None else 0

    noise_rms: "list[float]" = []
    frame_ms = FRAME_SAMPLES * 1000.0 / sample_rate
    for clip in clips:
        if clip.kind == "other":
            continue  # music/TV/radio, not room noise (operator diagnosis, 2026-09-29)
        samples = np.frombuffer(_channel_pcm16(clip, asr_channel), dtype="<i2")
        for i, start in enumerate(range(0, len(samples) - FRAME_SAMPLES + 1, FRAME_SAMPLES)):
            start_ms = i * frame_ms
            if _frame_is_room_noise(start_ms, start_ms + frame_ms, clip.speech_intervals_ms):
                frame_bytes = np.ascontiguousarray(samples[start : start + FRAME_SAMPLES]).astype("<i2").tobytes()
                noise_rms.append(rms_amplitude(frame_bytes))
    q6 = question_q6(noise_rms)
    speech_rms_floor = q6.get("speech_rms_floor") or 0.002

    enrollment_by_label: "dict[str, list[tuple[bytes, list]]]" = {}
    for clip in enrollment_clips:
        enrollment_by_label.setdefault(clip.label, []).append((_channel_pcm16(clip, asr_channel), clip.speech_intervals_ms))
    command_clips = [(c.label, _channel_pcm16(c, asr_channel), c.speech_intervals_ms) for c in command_clips_raw]
    other_clips = [(_channel_pcm16(c, asr_channel), c.speech_intervals_ms) for c in other_clips_raw]
    reply_clips = [(c.label, _channel_pcm16(c, asr_channel), c.speech_intervals_ms) for c in reply_clips_raw]

    q2 = question_q2(
        models=models, embedder_factory=embedder_factory, enrollment_by_label=enrollment_by_label, command_clips=command_clips,
        other_clips=other_clips, window_ms=window_options[0], speech_rms_floor=speech_rms_floor, sample_rate=sample_rate,
    )
    results: "dict[str, Any]" = {"q1": q1, "q2": q2, "q6": q6}
    winner_model = q2.get("winner")
    q7 = question_q7([c.speech_intervals_ms for c in enrollment_clips])

    if winner_model is None:
        results.update(
            q3={"result": "FAIL (no winning model from Q2)", "windows": {}, "winner_window_ms": None},
            q4={"result": "FAIL (no winning model from Q2)", "threshold": None},
            q5={"result": "FAIL (no winning model from Q2)", "floors": {}, "chosen_floor": None},
            q7=q7,
            q8={"result": "FAIL (no winning model from Q2)", "scored_count": 0, "accepted_count": 0, "acceptance_share": None},
            q9={"result": "FAIL (no embedding time measurements)", "p50_ms": None, "p95_ms": None},
        )
        results["verdict"] = overall_verdict(results)
        results["keys"] = _report_keys(q1, None, None, None, None, q6, q7, None, None)
        return results

    q3 = question_q3(
        winning_model=winner_model, embedder_factory=embedder_factory, enrollment_by_label=enrollment_by_label, command_clips=command_clips,
        other_clips=other_clips, speech_rms_floor=speech_rms_floor, sample_rate=sample_rate, window_options=window_options,
    )
    winner_window_ms = q3.get("winner_window_ms")
    winner_window_result = q3.get("windows", {}).get(winner_window_ms, {}) if winner_window_ms else {}
    q4 = question_q4(genuine_scores=winner_window_result.get("genuine_scores", []), impostor_scores=winner_window_result.get("impostor_scores", []))

    winner_embedder = embedder_factory(winner_model)
    effective_window_ms = winner_window_ms or window_options[0]
    clip_windows = [
        (label, embed_windows(pcm16, intervals, winner_embedder, window_ms=effective_window_ms, speech_rms_floor=speech_rms_floor, sample_rate=sample_rate))
        for label, pcm16, intervals in command_clips
    ]
    q5 = question_q5(clip_windows=clip_windows, window_ms=effective_window_ms, floor_grid=floor_grid)

    references = q2["models"].get(winner_model, {}).get("references", {})
    q8 = question_q8(
        reply_clips=reply_clips, references=references, embedder=winner_embedder, window_ms=effective_window_ms,
        speech_rms_floor=speech_rms_floor, threshold=q4.get("threshold"), sample_rate=sample_rate,
    )
    q9 = question_q9(embed_times_ms=winner_window_result.get("embed_times_ms", []))

    results.update(q3=q3, q4=q4, q5=q5, q7=q7, q8=q8, q9=q9)
    results["verdict"] = overall_verdict(results)
    results["keys"] = _report_keys(q1, winner_model, winner_window_ms, q4, q5, q6, q7, q8, q9)
    return results

def _default_embedder_factory(model_dir: Path) -> "Callable[[str], Any]":
    from atlas.speaker_id.embedding import SherpaEmbedder  # noqa: PLC0415 -- lazy, no device needed for --help

    cache: "dict[str, Any]" = {}

    def factory(model: str) -> Any:
        if model not in cache:
            cache[model] = SherpaEmbedder(model_dir / SPEAKER_MODEL_FILES[model])
        return cache[model]

    return factory

def _default_transcriber(whisper_model: str) -> "Callable[[bytes], str]":
    from faster_whisper import WhisperModel  # noqa: PLC0415 -- lazy, no model load for --help

    model = WhisperModel(whisper_model, device="cpu", compute_type="int8")

    def transcribe(pcm16: bytes) -> str:
        samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
        segments, _ = model.transcribe(samples, beam_size=5)
        return " ".join(segment.text.strip() for segment in segments).strip()

    return transcribe

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Score a recorded speaker corpus into a spike verdict (11-05-PLAN.md).")
    subparsers = parser.add_subparsers(dest="command", required=True)
    score_p = subparsers.add_parser("score", help="Score a corpus directory into a JSON report.")
    score_p.add_argument("--corpus", required=True, type=Path)
    score_p.add_argument("--model-dir", required=True, type=Path, dest="model_dir")
    score_p.add_argument("--whisper-model", default="small.en", dest="whisper_model")
    score_p.add_argument("--out", type=Path, default=None)
    score_p.set_defaults(func=cmd_score)
    return parser

def cmd_score(args: argparse.Namespace) -> int:
    report = score_corpus(args.corpus, embedder_factory=_default_embedder_factory(args.model_dir), transcriber=_default_transcriber(args.whisper_model))
    text = json.dumps(report, indent=2, default=str)
    print(text)
    if args.out is not None:
        args.out.write_text(text, encoding="utf-8")
    return 0

def main(argv: "list[str] | None" = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))

if __name__ == "__main__":
    sys.exit(main())
