"""The openWakeWord `WakeDetector` implementation: threshold-based scoring
against a custom "hey spire" ONNX model.

**That model does not exist.** `RESEARCH.md` Pitfall 3 establishes that
openWakeWord's pretrained models cover common phrases ("hey jarvis,"
"alexa") only -- a phrase-specific model for "hey spire" comes from
openWakeWord's own offline training pipeline (synthetic TTS-generated
positives plus real recordings, run through its `training_models.ipynb` or
`automatic_model_training.ipynb` notebooks), not from a coding task. This
module ships the adapter so both engines exist behind `WakeDetector`
regardless of which is configured (a PROJECT.md Key Decision) -- it does
not, and cannot, produce the model file itself. Training it is out of this
plan's scope entirely; there is no helper here, and none should be added,
that makes training look like a step a caller could just run.

Follows `vosk_engine.py`'s adapter shape: a thin class holding only its
config, translating this project's field names into the engine's own
constructor and method names explicitly (`OpenWakeWordConfig.model_path` /
`threshold` / `trigger_frames` -> `openwakeword.Model(wakeword_models=...,
inference_framework="onnx")` and its `.predict()` call), never passing
config keys straight through -- the failure mode `stt_xai.py`'s own
Pitfall 1 names for a different field.
"""

from __future__ import annotations

import logging
import os

from spire_voice.config import OpenWakeWordConfig
from spire_voice.wake.base import WakeError, WakeHit

logger = logging.getLogger("spire_voice.wake.openwakeword_engine")

# openWakeWord's own frame contract (dscripka/openWakeWord README,
# `Model.predict()` docstring): 80ms per frame at 16 kHz mono 16-bit PCM =
# 1280 samples = 2560 bytes. This is the library's own fact, not a
# tunable -- `config.example.yaml`'s own comment already states it, and
# `trigger_frames` counts frames of exactly this size.
_FRAME_BYTES = 2560

# `config.example.yaml`'s `wake.openwakeword.model_path` names a `.onnx`
# file, not `.tflite` -- `Model.__init__`'s own default `inference_framework`
# is "tflite" and would try to import `tflite_runtime`, which this project
# never installs (RESEARCH.md's own note: pin the ONNX path explicitly
# rather than let openWakeWord pick).
_INFERENCE_FRAMEWORK = "onnx"


class OpenWakeWordDetector:
    """Threshold-based openWakeWord wake detector, scored against exactly
    one custom model: the one `OpenWakeWordConfig.model_path` names."""

    def __init__(self, config: OpenWakeWordConfig) -> None:
        # Imported here, not at module load -- mirroring `vosk_engine.py`'s
        # own deferred import: openwakeword links onnxruntime, and only the
        # code path that actually selects this engine (`wake.engine ==
        # "openwakeword"`, or this plan's scoring harness) needs to load it.
        import openwakeword

        if not os.path.isfile(config.model_path):
            raise WakeError(
                f"openwakeword model not found: {config.model_path!r} -- training a "
                "phrase-specific model is an offline pipeline (openWakeWord's own "
                "training notebooks: synthetic TTS positives plus real recordings), "
                "not a step this codebase runs. See RESEARCH.md Pitfall 3."
            )

        self._model_name = os.path.splitext(os.path.basename(config.model_path))[0]
        self._model = openwakeword.Model(
            wakeword_models=[config.model_path],
            inference_framework=_INFERENCE_FRAMEWORK,
        )
        self._threshold = config.threshold
        self._trigger_frames = config.trigger_frames
        self._consecutive_hits = 0
        self._buffer = bytearray()

    def process(self, chunk: bytes) -> WakeHit | None:
        """Accumulate `chunk` into whole 80ms frames and score each one.

        `trigger_frames` consecutive frames at or above `threshold` before
        a hit counts (`config.example.yaml`'s own comment) -- a single
        loud frame does not fire alone, the same "not a single frame"
        discipline `CONTEXT.md` states for barge-in. Only the first
        qualifying frame within one `process()` call returns a `WakeHit`;
        callers see at most one hit per call, matching `VoskWakeDetector`'s
        own one-hit-per-call shape.
        """
        # Deferred with the rest of this engine's native stack -- only
        # loaded once this engine is actually selected.
        import numpy as np

        self._buffer += chunk
        hit: WakeHit | None = None
        while len(self._buffer) >= _FRAME_BYTES:
            frame = bytes(self._buffer[:_FRAME_BYTES])
            del self._buffer[:_FRAME_BYTES]
            samples = np.frombuffer(frame, dtype=np.int16)
            predictions = self._model.predict(samples)
            score = float(predictions.get(self._model_name, 0.0))
            if score >= self._threshold:
                self._consecutive_hits += 1
            else:
                self._consecutive_hits = 0
            if hit is None and self._consecutive_hits >= self._trigger_frames:
                self._consecutive_hits = 0
                hit = WakeHit(score=score)
        return hit

    def reset(self) -> None:
        """Clear the frame-accumulation buffer and consecutive-hit counter
        (WR-02, code review; see `VoskWakeDetector.reset`'s own docstring
        for why this matters to the offline scoring harness): otherwise a
        partial frame or an in-progress hit streak carried over from the
        previous recording's tail would count against the next
        recording's first bytes, the same misattribution risk Vosk's own
        recognizer state has."""
        self._buffer = bytearray()
        self._consecutive_hits = 0

    def close(self) -> None:
        # openwakeword's `Model` holds an onnxruntime `InferenceSession`
        # with no explicit close method of its own -- dropping the
        # reference is enough, the same discipline `VoskWakeDetector.close`
        # already states for vosk's native handles.
        self._model = None
