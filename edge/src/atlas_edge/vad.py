"""SileroGate: sherpa-onnx's VoiceActivityDetector over the ASR channel.

Importing this module never imports `sherpa_onnx` -- the detector builds
lazily, inside `SileroGate`, on first `push()`. The Pi runs capture and
Silero VAD only: no wake word, no speech-to-text, no speaker ID, no
diarization (D-06) -- this module never imports any of those libraries
either, and a repo-wide grep confirms it.
"""

from __future__ import annotations

import os
from typing import Callable

import numpy as np

# 512 samples at 16 kHz is 32 ms -- the same window the operator's Silero
# onset-delay benchmark on a Pi 4 measured against (10-RESEARCH.md Open
# Question 2, confirmed against the installed sherpa-onnx==1.13.8's own
# default `VadModelConfig.silero_vad.window_size`). Feeding a different
# window size would change the very number that benchmark measured.
VAD_WINDOW_SAMPLES = 512

# sherpa-onnx's own documented `min_silence_duration` default (seconds,
# expressed here in ms) -- the "modest default" D-11 keeps on the Pi. The
# server's own `end_of_speech_hangover_ms` is the tuning knob; this
# default is deliberately not tuned per-house.
DEFAULT_MIN_SILENCE_MS = 250

DEFAULT_THRESHOLD = 0.5

_BYTES_PER_SAMPLE = 2  # int16
_INT16_FULL_SCALE = 32768.0


class SileroGate:
    """Feeds mono PCM16 into `VAD_WINDOW_SAMPLES`-sample windows, runs
    sherpa-onnx's `VoiceActivityDetector`, and reports the latest
    `is_speech_detected()`. `detector_factory`, when given, replaces the
    real sherpa-onnx detector with a fake for tests -- production code
    never sets it."""

    def __init__(
        self,
        model_path: str,
        *,
        sample_rate: int = 16000,
        threshold: float = DEFAULT_THRESHOLD,
        min_silence_ms: int = DEFAULT_MIN_SILENCE_MS,
        min_speech_ms: int = 250,
        detector_factory: "Callable[[], object] | None" = None,
    ) -> None:
        self._model_path = model_path
        self._sample_rate = sample_rate
        self._threshold = threshold
        self._min_silence_ms = min_silence_ms
        self._min_speech_ms = min_speech_ms
        self._detector_factory = detector_factory
        self._detector = None
        self._buffer = bytearray()

    def _build_detector(self):
        if not os.path.exists(self._model_path):
            raise FileNotFoundError(
                f"Silero VAD model not found at {self._model_path!r} -- "
                "provisioning fetches it (plan 10-09); it is never "
                "downloaded here."
            )
        import sherpa_onnx  # lazy: never imported at module scope (D-06)

        config = sherpa_onnx.VadModelConfig()
        config.silero_vad.model = self._model_path
        config.silero_vad.threshold = self._threshold
        config.silero_vad.min_silence_duration = self._min_silence_ms / 1000.0
        config.silero_vad.min_speech_duration = self._min_speech_ms / 1000.0
        config.silero_vad.window_size = VAD_WINDOW_SAMPLES
        config.sample_rate = self._sample_rate
        return sherpa_onnx.VoiceActivityDetector(config, buffer_size_in_seconds=30)

    def push(self, samples: bytes) -> bool:
        """Feed mono PCM16 bytes (any length). Buffers into
        `VAD_WINDOW_SAMPLES`-sample windows, feeds each completed window
        as float32 in [-1, 1) to the detector, and pops every completed
        segment so memory stays bounded. Returns the latest
        `is_speech_detected()`."""
        if self._detector is None:
            self._detector = (
                self._detector_factory() if self._detector_factory is not None else self._build_detector()
            )
        self._buffer += samples
        window_bytes = VAD_WINDOW_SAMPLES * _BYTES_PER_SAMPLE
        while len(self._buffer) >= window_bytes:
            chunk = bytes(self._buffer[:window_bytes])
            del self._buffer[:window_bytes]
            window = np.frombuffer(chunk, dtype=np.int16).astype(np.float32) / _INT16_FULL_SCALE
            self._detector.accept_waveform(window)
            while not self._detector.empty():
                self._detector.pop()
        return self._detector.is_speech_detected()
