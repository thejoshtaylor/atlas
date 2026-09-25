"""RED for SileroGate (10-08-PLAN.md Task 1), against a fake detector --
no real sherpa-onnx model file is required to run this suite.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from atlas_edge.vad import DEFAULT_MIN_SILENCE_MS, VAD_WINDOW_SAMPLES, SileroGate


class FakeDetector:
    """Records every window it was fed and reports speech once a caller
    calls `set_speech(True)`. `pop()`-generated segments are queued
    explicitly via `queue_segment()`, mirroring the real detector's own
    "front"/"pop" pair."""

    def __init__(self) -> None:
        self.windows: list[np.ndarray] = []
        self._speech = False
        self._segments: list[object] = []

    def accept_waveform(self, window: np.ndarray) -> None:
        self.windows.append(window)

    def is_speech_detected(self) -> bool:
        return self._speech

    def empty(self) -> bool:
        return not self._segments

    def pop(self) -> None:
        self._segments.pop(0)

    def set_speech(self, value: bool) -> None:
        self._speech = value

    def queue_segment(self) -> None:
        self._segments.append(object())


def _int16_bytes(values: "list[int]") -> bytes:
    return np.array(values, dtype=np.int16).tobytes()


def test_gate_buffers_into_512_sample_windows_and_feeds_float32():
    fake = FakeDetector()
    gate = SileroGate("unused.onnx", detector_factory=lambda: fake)

    half_window = _int16_bytes([100] * (VAD_WINDOW_SAMPLES // 2))
    gate.push(half_window)
    assert fake.windows == []  # not a full window yet

    gate.push(half_window)
    assert len(fake.windows) == 1
    window = fake.windows[0]
    assert window.dtype == np.float32
    assert window.shape == (VAD_WINDOW_SAMPLES,)
    assert -1.0 <= float(window.max()) < 1.0
    assert -1.0 <= float(window.min()) < 1.0


def test_gate_returns_latest_is_speech_detected():
    fake = FakeDetector()
    gate = SileroGate("unused.onnx", detector_factory=lambda: fake)
    full_window = _int16_bytes([0] * VAD_WINDOW_SAMPLES)

    assert gate.push(full_window) is False

    fake.set_speech(True)
    assert gate.push(full_window) is True


def test_gate_pops_every_completed_segment_so_memory_stays_bounded():
    fake = FakeDetector()
    gate = SileroGate("unused.onnx", detector_factory=lambda: fake)
    full_window = _int16_bytes([0] * VAD_WINDOW_SAMPLES)

    fake.queue_segment()
    fake.queue_segment()
    gate.push(full_window)
    assert fake.empty()  # both segments were popped


def test_missing_model_file_raises_naming_the_path():
    gate = SileroGate("/nonexistent/path/silero_vad.onnx")
    with pytest.raises(FileNotFoundError, match=re.escape("/nonexistent/path/silero_vad.onnx")):
        gate.push(_int16_bytes([0] * VAD_WINDOW_SAMPLES))


def test_default_min_silence_ms_is_a_modest_default():
    assert DEFAULT_MIN_SILENCE_MS == 250


def test_importing_vad_module_never_imports_sherpa_onnx():
    text = Path(__file__).resolve().parent.parent.joinpath(
        "src", "atlas_edge", "vad.py"
    ).read_text()
    assert not re.search(r"^(import sherpa_onnx|from sherpa_onnx)", text, re.MULTILINE)
