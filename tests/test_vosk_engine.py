"""Unit tests for `VoskWakeDetector` against a fake `vosk` module.

The real Vosk model directory is a deployment artifact and is not in git,
so these tests inject a small fake `vosk` module into `sys.modules`. The
fake models the one Kaldi behavior that matters here: a partial result
grows as chunks arrive and stays in place until an endpoint (a chunk that
ends with the `|END` marker) or a `Reset()` call.

Chunks are UTF-8 text, for example `b"hey"` or `b"hey atlas [unk]|END"`.
"""

from __future__ import annotations

import json
import sys
import types

import pytest

from atlas.config import VoskWakeConfig
from atlas.wake.base import WakeHit
from atlas.wake.vosk_engine import VoskWakeDetector

_END = "|END"


class _FakeRecognizer:
    def __init__(self, model: object, sample_rate: int, grammar_json: str) -> None:
        self.pending: list[str] = []
        self.final = ""
        self.endpoints = 0
        self.reset_calls = 0

    def AcceptWaveform(self, chunk: bytes) -> bool:  # noqa: N802 - vosk's name
        text = chunk.decode("utf-8")
        endpoint = text.endswith(_END)
        if endpoint:
            text = text[: -len(_END)]
        self.pending.extend(text.split())
        if not endpoint:
            return False
        self.final = " ".join(self.pending)
        self.pending = []
        self.endpoints += 1
        return True

    def Result(self) -> str:  # noqa: N802
        text, self.final = self.final, ""
        return json.dumps({"text": text})

    def PartialResult(self) -> str:  # noqa: N802
        return json.dumps({"partial": " ".join(self.pending)})

    def Reset(self) -> None:  # noqa: N802
        self.pending = []
        self.final = ""
        self.reset_calls += 1


@pytest.fixture
def make_detector(monkeypatch, tmp_path):
    """Return a factory that builds a detector and its fake recognizer."""
    created: list[_FakeRecognizer] = []

    def _recognizer(model: object, sample_rate: int, grammar_json: str) -> _FakeRecognizer:
        rec = _FakeRecognizer(model, sample_rate, grammar_json)
        created.append(rec)
        return rec

    fake = types.ModuleType("vosk")
    fake.SetLogLevel = lambda level: None  # type: ignore[attr-defined]
    fake.Model = lambda path: path  # type: ignore[attr-defined]
    fake.KaldiRecognizer = _recognizer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "vosk", fake)

    def _make() -> tuple[VoskWakeDetector, _FakeRecognizer]:
        detector = VoskWakeDetector(VoskWakeConfig(model_path=str(tmp_path)), "hey atlas")
        return detector, created[-1]

    return _make


def test_partial_phrase_fires_before_endpoint(make_detector):
    """The hit arrives on the chunk that completes the partial, with no endpoint."""
    detector, rec = make_detector()
    assert detector.process(b"hey") is None
    assert detector.process(b"atlas") == WakeHit(score=1.0)
    assert rec.endpoints == 0
    assert rec.reset_calls == 1


def test_run_on_final_fires(make_detector):
    """A committed `hey atlas [unk] [unk]` holds the phrase as a token run."""
    detector, rec = make_detector()
    assert detector.process(b"hey atlas [unk] [unk]|END") == WakeHit(score=1.0)
    assert rec.endpoints == 1


def test_exact_final_fires(make_detector):
    """The old exact-match case still fires."""
    detector, _ = make_detector()
    assert detector.process(b"hey atlas|END") == WakeHit(score=1.0)


def test_leading_unknown_fires(make_detector):
    """A leading `[unk]` (the Pi segment case) does not block the phrase."""
    detector, _ = make_detector()
    assert detector.process(b"[unk]") is None
    assert detector.process(b"hey") is None
    assert detector.process(b"atlas") == WakeHit(score=1.0)


@pytest.mark.parametrize(
    "decoy",
    ["at last", "the atlas", "hey", "atlas", "hey at last", "hey alice", "[unk] atlas"],
)
@pytest.mark.parametrize("final", [False, True])
def test_decoys_stay_silent(make_detector, decoy, final):
    """Grammar decoys never fire, as a partial or as a final result."""
    detector, rec = make_detector()
    chunk = (decoy + (_END if final else "")).encode("utf-8")
    assert detector.process(chunk) is None
    assert rec.reset_calls == 0


def test_one_phrase_gives_one_hit(make_detector):
    """Empty chunks after a hit return None because the hit reset the recognizer."""
    detector, rec = make_detector()
    assert detector.process(b"hey") is None
    assert detector.process(b"atlas") == WakeHit(score=1.0)
    # The fake keeps a partial in place until an endpoint or `Reset()`, so
    # without the reset these empty chunks would fire again.
    for _ in range(3):
        assert detector.process(b"") is None
    assert rec.reset_calls == 1


def test_fires_again_for_a_new_phrase(make_detector):
    """After a hit and reset, a second spoken phrase gives a second hit."""
    detector, rec = make_detector()
    detector.process(b"hey")
    assert detector.process(b"atlas") == WakeHit(score=1.0)
    detector.process(b"")
    detector.process(b"hey")
    assert detector.process(b"atlas") == WakeHit(score=1.0)
    assert rec.reset_calls == 2


def test_reset_clears_open_partial(make_detector):
    """`reset()` calls `Reset()` once, and an old partial does not join later words."""
    detector, rec = make_detector()
    assert detector.process(b"hey") is None
    detector.reset()
    assert rec.reset_calls == 1
    assert detector.process(b"atlas") is None
