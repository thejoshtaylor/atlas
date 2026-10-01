"""The Vosk `WakeDetector` implementation: grammar-constrained decoding.

`vosk.KaldiRecognizer`'s three-argument constructor (model, sample rate,
grammar JSON) restricts the decoder to exactly the phrases named in that
grammar -- here, the configured wake phrase plus `"[unk]"` -- so anything
that is not the phrase decodes as unknown and is discarded before it ever
reaches this module as a match. That grammar constraint, not this module,
is what turns a full ASR engine into a cheap, low-false-positive wake
detector (CONTEXT.md, D-08).

Follows `stt_xai.py`'s adapter shape: a thin class holding only its config,
translating this project's field names into the engine's own constructor
and method names explicitly (`VoskWakeConfig.model_path`/`grammar` ->
`vosk.Model`/`vosk.KaldiRecognizer`), never passing config keys straight
through.
"""

from __future__ import annotations

import json
import logging
import os

from atlas.config import VoskWakeConfig
from atlas.wake.base import WakeError, WakeHit

logger = logging.getLogger("atlas.wake.vosk_engine")

# vosk.KaldiRecognizer's sample rate is fixed at construction time and must
# match every chunk `process()` receives -- the detector-only decode path in
# transports/camera.py resamples to exactly this rate.
_SAMPLE_RATE = 16000


def _contains_phrase(tokens: list[str], phrase_tokens: tuple[str, ...]) -> bool:
    """True when `phrase_tokens` occurs as a contiguous run inside `tokens`.

    An empty phrase never matches. Tokens compare exactly, with no case
    folding, because Vosk emits lowercase words from the grammar.
    """
    n = len(phrase_tokens)
    if n == 0:
        return False
    return any(tuple(tokens[i : i + n]) == phrase_tokens for i in range(len(tokens) - n + 1))


class VoskWakeDetector:
    """Grammar-constrained Vosk wake detector."""

    def __init__(self, config: VoskWakeConfig, phrase: str) -> None:
        # Imported here, not at module load: `vosk` links a native library,
        # and only the code path that actually selects this engine
        # (`wake.engine == "vosk"`) needs to load it.
        import vosk

        if not os.path.isdir(config.model_path):
            raise WakeError(
                f"vosk wake model directory not found: {config.model_path!r} -- "
                "a missing model is a startup failure that names itself, not a "
                "detector that silently never fires"
            )

        vosk.SetLogLevel(-1)
        self._model = vosk.Model(config.model_path)
        grammar_json = json.dumps(list(config.grammar))
        self._recognizer = vosk.KaldiRecognizer(self._model, _SAMPLE_RATE, grammar_json)
        self._phrase = phrase
        self._phrase_tokens = tuple(phrase.split())
        # The keyword is the last word of the phrase ("atlas" for "hey atlas").
        self._keyword_tokens = self._phrase_tokens[-1:]

    def process(self, chunk: bytes) -> WakeHit | None:
        """Feed one chunk of 16 kHz mono PCM16 to the recognizer.

        The detector reads the committed `Result()` when `AcceptWaveform`
        commits, else `PartialResult()`, and fires when the phrase tokens
        occur as a contiguous run in that text. Waiting for `AcceptWaveform`
        alone adds Kaldi's endpoint silence (about 0.5 to 1 s) to every
        wake, and vosk 0.3.44 has no call to shorten that silence.

        A run-on command commits as the phrase plus `[unk]` tokens (for
        example `hey atlas [unk] [unk]`), so an exact-equality check never
        fires on it. The match is a whole-word token run, so the grammar
        decoys (`at last`, `the atlas`) still never fire.

        `Reset()` after a hit is necessary. The partial stays in the
        recognizer until the endpoint, so it would fire again on every
        following chunk. The `WakeGate` refractory window is a second
        guard only.
        """
        return self._match(chunk, self._phrase_tokens)

    def process_during_playback(self, chunk: bytes) -> WakeHit | None:
        """Like `process`, but the keyword alone is a hit.

        The barge-in listener calls this while a reply plays on the edge
        source. The XVF3800 echo suppressor clips the start of speech during
        playback, so "hey" often decodes as the decoy "the" or "a". "atlas"
        decodes correctly. So "the atlas", "a atlas" and "atlas" fire here,
        and "at last" and "alice" do not. The idle path keeps the full
        phrase. Transcript verification (`turn/wake_echo.strip_wake_phrase`)
        and the reply's own wake-word windows (`sources/reply_timing.py`)
        use the same keyword.
        """
        return self._match(chunk, self._keyword_tokens)

    def _match(self, chunk: bytes, tokens: tuple[str, ...]) -> WakeHit | None:
        if self._recognizer.AcceptWaveform(chunk):
            text = json.loads(self._recognizer.Result()).get("text", "")
        else:
            text = json.loads(self._recognizer.PartialResult()).get("partial", "")
        if not _contains_phrase(text.split(), tokens):
            return None
        self._recognizer.Reset()
        # The grammar match itself is binary -- Vosk's own result carries no
        # confidence score the way openWakeWord's threshold-based detector
        # does. 1.0 names that a match against a constrained grammar is a
        # categorical hit, not a graded one; `WakeHit.score` stays a float
        # for both engines to share the same protocol member.
        return WakeHit(score=1.0)

    def reset(self) -> None:
        """Clear whatever utterance state `AcceptWaveform` has accumulated
        (WR-02, code review): without this, `scripts/score_wake_engines.py`
        reusing one `VoskWakeDetector` across every corpus recording could
        carry a still-open decode from one recording's trailing audio into
        the next one's very first bytes -- misattributing a hit (or a miss)
        to the wrong file. `KaldiRecognizer.Reset()` is vosk's own call for
        this, cheaper than discarding and reloading the whole model between
        recordings."""
        self._recognizer.Reset()

    def close(self) -> None:
        # Both `vosk.Model` and `vosk.KaldiRecognizer` free their native
        # handles in their own `__del__`; dropping the references is enough
        # and avoids reaching into vosk's private handle attributes.
        self._recognizer = None
        self._model = None
