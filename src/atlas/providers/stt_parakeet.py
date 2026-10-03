"""Local speech-to-text via NVIDIA Parakeet TDT 0.6b v2, run by sherpa-onnx.

The shape matches `stt_faster_whisper.py`. The provider decodes the whole
utterance once, after the endpoint. The model loads once, at construction,
from a directory that `scripts/fetch_models.py --only parakeet` filled
earlier. Nothing downloads at boot. A missing or incomplete directory
degrades this one slot with the path named. Decode runs off the event loop.

Parakeet v2 is English only, so `config.language` is ignored.

**Keyterms.** `stt.keyterms` doubles as the hotword list. With hotwords, the
recognizer uses `modified_beam_search`. Without them, it uses
`greedy_search`. Three rules come from measurement on real camera audio:

* The Parakeet vocabulary is cased. A hotword keeps the case the operator
  wrote. Uppercasing the terms measured 24 to 85 percent word error rate.
* A term that equals or contains the wake word is dropped. Biasing the wake
  word turned non-wake speech into "Hey Atlas" and defeated the transcript
  wake check (D-07).
* List proper names and device names only. A generic word such as "weather"
  pulls television speech toward it.

Biasing needs `bpe.vocab` beside the model files. Without that file, the
provider still works, with greedy decoding and no biasing, and it logs one
warning.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile
import weakref
from dataclasses import dataclass
from pathlib import Path
from typing import Any, AsyncIterator, Callable

from atlas.config import PARAKEET_BPE_VOCAB, PARAKEET_MODEL_FILES, SttConfig
from atlas.providers.base import FinalTranscript, PartialTranscript, SttError
from atlas.providers.boot import ProviderUnavailable
from atlas.providers.stt_faster_whisper import _TARGET_SAMPLE_RATE, _collect_frames, _to_float32_16k
from atlas.transports.base import SourceFormat

logger = logging.getLogger(__name__)

# D-07: the wake word. A keyterm that contains it is never a hotword.
_WAKE_WORD = "atlas"

# The same caps `config._parse_keyterms` enforces (D-09). Restated here
# because this module treats the terms as untrusted even after that check.
_MAX_HOTWORDS = 100
_MAX_HOTWORD_CHARS = 50

# sherpa-onnx reads ":" and "#" in a hotwords line as the start of a
# per-line score or threshold. An operator string must not set those.
_STRIPPED_CHARS = "\r\n\t:#"

_FIXED_BEAM_PATHS = 4
_GREEDY = "greedy_search"
_BEAM = "modified_beam_search"


@dataclass(frozen=True)
class ParakeetLoadSpec:
    """Everything the loader needs. This is the loader seam's one argument."""

    model_dir: str
    num_threads: int
    decoding_method: str
    hotwords_file: "str | None" = None
    hotwords_score: "float | None" = None
    bpe_vocab: "str | None" = None


def _load_sherpa_recognizer(spec: ParakeetLoadSpec) -> Any:
    """The real loader. `sherpa_onnx` is imported here, not at module scope,
    so selecting another provider never pays for the import."""
    import sherpa_onnx

    model_dir = Path(spec.model_dir)
    encoder, decoder, joiner, tokens = (str(model_dir / name) for name in PARAKEET_MODEL_FILES)
    kwargs: dict[str, Any] = {}
    if spec.hotwords_file is not None:
        kwargs.update(
            max_active_paths=_FIXED_BEAM_PATHS,
            hotwords_file=spec.hotwords_file,
            hotwords_score=spec.hotwords_score,
            modeling_unit="bpe",
            bpe_vocab=spec.bpe_vocab,
        )
    return sherpa_onnx.OfflineRecognizer.from_transducer(
        encoder=encoder,
        decoder=decoder,
        joiner=joiner,
        tokens=tokens,
        model_type="nemo_transducer",
        num_threads=spec.num_threads,
        decoding_method=spec.decoding_method,
        **kwargs,
    )


def hotword_lines(keyterms: "tuple[str, ...] | list[str]") -> list[str]:
    """Turn operator keyterms into hotword lines, one phrase each.

    Strips line breaks and the sherpa-onnx score markers, collapses
    whitespace, drops empty terms and any term that contains the wake word,
    removes duplicates in first-seen order, and applies the keyterm caps.
    Case is kept."""
    lines: list[str] = []
    seen: set[str] = set()
    for term in keyterms:
        cleaned = str(term)
        for char in _STRIPPED_CHARS:
            cleaned = cleaned.replace(char, " ")
        cleaned = " ".join(cleaned.split())
        if not cleaned or len(cleaned) > _MAX_HOTWORD_CHARS:
            continue
        if _WAKE_WORD in cleaned.casefold():
            continue
        if cleaned in seen:
            continue
        seen.add(cleaned)
        lines.append(cleaned)
        if len(lines) >= _MAX_HOTWORDS:
            break
    return lines


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except FileNotFoundError:
        pass


def _decode(recognizer: Any, audio: Any) -> str:
    """One blocking decode. Runs through `asyncio.to_thread`."""
    stream = recognizer.create_stream()
    stream.accept_waveform(_TARGET_SAMPLE_RATE, audio)
    recognizer.decode_stream(stream)
    return stream.result.text


class ParakeetStt:
    """Speech-to-text from a local Parakeet model. No account, no network
    call per turn, no download at boot.

    Parallel turns share one recognizer on a CPU-only host, and nobody has
    checked that one recognizer is safe to call from two threads at once.
    `_decode_lock` lets one decode run at a time, as `FasterWhisperStt`
    does. This class has no `hold_final` parameter on purpose: a
    whole-utterance provider must keep the controller's second-drain path.
    """

    # The provider gives no word until it is finalized. The controller reads
    # this flag: a Pi `vad.end` cannot wait for a word from this provider.
    words_before_finalize = False

    def __init__(
        self,
        config: SttConfig,
        *,
        load_recognizer: "Callable[[ParakeetLoadSpec], Any]" = _load_sherpa_recognizer,
    ) -> None:
        self._config = config
        self._decode_lock = asyncio.Lock()
        model_dir = Path(config.parakeet_model_dir)
        step = "scripts/fetch_models.py --only parakeet"
        if not model_dir.is_dir():
            raise ProviderUnavailable(
                f"No Parakeet model found at {config.parakeet_model_dir!r}. Run the model "
                f"provisioning step ({step}) before selecting this speech-to-text option."
            )
        for name in PARAKEET_MODEL_FILES:
            if not (model_dir / name).is_file():
                raise ProviderUnavailable(
                    f"The Parakeet model at {config.parakeet_model_dir!r} is missing {name}. "
                    f"Re-run the model provisioning step ({step})."
                )

        lines = hotword_lines(config.keyterms)
        vocab = model_dir / PARAKEET_BPE_VOCAB
        hotwords_path: "str | None" = None
        if lines and vocab.is_file():
            fd, hotwords_path = tempfile.mkstemp(prefix="atlas-parakeet-hotwords-", suffix=".txt")
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write("\n".join(lines) + "\n")
            weakref.finalize(self, _unlink_quietly, hotwords_path)
            spec = ParakeetLoadSpec(
                model_dir=str(model_dir),
                num_threads=config.parakeet_num_threads,
                decoding_method=_BEAM,
                hotwords_file=hotwords_path,
                hotwords_score=config.parakeet_hotwords_score,
                bpe_vocab=str(vocab),
            )
        else:
            if lines:
                logger.warning(
                    "Parakeet keyterm biasing is off: %s is missing, so %d keyterm(s) are "
                    "ignored and decoding is greedy. Re-run %s to build it.",
                    vocab,
                    len(lines),
                    step,
                )
            spec = ParakeetLoadSpec(
                model_dir=str(model_dir),
                num_threads=config.parakeet_num_threads,
                decoding_method=_GREEDY,
            )
        self._spec = spec

        try:
            self._recognizer = load_recognizer(spec)
        except ProviderUnavailable:
            if hotwords_path is not None:
                _unlink_quietly(hotwords_path)
            raise
        except Exception as exc:  # noqa: BLE001 -- any load failure degrades this slot
            if hotwords_path is not None:
                _unlink_quietly(hotwords_path)
            raise ProviderUnavailable(
                f"The Parakeet model at {config.parakeet_model_dir!r} could not be loaded "
                f"({exc}). Re-run the model provisioning step ({step}), or pick another "
                "speech-to-text option in Settings."
            ) from exc

    @property
    def spec(self) -> ParakeetLoadSpec:
        return self._spec

    async def stream(
        self,
        frames: AsyncIterator[bytes],
        source_format: SourceFormat,
        *,
        finalize: "asyncio.Event | None" = None,
    ) -> AsyncIterator[PartialTranscript | FinalTranscript]:
        """Buffer the turn, convert it to 16 kHz float32, decode once, and
        yield exactly one `FinalTranscript`. `finalize` ends the segment at
        once, as it does for `FasterWhisperStt`."""
        if source_format.channels != 1:
            raise SttError(
                f"local speech-to-text requires a single-channel source, got "
                f"{source_format.channels} channels"
            )

        raw = await _collect_frames(frames, finalize)
        audio = _to_float32_16k(raw, source_format)

        try:
            async with self._decode_lock:
                text = await asyncio.to_thread(_decode, self._recognizer, audio)
        except Exception as exc:
            raise SttError(f"local speech-to-text model failed: {exc}") from exc

        yield FinalTranscript(text=text or "")
