"""Local speech-to-text via `faster-whisper` -- no cloud account, no
per-turn network call, no download at boot (PROV-06, D-09, D-11).

This is segment-level transcription, not true incremental streaming: a
CPU-only local model runs behind live speech, and `faster_whisper` itself
returns a whole utterance's segments only once decoding finishes. This is
the same honesty this project already states for xAI's REST-only
text-to-speech transport (`tts_xai.py`), applied here on the input side.

The model is loaded once, at construction, from a directory a documented
provisioning step (plan 07-04) has already populated -- never fetched from
the network. A missing directory degrades this slot by name rather than
falling back to a download; a first boot that quietly pulls hundreds of
megabytes from the internet is exactly what D-11 exists to forbid.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, AsyncIterator, Callable

import numpy as np

from spire_voice.audio.alaw import alaw_to_pcm16
from spire_voice.config import SttConfig
from spire_voice.providers.base import FinalTranscript, PartialTranscript, SttError
from spire_voice.providers.boot import ProviderUnavailable
from spire_voice.transports.base import SourceFormat

# `faster_whisper`'s own feature extractor expects 16 kHz mono -- the same
# rate `transports/camera.py`'s wake-detector resampler targets, restated
# here rather than imported, since that module's resampler is PyAV-based
# and stateful, and this module needs neither.
_TARGET_SAMPLE_RATE = 16000


def _load_faster_whisper_model(config: SttConfig) -> Any:
    """The real loader every deployment uses. Imported inside this
    function, not at module scope, so selecting the xAI entry never pays
    the cost of importing `faster_whisper`/`ctranslate2` at all.

    `local_files_only=True` is the one knob that turns a cache-miss into a
    raised error rather than a silent fetch from the Hugging Face hub --
    D-11 forbids the fetch outright.
    """
    from faster_whisper import WhisperModel

    return WhisperModel(
        config.local_model_dir,
        device="cpu",
        compute_type=config.local_compute_type,
        local_files_only=True,
    )


def _resample_pcm16(pcm16: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resample of 16-bit signed PCM.

    Exact for the one ratio this module ever needs -- 8 kHz camera audio to
    16 kHz -- and simple enough to carry with no dependency beyond the
    `numpy` this codebase already declares, rather than reaching for
    PyAV's stateful, frame-oriented `AudioResampler`
    (`transports/camera.py`) for a single one-shot buffer.
    """
    if from_rate == to_rate or not pcm16:
        return pcm16
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float64)
    duration_s = len(samples) / from_rate
    new_length = max(1, round(duration_s * to_rate))
    old_indices = np.arange(len(samples))
    new_indices = np.linspace(0, len(samples) - 1, new_length)
    resampled = np.interp(new_indices, old_indices, samples)
    return resampled.astype(np.int16).tobytes()


def _pcm16_to_float32(pcm16: bytes) -> "np.ndarray":
    """16-bit signed PCM to the normalized float32 array
    `WhisperModel.transcribe` expects."""
    samples = np.frombuffer(pcm16, dtype="<i2").astype(np.float32)
    return samples / 32768.0


def _run_transcribe(model: Any, audio: "np.ndarray", language: str) -> list:
    """Materialize every segment before returning.

    Runs off the event loop (via `asyncio.to_thread`, in `stream()` below)
    because `WhisperModel.transcribe` is a blocking CPU call. This class
    needs to know which segment is last before it can decide between
    `PartialTranscript` and `FinalTranscript`, and a lazily-evaluated
    generator can't answer that without being fully consumed anyway.
    """
    segments, _info = model.transcribe(audio, language=language)
    return list(segments)


class FasterWhisperStt:
    """Speech-to-text via a local `faster-whisper` model -- no account, no
    per-turn network call, no download at boot (D-09, D-11)."""

    def __init__(
        self,
        config: SttConfig,
        *,
        load_model: "Callable[[SttConfig], Any]" = _load_faster_whisper_model,
    ) -> None:
        self._config = config
        if not os.path.isdir(config.local_model_dir):
            raise ProviderUnavailable(
                f"No faster-whisper model found at {config.local_model_dir!r}. Run the "
                "model provisioning step to place one there before selecting this "
                "speech-to-text option."
            )
        self._model = load_model(config)

    async def stream(
        self, frames: AsyncIterator[bytes], source_format: SourceFormat
    ) -> AsyncIterator[PartialTranscript | FinalTranscript]:
        """Buffer the whole turn, decode/resample to 16 kHz PCM16 when the
        source is A-law, and transcribe once. See the module docstring for
        why this is segment-level, not truly incremental.
        """
        chunks = [chunk async for chunk in frames]
        raw = b"".join(chunks)

        if source_format.encoding == "alaw":
            pcm16 = alaw_to_pcm16(raw)
            pcm16 = _resample_pcm16(pcm16, source_format.sample_rate, _TARGET_SAMPLE_RATE)
        elif source_format.encoding == "pcm":
            pcm16 = raw
        else:
            raise SttError(
                f"local speech-to-text cannot handle source encoding {source_format.encoding!r}"
            )

        audio = _pcm16_to_float32(pcm16)

        try:
            segments = await asyncio.to_thread(
                _run_transcribe, self._model, audio, self._config.language
            )
        except Exception as exc:
            raise SttError(f"local speech-to-text model failed: {exc}") from exc

        if not segments:
            yield FinalTranscript(text="")
            return

        for segment in segments[:-1]:
            yield PartialTranscript(text=segment.text)
        yield FinalTranscript(text=segments[-1].text)
