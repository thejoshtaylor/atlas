"""Local text-to-speech via Piper -- no account, no per-turn network call,
optional to install (PROV-06, D-09, D-10).

Piper's own inference code is GPL-3.0-or-later (D-10) -- unlike
`faster-whisper` (D-09, a default dependency), it ships behind the
`piper` extra, and `pip install atlas` alone never pulls it in. A
default install that selects this provider anyway must degrade cleanly
by name, exactly like a missing model file, rather than crashing the
boot on a raw `ImportError` -- that is the whole reason the `piper`
import lives inside `_load_piper_voice`, not at this module's top level.

Renders a whole utterance in one call (D-05) -- this class owns no
chunking loop of its own. `providers/batch_tts_adapter.py::BatchTtsAdapter`
is the one place that chunks a batch provider's output; a second chunking
loop here would be exactly the drift D-07 forbids.
"""

from __future__ import annotations

import asyncio
import os
from typing import Any, Callable

import numpy as np

from atlas.audio.alaw import pcm16_to_alaw
from atlas.config import TtsConfig
from atlas.providers.boot import ProviderUnavailable
from atlas.providers.tts_xai import SinkFormat


def _load_piper_voice(config: TtsConfig) -> Any:
    """The real loader every deployment uses. Imported inside this
    function, not at module scope, so a default install -- which does not
    carry the `piper` extra (D-10) -- never fails importing this module at
    all. The failure surfaces here instead, as the same `ProviderUnavailable`
    a missing model produces, naming the extra to install rather than
    leaking a raw `ImportError` that would stop the boot.
    """
    try:
        from piper import PiperVoice
    except ImportError as exc:
        raise ProviderUnavailable(
            "The 'piper' extra is not installed. Run `pip install "
            "'atlas[piper]'` before selecting Piper as the text-to-speech "
            "option."
        ) from exc

    return PiperVoice.load(config.piper_voice_path, config_path=config.piper_config_path)


def _resample_pcm16(pcm16: bytes, from_rate: int, to_rate: int) -> bytes:
    """Linear-interpolation resample of 16-bit signed PCM.

    A small, self-contained duplicate of `stt_faster_whisper.py`'s own
    helper, deliberately -- the same independence `audio/alaw.py`'s module
    docstring states for why it supplies pure functions rather than
    borrowing a decode context from a source that has one: this class
    needs no coupling to the speech-to-text module to do one thing neither
    of them owns.
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


def _render_pcm16(voice: Any, text: str) -> "tuple[bytes, int]":
    """Render `text` through `voice` in one call: every audio chunk Piper
    yields, joined as 16-bit PCM, plus the native sample rate those chunks
    carry. Runs off the event loop (`asyncio.to_thread`, in
    `synthesize_once` below) because Piper's own synthesis is a blocking
    CPU call.
    """
    chunks = list(voice.synthesize(text))
    if not chunks:
        return b"", 0
    pcm16 = b"".join(chunk.audio_int16_bytes for chunk in chunks)
    return pcm16, chunks[0].sample_rate


def _render_for_sink(voice: Any, text: str, sample_rate: int, codec: str) -> bytes:
    """Quick task 260924-4is (D3): render, resample, and the optional
    A-law encode, all in the one worker-thread call `synthesize_once`
    below makes -- `_render_pcm16`/`_resample_pcm16`/`pcm16_to_alaw` are
    looked up as this module's own globals at call time, so an existing
    test's `monkeypatch.setattr` on any of them still applies here."""
    pcm16, native_rate = _render_pcm16(voice, text)
    pcm16 = _resample_pcm16(pcm16, native_rate, sample_rate)
    if codec == "alaw":
        return pcm16_to_alaw(pcm16)
    return pcm16


class PiperTts:
    """Speech synthesis via a local Piper voice -- no account, no per-turn
    network call (D-09, D-10)."""

    def __init__(
        self,
        config: TtsConfig,
        *,
        load_voice: "Callable[[TtsConfig], Any]" = _load_piper_voice,
    ) -> None:
        self._config = config
        if not os.path.isfile(config.piper_voice_path):
            raise ProviderUnavailable(
                f"No Piper voice found at {config.piper_voice_path!r}. Run the model "
                "provisioning step to place one there before selecting this "
                "text-to-speech option."
            )
        if not os.path.isfile(config.piper_config_path):
            raise ProviderUnavailable(
                f"No Piper voice configuration found at {config.piper_config_path!r}. "
                "Run the model provisioning step to place one there before selecting "
                "this text-to-speech option."
            )
        # CR-04 (code review): the same guard `stt_faster_whisper.py`
        # carries, for the same reason. `_load_piper_voice` already turns
        # a missing `piper` extra into `ProviderUnavailable`, but a voice
        # file that exists and is truncated or corrupt (an interrupted
        # `scripts/fetch_models.py` run) fails inside onnxruntime instead,
        # with an exception this module had no branch for -- and that
        # stopped the whole boot rather than degrading this one slot,
        # which is the admin lockout D-04 exists to prevent.
        try:
            self._voice = load_voice(config)
        except ProviderUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001 -- any load failure degrades this slot
            raise ProviderUnavailable(
                f"The Piper voice at {config.piper_voice_path!r} could not be loaded "
                f"({exc}). Re-run the model provisioning step, or pick another "
                "text-to-speech option in Settings."
            ) from exc

    def browser_sink(self) -> SinkFormat:
        """The same browser-facing default `XaiTts.browser_sink` states --
        Web Audio-playable PCM, never A-law."""
        return SinkFormat(codec=self._config.browser_codec, sample_rate=self._config.browser_sample_rate)

    async def synthesize_once(self, text: str, sink: "SinkFormat | None" = None) -> bytes:
        """Render `text` through the voice in one call, resample to
        `sink`'s rate, and encode to A-law when `sink` asks for it.

        `sink` is a parameter this method reads, never a codec/rate this
        module assumes -- the same discipline `tts_xai.py`'s own docstring
        states for why the camera-facing A-law path and the
        Web-Audio-facing PCM path cannot share one hardcoded shape.
        """
        sink = sink or self.browser_sink()
        return await asyncio.to_thread(_render_for_sink, self._voice, text, sink.sample_rate, sink.codec)
