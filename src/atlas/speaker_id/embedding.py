"""Speaker-embedding inference: one PCM16 buffer in, one L2-normalized
embedding vector out, through sherpa-onnx (D-05).

This is speaker identification, not `atlas.speaker` -- that package is the
audio *output* path (the FIFO, the ffmpeg supervisor, Tapo talk). The two
share a similar name and nothing else; do not import one expecting the
other.

Inference never runs on the event loop. `SherpaEmbedder` itself is
synchronous, native-call-per-invocation code -- exactly the shape
`sources/runner.py`'s own `_detector_executor` comment describes for the
wake detector and the barge-in energy reader: a stateful native model, and
a cancelled `await` does not stop a call already running in a thread.
`EmbeddingWorker` is the one-worker executor that keeps every call to a
`SpeakerEmbedder` off the loop thread, mirroring that same discipline.
"""

from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Protocol

import numpy as np

# 16 kHz PCM16 mono is this module's only supported input shape -- the same
# convention `audio/energy.py` fixes for full-scale amplitude, and the only
# rate the two candidate models (D-05) were trained on (11-RESEARCH.md).
SAMPLE_RATE = 16000

# Bounds one `embed()` call's native-code work: this codebase's own turns
# are seconds long, never minutes, so a caller (or a corrupted buffer) that
# hands this function half an hour of audio is a bug or an attack, not a
# real embedding request. Checked before any native call, in bytes, so the
# rejection costs nothing.
MAX_EMBED_SECONDS = 30.0

# PCM16 full scale, matching `audio/energy.py::_INT16_FULL_SCALE` exactly --
# a different divisor here would silently retrain what "loud" means for
# this module alone.
_PCM16_FULL_SCALE = 32768.0


class SpeakerModelError(Exception):
    """Raised when a speaker embedding model cannot be loaded -- a missing
    file or a model sherpa-onnx itself refuses -- naming the path and
    `scripts/fetch_models.py`, never a silent fallback to no embedder."""


class SpeakerAudioTooShort(ValueError):
    """Raised when a buffer is within the byte-length bounds `embed()`
    checks up front, but still too short for the extractor to produce an
    embedding from (`SpeakerEmbeddingExtractor.is_ready` is false)."""


def pcm16_to_float32(pcm16_mono: bytes) -> "np.ndarray":
    """Decode a little-endian, 16-bit signed, mono PCM buffer into a
    `float32` array scaled by the same full-scale divisor
    `audio/energy.py` uses -- the two modules must never disagree about
    what full scale means for this codec."""
    samples = np.frombuffer(pcm16_mono, dtype="<i2").astype(np.float32)
    return samples / _PCM16_FULL_SCALE


class SpeakerEmbedder(Protocol):
    """What `EmbeddingWorker` -- and every later plan's gate/matching code
    -- needs from an embedder, real or fake. `dim` is the embedding
    vector's length, fixed once the underlying model is loaded."""

    dim: int

    def embed(self, pcm16_mono: bytes) -> "np.ndarray":
        """Return an L2-normalized `float32` embedding for `pcm16_mono`, a
        16 kHz mono PCM16 buffer. Raises `ValueError` for a buffer outside
        this module's accepted bounds, `SpeakerAudioTooShort` for one
        inside those bounds but still too short to embed."""
        ...


class SherpaEmbedder:
    """A `SpeakerEmbedder` backed by `sherpa_onnx.SpeakerEmbeddingExtractor`
    (11-RESEARCH.md's confirmed API shape).

    `sherpa_onnx` is imported inside `__init__`, not at module import time,
    so importing this module never requires the package to be installed --
    a test that only needs `FakeEmbedder` (`tests/speaker_fakes.py`) never
    pays for it.
    """

    def __init__(self, model_path: "str | Path", *, num_threads: int = 1) -> None:
        import sherpa_onnx  # noqa: PLC0415 -- see the class docstring

        path = Path(model_path)
        if not path.is_file():
            raise SpeakerModelError(
                f"no speaker embedding model at {path} -- run "
                "`scripts/fetch_models.py --only speaker-id` to download it"
            )

        config = sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(path), num_threads=num_threads, debug=False, provider="cpu"
        )
        if not config.validate():
            raise SpeakerModelError(
                f"sherpa-onnx rejected the speaker embedding model at {path} -- see "
                "scripts/fetch_models.py for how it is fetched and verified"
            )

        self._extractor = sherpa_onnx.SpeakerEmbeddingExtractor(config)
        self.dim: int = self._extractor.dim

    def embed(self, pcm16_mono: bytes) -> "np.ndarray":
        if len(pcm16_mono) == 0:
            raise ValueError("embed() got an empty buffer")
        if len(pcm16_mono) % 2 != 0:
            raise ValueError(
                f"embed() got an odd byte count ({len(pcm16_mono)}) -- PCM16 samples are "
                "2 bytes each"
            )
        max_bytes = int(MAX_EMBED_SECONDS * SAMPLE_RATE) * 2
        if len(pcm16_mono) > max_bytes:
            raise ValueError(
                f"embed() got {len(pcm16_mono)} bytes, more than MAX_EMBED_SECONDS="
                f"{MAX_EMBED_SECONDS}s of {SAMPLE_RATE} Hz PCM16 ({max_bytes} bytes)"
            )

        samples = pcm16_to_float32(pcm16_mono)
        stream = self._extractor.create_stream()
        stream.accept_waveform(sample_rate=SAMPLE_RATE, waveform=samples)
        stream.input_finished()

        if not self._extractor.is_ready(stream):
            duration_s = len(samples) / SAMPLE_RATE
            raise SpeakerAudioTooShort(
                f"embed() got {len(samples)} samples ({duration_s:.3f}s), too short for the "
                "speaker embedding extractor to produce an embedding from"
            )

        embedding = np.asarray(self._extractor.compute(stream), dtype=np.float32)
        norm = float(np.linalg.norm(embedding))
        if norm > 0.0:
            embedding = embedding / norm
        return embedding


class EmbeddingWorker:
    """Runs every call to one `SpeakerEmbedder` on a single dedicated
    thread, never the event loop thread and never `asyncio.to_thread`'s
    shared default pool -- the same reasoning `sources/runner.py`'s
    `_detector_executor` states for the wake detector: the underlying
    model is stateful and not safe to call from two threads at once, and a
    cancelled `await` does not stop a call already running in a thread, so
    a second call must queue behind the first rather than race it.
    """

    def __init__(self, embedder: "SpeakerEmbedder") -> None:
        self._embedder = embedder
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="atlas-speaker-id")

    def submit(self, pcm16_mono: bytes) -> "asyncio.Future[np.ndarray]":
        loop = asyncio.get_running_loop()
        return loop.run_in_executor(self._executor, self._embedder.embed, pcm16_mono)

    async def embed(self, pcm16_mono: bytes) -> "np.ndarray":
        return await self.submit(pcm16_mono)

    def close(self) -> None:
        """Wait for any in-flight call to finish, then release the worker
        thread -- never abandon a call mid-flight the way
        `cancel_futures=True` alone would."""
        self._executor.shutdown(wait=True, cancel_futures=True)
