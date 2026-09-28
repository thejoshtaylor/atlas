"""`FakeEmbedder`: the deterministic embedder every later speaker-id test
(this plan's own worker/config tests, and every plan after it) builds on,
mirroring `tests/conftest.py`'s existing `Fake*` fixtures for the other
providers.

No `sherpa_onnx` import here at all -- this fake never touches the real
package, so tests that only need it never pay the real model's load cost
and never need a model file on disk.
"""

from __future__ import annotations

import threading

import numpy as np

from atlas.speaker_id.embedding import pcm16_to_float32


def _bucket_vector(bucket: int, dim: int) -> "np.ndarray":
    """A fixed unit vector for `bucket`, deterministic across calls and
    across processes -- the same bucket always yields the same vector, and
    different buckets yield (with overwhelming probability) different,
    non-parallel vectors. Two "voices" in a test are two known vectors."""
    rng = np.random.RandomState(seed=bucket & 0xFFFFFFFF)
    vector = rng.standard_normal(dim).astype(np.float32)
    norm = float(np.linalg.norm(vector))
    if norm > 0.0:
        vector = vector / norm
    return vector


class FakeEmbedder:
    """A `SpeakerEmbedder` (`atlas.speaker_id.embedding.SpeakerEmbedder`)
    that never touches sherpa-onnx. `embed()` maps the input buffer's mean
    PCM16 sample value into a 1000-count bucket and returns that bucket's
    fixed unit vector (see `_bucket_vector`) -- constant-value audio at a
    known level always embeds to a known vector, so a test can construct
    "two speakers" by choosing two sample values far enough apart to land
    in different buckets.

    Records every call's raw buffer and the name of the thread `embed()`
    ran on, in call order -- `EmbeddingWorker` tests read `thread_names` to
    prove inference ran off the event loop thread.
    """

    def __init__(self, dim: int = 8) -> None:
        self.dim = dim
        self.calls: "list[bytes]" = []
        self.thread_names: "list[str]" = []

    def embed(self, pcm16_mono: bytes) -> "np.ndarray":
        self.calls.append(pcm16_mono)
        self.thread_names.append(threading.current_thread().name)

        if len(pcm16_mono) == 0:
            mean_value = 0.0
        else:
            raw_samples = np.frombuffer(pcm16_mono, dtype="<i2").astype(np.float64)
            mean_value = float(np.mean(raw_samples))
        bucket = int(mean_value // 1000.0)
        return _bucket_vector(bucket, self.dim)


# Re-exported so a test that wants the real scaling convention (and not
# just the fake) can import both from one module.
__all__ = ["FakeEmbedder", "pcm16_to_float32"]
