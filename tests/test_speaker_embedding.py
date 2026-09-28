"""`src/atlas/speaker_id/embedding.py`: `SherpaEmbedder` against both real
speaker-embedding models (D-05's spike candidates), `EmbeddingWorker`'s
off-loop discipline against `FakeEmbedder`, and the input-bounds checks
`embed()` runs before any native call.

The real-model tests skip only when the model file itself is absent --
`scripts/fetch_models.py --only speaker-id` downloads both, and this
repository never vendors them (`models/` is gitignored). The worker and
`FakeEmbedder` tests always run; they need no model file at all.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import numpy as np
import pytest

from atlas.speaker_id.embedding import (
    MAX_EMBED_SECONDS,
    SAMPLE_RATE,
    EmbeddingWorker,
    SpeakerAudioTooShort,
    SpeakerModelError,
    SherpaEmbedder,
    pcm16_to_float32,
)

from tests.speaker_fakes import FakeEmbedder

_MODEL_ROOT = Path(__file__).resolve().parent.parent / "models" / "speaker-id"
_CAMPPLUS_PATH = _MODEL_ROOT / "3dspeaker_speech_campplus_sv_en_voxceleb_16k.onnx"
_TITANET_PATH = _MODEL_ROOT / "nemo_en_titanet_small.onnx"

_SKIP_REASON = (
    "speaker embedding model not present at {path} -- run "
    "`.venv/bin/python scripts/fetch_models.py --only speaker-id` (or `curl -L` the "
    "URL in 11-01-PLAN.md's <interfaces>) to download it before running this test"
)


def _synthetic_voiced_signal(duration_s: float, *, sample_rate: int = SAMPLE_RATE) -> bytes:
    """Two seconds (by default) of a synthetic voiced-range tone, as PCM16
    mono bytes -- not real speech, but real enough waveform energy for
    `SherpaEmbedder` to produce a real embedding from."""
    n = int(duration_s * sample_rate)
    t = np.arange(n) / sample_rate
    tone = 0.2 * np.sin(2 * np.pi * 180.0 * t) + 0.05 * np.sin(2 * np.pi * 360.0 * t)
    samples = (tone * 32767.0).astype("<i2")
    return samples.tobytes()


def _constant_pcm16(value: int, count: int) -> bytes:
    samples = np.full(count, value, dtype="<i2")
    return samples.tobytes()


# --- pcm16_to_float32 ----------------------------------------------------


def test_pcm16_to_float32_scales_by_full_scale():
    pcm = _constant_pcm16(16384, 4)  # half of int16 full scale
    result = pcm16_to_float32(pcm)
    assert result.dtype == np.float32
    assert result == pytest.approx(0.5, abs=1e-6)


# --- SherpaEmbedder: real models -----------------------------------------


@pytest.mark.parametrize(
    "model_path",
    [
        pytest.param(_CAMPPLUS_PATH, id="campplus"),
        pytest.param(_TITANET_PATH, id="titanet_small"),
    ],
)
def test_sherpa_embedder_embeds_a_voiced_signal_to_a_unit_vector(model_path):
    if not model_path.is_file():
        pytest.skip(_SKIP_REASON.format(path=model_path))

    embedder = SherpaEmbedder(model_path)
    pcm = _synthetic_voiced_signal(2.0)

    embedding = embedder.embed(pcm)

    assert embedding.dtype == np.float32
    assert embedding.shape == (embedder.dim,)
    assert np.linalg.norm(embedding) == pytest.approx(1.0, abs=1e-5)


@pytest.mark.parametrize(
    "model_path",
    [
        pytest.param(_CAMPPLUS_PATH, id="campplus"),
        pytest.param(_TITANET_PATH, id="titanet_small"),
    ],
)
def test_sherpa_embedder_is_deterministic_for_the_same_buffer(model_path):
    if not model_path.is_file():
        pytest.skip(_SKIP_REASON.format(path=model_path))

    embedder = SherpaEmbedder(model_path)
    pcm = _synthetic_voiced_signal(2.0)

    first = embedder.embed(pcm)
    second = embedder.embed(pcm)

    cosine = float(np.dot(first, second))
    assert cosine > 0.999


def test_sherpa_embedder_on_a_missing_path_raises_speaker_model_error(tmp_path):
    missing = tmp_path / "does-not-exist.onnx"

    with pytest.raises(SpeakerModelError) as excinfo:
        SherpaEmbedder(missing)

    message = str(excinfo.value)
    assert str(missing) in message
    assert "fetch_models.py" in message


# --- embed(): input bounds, checked before any native call ---------------


@pytest.mark.parametrize(
    "model_path",
    [
        pytest.param(_CAMPPLUS_PATH, id="campplus"),
        pytest.param(_TITANET_PATH, id="titanet_small"),
    ],
)
class TestEmbedInputBounds:
    def test_empty_buffer_raises_value_error(self, model_path):
        if not model_path.is_file():
            pytest.skip(_SKIP_REASON.format(path=model_path))
        embedder = SherpaEmbedder(model_path)
        with pytest.raises(ValueError):
            embedder.embed(b"")

    def test_odd_byte_count_raises_value_error(self, model_path):
        if not model_path.is_file():
            pytest.skip(_SKIP_REASON.format(path=model_path))
        embedder = SherpaEmbedder(model_path)
        with pytest.raises(ValueError):
            embedder.embed(b"\x00\x01\x02")

    def test_more_than_max_embed_seconds_raises_value_error(self, model_path):
        if not model_path.is_file():
            pytest.skip(_SKIP_REASON.format(path=model_path))
        embedder = SherpaEmbedder(model_path)
        too_long = _synthetic_voiced_signal(MAX_EMBED_SECONDS + 1.0)
        with pytest.raises(ValueError):
            embedder.embed(too_long)

    def test_a_buffer_too_short_for_the_extractor_raises_speaker_audio_too_short(
        self, model_path
    ):
        if not model_path.is_file():
            pytest.skip(_SKIP_REASON.format(path=model_path))
        embedder = SherpaEmbedder(model_path)
        # A handful of samples -- inside the byte-length bounds, but far too
        # short for the extractor's own `is_ready` to ever return True.
        too_short = _constant_pcm16(0, 10)
        with pytest.raises(SpeakerAudioTooShort):
            embedder.embed(too_short)


# --- EmbeddingWorker: off-loop discipline, against FakeEmbedder ----------


async def test_embedding_worker_runs_on_a_dedicated_thread_never_the_loop_thread():
    embedder = FakeEmbedder(dim=8)
    worker = EmbeddingWorker(embedder)
    try:
        embedding = await worker.embed(_constant_pcm16(1000, 100))
        assert embedding.shape == (8,)
        assert len(embedder.thread_names) == 1
        assert embedder.thread_names[0].startswith("atlas-speaker-id")
    finally:
        worker.close()


async def test_embedding_worker_serializes_concurrent_calls_on_one_worker():
    embedder = FakeEmbedder(dim=8)
    worker = EmbeddingWorker(embedder)
    try:
        results = await asyncio.gather(
            worker.embed(_constant_pcm16(100, 50)),
            worker.embed(_constant_pcm16(2000, 50)),
            worker.embed(_constant_pcm16(9000, 50)),
        )
        assert len(results) == 3
        # Every call ran on the same single dedicated thread.
        assert len(set(embedder.thread_names)) == 1
        assert embedder.thread_names[0].startswith("atlas-speaker-id")
    finally:
        worker.close()


async def test_embedding_worker_close_waits_for_an_in_flight_call():
    embedder = FakeEmbedder(dim=8)
    worker = EmbeddingWorker(embedder)
    task = asyncio.ensure_future(worker.embed(_constant_pcm16(500, 100)))
    result = await task
    worker.close()  # must not raise, and must not orphan the completed call
    assert result.shape == (8,)


def test_embedding_worker_close_is_safe_with_no_calls_made():
    embedder = FakeEmbedder(dim=8)
    worker = EmbeddingWorker(embedder)
    worker.close()


# --- FakeEmbedder: the vector rule (11-01-PLAN.md <behavior>) ------------


def test_fake_embedder_maps_the_same_bucket_to_the_same_vector():
    embedder = FakeEmbedder(dim=8)
    first = embedder.embed(_constant_pcm16(5000, 100))
    second = embedder.embed(_constant_pcm16(5050, 100))  # same 1000-count bucket

    assert np.array_equal(first, second)
    assert np.linalg.norm(first) == pytest.approx(1.0, abs=1e-6)


def test_fake_embedder_maps_different_buckets_to_different_vectors():
    embedder = FakeEmbedder(dim=8)
    voice_a = embedder.embed(_constant_pcm16(1000, 100))
    voice_b = embedder.embed(_constant_pcm16(9000, 100))

    assert not np.array_equal(voice_a, voice_b)


def test_fake_embedder_records_every_call():
    embedder = FakeEmbedder(dim=8)
    pcm_one = _constant_pcm16(1000, 100)
    pcm_two = _constant_pcm16(2000, 100)

    embedder.embed(pcm_one)
    embedder.embed(pcm_two)

    assert embedder.calls == [pcm_one, pcm_two]
    assert len(embedder.thread_names) == 2
