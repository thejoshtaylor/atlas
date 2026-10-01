"""The Parakeet (local) speech-to-text provider, driven against a fake
recognizer injected through the loader seam. No real model is used
(261001-mp8 D-12)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterator

import numpy as np
import pytest

from atlas.audio.alaw import pcm16_to_alaw
from atlas.config import PARAKEET_BPE_VOCAB, PARAKEET_MODEL_FILES, SttConfig
from atlas.providers.base import FinalTranscript, SttError
from atlas.providers.boot import ProviderUnavailable
from atlas.providers.stt_parakeet import ParakeetLoadSpec, ParakeetStt, hotword_lines
from atlas.transports.base import SourceFormat


class _FakeResult:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeStream:
    def __init__(self) -> None:
        self.sample_rate: "int | None" = None
        self.audio: "np.ndarray | None" = None
        self.result = _FakeResult("")

    def accept_waveform(self, sample_rate, audio) -> None:
        self.sample_rate = sample_rate
        self.audio = audio


class _FakeRecognizer:
    def __init__(self, text: str = "", error: Exception | None = None) -> None:
        self.text = text
        self.error = error
        self.streams: list[_FakeStream] = []

    def create_stream(self) -> _FakeStream:
        stream = _FakeStream()
        self.streams.append(stream)
        return stream

    def decode_stream(self, stream: _FakeStream) -> None:
        if self.error is not None:
            raise self.error
        stream.result = _FakeResult(self.text)


class _Loader:
    """Records the spec it was called with and hands back a fake recognizer."""

    def __init__(self, recognizer: _FakeRecognizer | None = None, error: Exception | None = None):
        self.recognizer = recognizer or _FakeRecognizer()
        self.error = error
        self.specs: list[ParakeetLoadSpec] = []
        self.hotwords_seen: list[str] = []

    def __call__(self, spec: ParakeetLoadSpec):
        self.specs.append(spec)
        if spec.hotwords_file is not None:
            with open(spec.hotwords_file, encoding="utf-8") as fh:
                self.hotwords_seen.append(fh.read())
        if self.error is not None:
            raise self.error
        return self.recognizer


def _model_dir(tmp_path, *, files=PARAKEET_MODEL_FILES, vocab: bool = True):
    model_dir = tmp_path / "parakeet"
    model_dir.mkdir()
    for name in files:
        (model_dir / name).write_bytes(b"x")
    if vocab:
        (model_dir / PARAKEET_BPE_VOCAB).write_text("a\t0\n", encoding="utf-8")
    return model_dir


def _config(model_dir, **overrides) -> SttConfig:
    return SttConfig(parakeet_model_dir=str(model_dir), **overrides)


async def _frames(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def test_missing_model_directory_raises_provider_unavailable_naming_path_and_step(tmp_path):
    missing = str(tmp_path / "nope")

    with pytest.raises(ProviderUnavailable) as excinfo:
        ParakeetStt(SttConfig(parakeet_model_dir=missing), load_recognizer=_Loader())

    message = str(excinfo.value)
    assert missing in message
    assert "provisioning" in message
    assert "scripts/fetch_models.py --only parakeet" in message


def test_a_missing_model_file_raises_provider_unavailable_naming_the_file(tmp_path):
    model_dir = _model_dir(tmp_path, files=PARAKEET_MODEL_FILES[:-1])

    with pytest.raises(ProviderUnavailable) as excinfo:
        ParakeetStt(_config(model_dir), load_recognizer=_Loader())

    assert PARAKEET_MODEL_FILES[-1] in str(excinfo.value)
    assert "--only parakeet" in str(excinfo.value)


def test_a_loader_that_raises_becomes_provider_unavailable(tmp_path):
    model_dir = _model_dir(tmp_path)

    with pytest.raises(ProviderUnavailable) as excinfo:
        ParakeetStt(
            _config(model_dir, keyterms=("Spotify",)),
            load_recognizer=_Loader(error=RuntimeError("bad onnx")),
        )

    assert "bad onnx" in str(excinfo.value)
    assert str(model_dir) in str(excinfo.value)


def test_keyterms_with_a_vocab_select_beam_search_and_a_case_kept_hotwords_file(tmp_path):
    model_dir = _model_dir(tmp_path)
    loader = _Loader()
    config = _config(
        model_dir,
        keyterms=("Atlas", "Kitchen Lamp", "hey atlas", "ATLAS desk", "Spotify"),
        parakeet_num_threads=4,
        parakeet_hotwords_score=2.5,
    )

    stt = ParakeetStt(config, load_recognizer=loader)

    spec = loader.specs[0]
    assert spec is stt.spec
    assert spec.decoding_method == "modified_beam_search"
    assert spec.num_threads == 4
    assert spec.hotwords_score == 2.5
    assert spec.bpe_vocab == str(model_dir / "bpe.vocab")
    assert spec.hotwords_file is not None
    assert loader.hotwords_seen == ["Kitchen Lamp\nSpotify\n"]


@pytest.mark.parametrize("keyterms", [(), ("Atlas", "hey ATLAS", "the Atlas lamp")])
def test_no_usable_keyterm_selects_greedy_search_with_no_hotword_arguments(tmp_path, keyterms):
    model_dir = _model_dir(tmp_path)
    loader = _Loader()

    ParakeetStt(_config(model_dir, keyterms=keyterms), load_recognizer=loader)

    spec = loader.specs[0]
    assert spec.decoding_method == "greedy_search"
    assert spec.hotwords_file is None
    assert spec.hotwords_score is None
    assert spec.bpe_vocab is None


def test_keyterms_without_a_vocab_fall_back_to_greedy_and_warn_without_the_terms(
    tmp_path, caplog
):
    model_dir = _model_dir(tmp_path, vocab=False)
    loader = _Loader()

    with caplog.at_level(logging.WARNING, logger="atlas.providers.stt_parakeet"):
        ParakeetStt(
            _config(model_dir, keyterms=("Secret Name", "Other Name")), load_recognizer=loader
        )

    assert loader.specs[0].decoding_method == "greedy_search"
    assert loader.specs[0].hotwords_file is None
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert len(warnings) == 1
    text = warnings[0].getMessage()
    assert str(model_dir / "bpe.vocab") in text
    assert "2" in text
    assert "Secret Name" not in text and "Other Name" not in text


def test_hotword_lines_sanitizes_dedupes_and_caps():
    lines = hotword_lines(
        ["Kitchen\nLamp", "Score: 3.0", "#comment", "  ", ":", "Spotify", "Spotify", "ATLAS desk"]
    )

    assert lines == ["Kitchen Lamp", "Score 3.0", "comment", "Spotify"]
    assert len(hotword_lines([f"term{i}" for i in range(150)])) == 100
    assert hotword_lines(["x" * 51]) == []


def test_the_hotwords_file_exists_after_construction_and_is_removed_on_finalize(tmp_path):
    import gc

    model_dir = _model_dir(tmp_path)
    loader = _Loader()
    stt = ParakeetStt(_config(model_dir, keyterms=("Spotify",)), load_recognizer=loader)
    path = stt.spec.hotwords_file
    assert path is not None and os.path.exists(path)

    del stt
    gc.collect()

    assert not os.path.exists(path)


@pytest.mark.asyncio
async def test_the_recognizer_loads_once_at_construction_not_per_turn(tmp_path):
    loader = _Loader(_FakeRecognizer(text="hello"))
    stt = ParakeetStt(_config(_model_dir(tmp_path)), load_recognizer=loader)
    assert len(loader.specs) == 1

    for _ in range(2):
        async for _event in stt.stream(_frames(b"\x00\x00" * 100), SourceFormat("pcm", 16000)):
            pass

    assert len(loader.specs) == 1


@pytest.mark.asyncio
async def test_8khz_alaw_reaches_the_recognizer_as_1600_float32_samples_at_16khz(tmp_path):
    recognizer = _FakeRecognizer(text="Turn on the Kitchen Lamp.")
    stt = ParakeetStt(_config(_model_dir(tmp_path)), load_recognizer=_Loader(recognizer))
    pcm16 = (np.arange(800, dtype=np.int16) - 400).tobytes()  # 100 ms at 8 kHz

    events = [
        event
        async for event in stt.stream(_frames(pcm16_to_alaw(pcm16)), SourceFormat("alaw", 8000))
    ]

    assert events == [FinalTranscript(text="Turn on the Kitchen Lamp.")]
    stream = recognizer.streams[0]
    assert stream.sample_rate == 16000
    assert stream.audio is not None
    assert len(stream.audio) == 1600
    assert stream.audio.dtype == np.float32


@pytest.mark.asyncio
async def test_16khz_pcm_reaches_the_recognizer_with_no_resample(tmp_path):
    recognizer = _FakeRecognizer(text="hello there")
    stt = ParakeetStt(_config(_model_dir(tmp_path)), load_recognizer=_Loader(recognizer))
    pcm16 = (np.arange(1600, dtype=np.int16) - 800).tobytes()

    events = [e async for e in stt.stream(_frames(pcm16), SourceFormat("pcm", 16000))]

    assert events == [FinalTranscript(text="hello there")]
    expected = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    assert np.array_equal(recognizer.streams[0].audio, expected)


@pytest.mark.asyncio
async def test_empty_text_gives_one_empty_final_transcript(tmp_path):
    stt = ParakeetStt(_config(_model_dir(tmp_path)), load_recognizer=_Loader())

    events = [e async for e in stt.stream(_frames(b"\x00\x00" * 10), SourceFormat("pcm", 16000))]

    assert events == [FinalTranscript(text="")]


@pytest.mark.asyncio
async def test_a_decode_error_becomes_stt_error(tmp_path):
    recognizer = _FakeRecognizer(error=RuntimeError("onnx exploded"))
    stt = ParakeetStt(_config(_model_dir(tmp_path)), load_recognizer=_Loader(recognizer))

    with pytest.raises(SttError) as excinfo:
        async for _ in stt.stream(_frames(b"\x00\x00" * 10), SourceFormat("pcm", 16000)):
            pass

    assert "onnx exploded" in str(excinfo.value)


@pytest.mark.asyncio
async def test_a_two_channel_source_is_refused(tmp_path):
    stt = ParakeetStt(_config(_model_dir(tmp_path)), load_recognizer=_Loader())

    with pytest.raises(SttError):
        async for _ in stt.stream(
            _frames(b"\x00\x00"), SourceFormat("pcm", 16000, channels=2)
        ):
            pass


@pytest.mark.asyncio
async def test_finalize_ends_a_source_that_never_ends(tmp_path):
    recognizer = _FakeRecognizer(text="stop now")
    stt = ParakeetStt(_config(_model_dir(tmp_path)), load_recognizer=_Loader(recognizer))
    finalize = asyncio.Event()

    async def _endless() -> AsyncIterator[bytes]:
        yield b"\x00\x00" * 160
        finalize.set()
        await asyncio.Event().wait()
        yield b""  # pragma: no cover

    events = [
        e
        async for e in stt.stream(_endless(), SourceFormat("pcm", 16000), finalize=finalize)
    ]

    assert events == [FinalTranscript(text="stop now")]


def test_the_signature_has_no_hold_final_parameter():
    import inspect

    assert "hold_final" not in inspect.signature(ParakeetStt.stream).parameters
