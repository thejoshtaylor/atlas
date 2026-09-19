"""Local providers -- faster-whisper speech-to-text and Piper text-to-speech
-- driven by real classes against fake models, never a fake subclass
(PROV-06, D-09, D-10, D-11).
"""

from __future__ import annotations

from typing import AsyncIterator

import numpy as np
import pytest

from spire_voice.audio.alaw import pcm16_to_alaw
from spire_voice.config import SttConfig
from spire_voice.providers.base import FinalTranscript, PartialTranscript, SttError
from spire_voice.providers.boot import ProviderUnavailable
from spire_voice.providers.stt_faster_whisper import FasterWhisperStt
from spire_voice.transports.base import SourceFormat


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    """A `WhisperModel` double: records the audio and language it was
    called with, and either yields the configured segments or raises."""

    def __init__(self, segment_texts: tuple[str, ...] = (), error: Exception | None = None) -> None:
        self.segment_texts = segment_texts
        self.error = error
        self.received_audio: "np.ndarray | None" = None
        self.received_language: "str | None" = None
        self.transcribe_calls = 0

    def transcribe(self, audio, language=None, **kwargs):
        self.transcribe_calls += 1
        self.received_audio = audio
        self.received_language = language
        if self.error is not None:
            raise self.error
        return (iter(_FakeSegment(t) for t in self.segment_texts), None)


def _stt_config(tmp_path, **overrides) -> SttConfig:
    if "local_model_dir" not in overrides:
        model_dir = tmp_path / "faster-whisper"
        model_dir.mkdir()
        overrides["local_model_dir"] = str(model_dir)
    return SttConfig(**overrides)


async def _frames(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


def test_missing_model_directory_raises_provider_unavailable_naming_path_and_step(tmp_path):
    missing = str(tmp_path / "does-not-exist")

    with pytest.raises(ProviderUnavailable) as excinfo:
        FasterWhisperStt(
            SttConfig(local_model_dir=missing),
            load_model=lambda config: _FakeWhisperModel(),
        )

    message = str(excinfo.value)
    assert missing in message
    assert "provisioning" in message.lower()


@pytest.mark.asyncio
async def test_model_loads_once_at_construction_not_per_turn(tmp_path):
    fake_model = _FakeWhisperModel(segment_texts=("hello",))
    load_calls: list[SttConfig] = []

    def _load(config: SttConfig):
        load_calls.append(config)
        return fake_model

    stt = FasterWhisperStt(_stt_config(tmp_path), load_model=_load)
    assert len(load_calls) == 1

    async for _ in stt.stream(_frames(b"\x00\x00" * 100), SourceFormat("pcm", 16000)):
        pass
    async for _ in stt.stream(_frames(b"\x00\x00" * 100), SourceFormat("pcm", 16000)):
        pass

    assert len(load_calls) == 1


@pytest.mark.asyncio
async def test_streaming_8khz_alaw_frames_decodes_and_resamples_before_transcribing(tmp_path):
    fake_model = _FakeWhisperModel(segment_texts=("hola",))
    stt = FasterWhisperStt(_stt_config(tmp_path), load_model=lambda config: fake_model)

    pcm16 = (np.arange(800, dtype=np.int16) - 400).tobytes()  # 100ms @ 8kHz
    alaw_bytes = pcm16_to_alaw(pcm16)

    events = [event async for event in stt.stream(_frames(alaw_bytes), SourceFormat("alaw", 8000))]

    assert events == [FinalTranscript(text="hola")]
    # Decode-and-resample ran: 800 samples @ 8kHz resampled to 16kHz is
    # exactly 1600 samples -- proof the A-law-then-resample path executed,
    # not merely that some audio arrived.
    assert fake_model.received_audio is not None
    assert len(fake_model.received_audio) == 1600


@pytest.mark.asyncio
async def test_streaming_16khz_pcm16_frames_reaches_the_model_with_no_decode_step(tmp_path):
    fake_model = _FakeWhisperModel(segment_texts=("hello there",))
    stt = FasterWhisperStt(_stt_config(tmp_path), load_model=lambda config: fake_model)

    pcm16 = (np.arange(1600, dtype=np.int16) - 800).tobytes()

    events = [event async for event in stt.stream(_frames(pcm16), SourceFormat("pcm", 16000))]

    assert events == [FinalTranscript(text="hello there")]
    expected = np.frombuffer(pcm16, dtype="<i2").astype(np.float32) / 32768.0
    assert np.array_equal(fake_model.received_audio, expected)


@pytest.mark.asyncio
async def test_every_segment_before_the_last_is_partial_and_the_last_is_final(tmp_path):
    fake_model = _FakeWhisperModel(segment_texts=("turn the", "lights on"))
    stt = FasterWhisperStt(_stt_config(tmp_path), load_model=lambda config: fake_model)

    events = [
        event async for event in stt.stream(_frames(b"\x00\x00" * 10), SourceFormat("pcm", 16000))
    ]

    assert events == [PartialTranscript(text="turn the"), FinalTranscript(text="lights on")]


@pytest.mark.asyncio
async def test_empty_utterance_yields_one_final_transcript_with_empty_text(tmp_path):
    fake_model = _FakeWhisperModel(segment_texts=())
    stt = FasterWhisperStt(_stt_config(tmp_path), load_model=lambda config: fake_model)

    events = [event async for event in stt.stream(_frames(b""), SourceFormat("pcm", 16000))]

    assert events == [FinalTranscript(text="")]


@pytest.mark.asyncio
async def test_a_raising_model_produces_stt_error_not_silence(tmp_path):
    fake_model = _FakeWhisperModel(error=RuntimeError("model exploded"))
    stt = FasterWhisperStt(_stt_config(tmp_path), load_model=lambda config: fake_model)

    with pytest.raises(SttError):
        async for _ in stt.stream(_frames(b"\x00\x00" * 10), SourceFormat("pcm", 16000)):
            pass
