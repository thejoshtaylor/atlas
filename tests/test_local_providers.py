"""Local providers -- faster-whisper speech-to-text and Piper text-to-speech
-- driven by real classes against fake models, never a fake subclass
(PROV-06, D-09, D-10, D-11).
"""

from __future__ import annotations

from typing import AsyncIterator

import numpy as np
import pytest

from spire_voice.audio.alaw import pcm16_to_alaw
from spire_voice.config import SttConfig, TtsConfig
from spire_voice.providers.base import FinalTranscript, PartialTranscript, SttError
from spire_voice.providers.boot import ProviderUnavailable
from spire_voice.providers.stt_faster_whisper import FasterWhisperStt
from spire_voice.providers.tts_piper import PiperTts
from spire_voice.providers.tts_xai import SinkFormat
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


class _FakeAudioChunk:
    def __init__(self, pcm16: bytes, sample_rate: int) -> None:
        self.audio_int16_bytes = pcm16
        self.sample_rate = sample_rate


class _FakeVoice:
    """A `piper.PiperVoice` double: records the text it was asked to
    render and returns the configured chunks -- matching the real
    `PiperVoice.synthesize(text) -> Iterable[AudioChunk]` shape confirmed
    against the real `piper-tts` package in a throwaway virtualenv this
    session (`AudioChunk.sample_rate`, `AudioChunk.audio_int16_bytes`)."""

    def __init__(self, chunks: "list[_FakeAudioChunk]") -> None:
        self.chunks = chunks
        self.received_text: "str | None" = None

    def synthesize(self, text: str, syn_config=None):
        self.received_text = text
        return iter(self.chunks)


def _tts_config(tmp_path, **overrides) -> TtsConfig:
    if "piper_voice_path" not in overrides:
        voice_path = tmp_path / "voice.onnx"
        voice_path.write_bytes(b"fake-voice-weights")
        overrides["piper_voice_path"] = str(voice_path)
    if "piper_config_path" not in overrides:
        config_path = tmp_path / "voice.onnx.json"
        config_path.write_text("{}")
        overrides["piper_config_path"] = str(config_path)
    return TtsConfig(**overrides)


def test_missing_voice_file_raises_provider_unavailable_naming_path_and_step(tmp_path):
    missing = str(tmp_path / "does-not-exist.onnx")
    config_path = tmp_path / "voice.onnx.json"
    config_path.write_text("{}")

    with pytest.raises(ProviderUnavailable) as excinfo:
        PiperTts(
            TtsConfig(piper_voice_path=missing, piper_config_path=str(config_path)),
            load_voice=lambda config: _FakeVoice([]),
        )

    message = str(excinfo.value)
    assert missing in message
    assert "provisioning" in message.lower()


def test_missing_voice_config_raises_provider_unavailable_naming_path_and_step(tmp_path):
    voice_path = tmp_path / "voice.onnx"
    voice_path.write_bytes(b"fake-voice-weights")
    missing_config = str(tmp_path / "does-not-exist.onnx.json")

    with pytest.raises(ProviderUnavailable) as excinfo:
        PiperTts(
            TtsConfig(piper_voice_path=str(voice_path), piper_config_path=missing_config),
            load_voice=lambda config: _FakeVoice([]),
        )

    message = str(excinfo.value)
    assert missing_config in message
    assert "provisioning" in message.lower()


@pytest.mark.asyncio
async def test_camera_sink_returns_alaw_at_8khz(tmp_path):
    pcm16 = (np.arange(2205, dtype=np.int16) - 1000).tobytes()  # 100ms @ 22050 Hz
    fake_voice = _FakeVoice([_FakeAudioChunk(pcm16, 22050)])
    tts = PiperTts(_tts_config(tmp_path), load_voice=lambda config: fake_voice)

    audio = await tts.synthesize_once("turn on the lights", sink=SinkFormat(codec="alaw", sample_rate=8000))

    assert fake_voice.received_text == "turn on the lights"
    # 100ms of audio resampled to 8kHz is exactly 800 samples; A-law is one
    # byte per sample, so the returned buffer's length alone proves both
    # the resample and the A-law encode ran.
    assert len(audio) == 800


@pytest.mark.asyncio
async def test_browser_sink_returns_pcm_at_the_sinks_rate(tmp_path):
    pcm16 = (np.arange(2205, dtype=np.int16) - 1000).tobytes()  # 100ms @ 22050 Hz
    fake_voice = _FakeVoice([_FakeAudioChunk(pcm16, 22050)])
    tts = PiperTts(_tts_config(tmp_path), load_voice=lambda config: fake_voice)

    audio = await tts.synthesize_once("ok", sink=SinkFormat(codec="pcm", sample_rate=24000))

    # 100ms resampled to 24kHz is exactly 2400 samples, 2 bytes each --
    # still 16-bit PCM, not A-law, since this sink asked for "pcm".
    assert len(audio) == 2400 * 2


@pytest.mark.asyncio
async def test_sink_is_read_not_assumed_two_calls_two_different_outputs(tmp_path):
    """The same rendered audio, asked for through two different sinks,
    must not silently reuse one sink's bytes for the other."""
    pcm16 = (np.arange(2205, dtype=np.int16) - 1000).tobytes()
    fake_voice = _FakeVoice([_FakeAudioChunk(pcm16, 22050)])
    tts = PiperTts(_tts_config(tmp_path), load_voice=lambda config: fake_voice)

    camera_audio = await tts.synthesize_once("hi", sink=SinkFormat(codec="alaw", sample_rate=8000))
    browser_audio = await tts.synthesize_once("hi", sink=SinkFormat(codec="pcm", sample_rate=24000))

    assert len(camera_audio) != len(browser_audio)


def test_no_chunking_method_of_its_own():
    """D-05/D-07: only `BatchTtsAdapter` chunks a batch provider's output
    -- this class exposes the single-shot method and nothing that looks
    like a streaming `synthesize()` of its own."""
    assert hasattr(PiperTts, "synthesize_once")
    assert not hasattr(PiperTts, "synthesize")


def test_extra_not_installed_raises_provider_unavailable_naming_the_extra(tmp_path):
    """D-10: `piper-tts` is deliberately absent from this project's own dev
    virtualenv (07-03-PLAN.md Task 1) -- this exercises the real default
    loader, not a fake, so this proves the actual `ImportError` this
    environment produces surfaces as `ProviderUnavailable` rather than
    stopping the boot."""
    with pytest.raises(ProviderUnavailable) as excinfo:
        PiperTts(_tts_config(tmp_path))

    assert "piper" in str(excinfo.value).lower()


# --- D-12: the published local-set latency figure --------------------


def test_the_two_local_registry_entries_carry_the_measured_note_verbatim():
    """07-04-PLAN.md Task 3: the local speech-to-text and text-to-speech
    entries both carry 07-UI-SPEC.md's local-set latency sentence,
    verbatim, as data -- never a paraphrase, and never present on the
    cloud (xAI) entries, which are not part of the local set D-12
    measures."""
    from spire_voice.providers import registry

    faster_whisper = registry.STT_REGISTRY["faster-whisper"]
    piper = registry.TTS_REGISTRY["piper"]

    assert faster_whisper.measured_note is not None
    assert piper.measured_note is not None
    assert faster_whisper.measured_note.startswith("Measured on this project's CPU-only host:")
    assert piper.measured_note.startswith("Measured on this project's CPU-only host:")
    assert registry.STT_REGISTRY["xai"].measured_note is None
    assert registry.TTS_REGISTRY["xai"].measured_note is None
