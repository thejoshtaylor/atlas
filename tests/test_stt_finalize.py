"""D-12: a per-stream `finalize` hook on both speech-to-text providers.

Reuses the fake-socket shape `tests/test_providers.py` already established
for `XaiStt` (a `_NeverEndingFrames` double that never stops on its own,
matching a live mic) and `tests/test_local_providers.py`'s injected-model
shape for `FasterWhisperStt`.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import AsyncIterator

import numpy as np
import pytest

from atlas.config import SttConfig
from atlas.providers.base import FinalTranscript, SttError
from atlas.providers.registry import STT_REGISTRY
from atlas.providers.stt_faster_whisper import FasterWhisperStt
from atlas.providers.stt_xai import FINALIZE_MESSAGE, XaiStt
from atlas.transports.base import SourceFormat


def _stt_cfg() -> SttConfig:
    return SttConfig(
        url="wss://api.x.ai/v1/stt",
        api_key="test-key",
        endpointing_ms=200,
        smart_turn=0.7,
        smart_turn_timeout_ms=1200,
        vad_threshold=0.08,
        interim_results=True,
        language="en",
    )


def _stt_config(tmp_path, **overrides) -> SttConfig:
    if "local_model_dir" not in overrides:
        model_dir = tmp_path / "faster-whisper"
        model_dir.mkdir()
        overrides["local_model_dir"] = str(model_dir)
    return SttConfig(**overrides)


class _NeverEndingFrames:
    """Mirrors a live mic: keeps streaming past the turn's own end. See
    `tests/test_providers.py`'s own copy of this fixture for the full
    rationale."""

    async def __call__(self):
        while True:
            yield b"\x00\x00"
            await asyncio.sleep(0.01)


class _FakeXaiWebsocket:
    """A minimal double for `websockets.connect()`'s return value --
    the CR-02 shape from `tests/test_providers.py`, restated here so this
    file has no import dependency on that one."""

    def __init__(self, events):
        self._events = list(events)
        self.sent: list[bytes] = []

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def recv(self):
        return json.dumps({"type": "transcript.created"})

    async def send(self, data):
        self.sent.append(data)

    async def __aiter__(self):
        for event in self._events:
            yield json.dumps(event)


class _FinalizeAwareXaiWebsocket:
    """Like `_FakeXaiWebsocket`, but yields its one `speech_final` partial
    only once it has received `FINALIZE_MESSAGE` on `send()` -- proof that
    the provider's own finalizer sent it, not merely that some event
    eventually arrived."""

    def __init__(self, reply_text: str) -> None:
        self.sent: list[str] = []
        self._reply_text = reply_text
        self._finalize_received = asyncio.Event()

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def recv(self):
        return json.dumps({"type": "transcript.created"})

    async def send(self, data):
        self.sent.append(data)
        if data == json.dumps(FINALIZE_MESSAGE):
            self._finalize_received.set()

    async def __aiter__(self):
        await self._finalize_received.wait()
        yield json.dumps({"type": "transcript.partial", "speech_final": True, "text": self._reply_text})


async def test_xai_sends_finalize_once_when_the_event_is_set(monkeypatch):
    fake_ws = _FinalizeAwareXaiWebsocket("turn off the fan")
    monkeypatch.setattr("atlas.providers.stt_xai.websockets.connect", lambda *args, **kwargs: fake_ws)

    stt = XaiStt(_stt_cfg())
    finalize_event = asyncio.Event()

    async def drain():
        events = []
        async for event in stt.stream(
            _NeverEndingFrames()(), SourceFormat("pcm", 16000), finalize=finalize_event
        ):
            events.append(event)
        return events

    task = asyncio.ensure_future(drain())
    await asyncio.sleep(0.02)
    finalize_event.set()

    events = await asyncio.wait_for(task, timeout=1.0)

    assert events == [FinalTranscript(text="turn off the fan")]
    finalize_sends = [item for item in fake_ws.sent if item == json.dumps(FINALIZE_MESSAGE)]
    assert len(finalize_sends) == 1


async def test_xai_without_finalize_is_unchanged(monkeypatch):
    fake_ws = _FakeXaiWebsocket([{"type": "transcript.done", "text": "turn on the fan"}])
    monkeypatch.setattr("atlas.providers.stt_xai.websockets.connect", lambda *args, **kwargs: fake_ws)

    stt = XaiStt(_stt_cfg())

    async def drain():
        events = []
        async for event in stt.stream(_NeverEndingFrames()(), SourceFormat("pcm", 16000)):
            events.append(event)
        return events

    events = await asyncio.wait_for(drain(), timeout=1.0)

    assert events == [FinalTranscript(text="turn on the fan")]


class _FakeSegment:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeWhisperModel:
    def __init__(self, segment_texts: "tuple[str, ...]" = ()) -> None:
        self.segment_texts = segment_texts
        self.received_audio: "np.ndarray | None" = None

    def transcribe(self, audio, language=None, **kwargs):
        self.received_audio = audio
        return (iter(_FakeSegment(t) for t in self.segment_texts), None)


async def _queued_frames(queue: "asyncio.Queue[bytes]") -> AsyncIterator[bytes]:
    while True:
        yield await queue.get()


async def test_faster_whisper_finalize_transcribes_what_it_has(tmp_path):
    fake_model = _FakeWhisperModel(segment_texts=("stop",))
    stt = FasterWhisperStt(_stt_config(tmp_path), load_model=lambda config: fake_model)

    queue: "asyncio.Queue[bytes]" = asyncio.Queue()
    chunk = (np.arange(100, dtype=np.int16) - 50).tobytes()  # 100 samples @ 16kHz
    for _ in range(3):
        queue.put_nowait(chunk)

    finalize_event = asyncio.Event()

    async def drain():
        events = []
        async for event in stt.stream(
            _queued_frames(queue), SourceFormat("pcm", 16000), finalize=finalize_event
        ):
            events.append(event)
        return events

    task = asyncio.ensure_future(drain())
    # Let the reader task drain the three already-queued chunks and block
    # on a fourth (never provided) before finalizing.
    await asyncio.sleep(0.02)
    finalize_event.set()

    events = await asyncio.wait_for(task, timeout=1.0)

    assert events == [FinalTranscript(text="stop")]
    assert fake_model.received_audio is not None
    assert len(fake_model.received_audio) == 300  # exactly the 3 collected chunks


async def _frames(*chunks: bytes) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk


@pytest.mark.parametrize(
    "build_stt",
    [
        lambda tmp_path: XaiStt(_stt_cfg()),
        lambda tmp_path: FasterWhisperStt(
            _stt_config(tmp_path), load_model=lambda config: _FakeWhisperModel()
        ),
    ],
    ids=["xai", "faster-whisper"],
)
async def test_multichannel_source_raises_stterror_naming_channel_count(tmp_path, build_stt):
    stt = build_stt(tmp_path)
    two_channel = SourceFormat("pcm", 16000, channels=2, asr_channel=1)

    with pytest.raises(SttError) as excinfo:
        async for _ in stt.stream(_frames(b"\x00\x00"), two_channel):
            pass

    assert "2" in str(excinfo.value)


def test_every_registered_stt_provider_accepts_finalize(tmp_path):
    for name, entry in STT_REGISTRY.items():
        if name == "faster-whisper":
            # `entry.build` needs a real, existing `local_model_dir` --
            # built directly here rather than through the registry, the
            # same way `_stt_config(tmp_path)` already does for every other
            # test in this file.
            provider = FasterWhisperStt(
                _stt_config(tmp_path), load_model=lambda config: _FakeWhisperModel()
            )
        else:
            provider = entry.build(_stt_cfg(), "test-key", {})

        signature = inspect.signature(provider.stream)
        finalize_param = signature.parameters.get("finalize")
        assert finalize_param is not None, f"{name} provider's stream() has no finalize parameter"
        assert finalize_param.kind == inspect.Parameter.KEYWORD_ONLY, name
        assert finalize_param.default is None, name


def test_finalize_message_is_the_live_verified_value():
    """10-05-PLAN.md Task 3: `scripts/verify_xai_finalize.py` proved both
    spellings work against the live socket (2026-09-25); D-12's own
    lowercase spelling is what the constant keeps. Pinned exactly, so a
    later edit cannot drift away from the live-verified value silently."""
    assert FINALIZE_MESSAGE == {"type": "finalize"}
