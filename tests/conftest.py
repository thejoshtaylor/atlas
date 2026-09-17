"""Shared test fixtures: fake providers, a fake audio source, a fake Home Assistant.

Every turn-pipeline test in this phase runs against these fakes instead of a
real network call. `fake_ha` is the one Home Assistant this project's tests
ever talk to; every entity id in it is invented, following the rule
`safety.py`'s own self-check states: no real house appears in this file.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any, AsyncIterator, Sequence

import httpx
import pytest
import pytest_asyncio


@dataclass
class PartialTranscript:
    """A speech-to-text event carrying an in-progress transcript."""

    text: str


@dataclass
class FinalTranscript:
    """A speech-to-text event carrying the finished transcript.

    `text == ""` is the empty-transcript case (VOICE-08): the operator
    toggled the microphone and said nothing intelligible.
    """

    text: str


@dataclass
class ToolCall:
    """One tool call the language model asked for."""

    name: str
    arguments: dict


@dataclass
class BrainReply:
    """One `chat()` result: zero or more tool calls, and/or reply text."""

    tool_calls: list[ToolCall] = field(default_factory=list)
    text: str = ""


class FakeStt:
    """Replays a scripted sequence of transcript events, then stops -- or hangs.

    Three constructible modes, all through the same `events`/`hang` pair:
    scripted events ending in a real `FinalTranscript` (the happy path),
    scripted events ending in an empty `FinalTranscript` (VOICE-08's first
    case -- something arrived, but it decoded to nothing), and `hang=True`
    (VOICE-08's second case -- the provider never sends a
    `transcript.partial`/`transcript.done` event at all). `events=()` alone
    still exhausts immediately rather than hanging; only `hang=True` actually
    suspends forever, which is what makes it a faithful stand-in for a real
    socket that a client-side timeout -- not a server event -- must close.
    """

    def __init__(self, events: Sequence[object] = (), hang: bool = False) -> None:
        self._events = list(events)
        self._hang = hang

    async def stream(self, frames) -> AsyncIterator[object]:
        # `frames` is accepted and ignored: this fake replays its scripted
        # events regardless of what audio it was handed.
        for event in self._events:
            yield event
        if self._hang:
            # Never resolves on its own -- only cancellation (the
            # controller's timeout guard) or garbage collection ends this.
            await asyncio.Event().wait()


@pytest.fixture
def fake_stt():
    """Factory: `fake_stt(events=[...])` builds a scripted `FakeStt`."""
    return FakeStt


class FakeBrain:
    """A streaming, tool-calling language model fake.

    Each call to `chat()` consumes and returns the next scripted
    `BrainReply`. Calling `chat()` more times than replies were scripted
    raises, so a test proving the `max_tool_rounds` cap can construct a
    reply list shorter than the rounds it expects the pipeline to attempt.
    """

    def __init__(self, replies: Sequence[BrainReply] = ()) -> None:
        self._replies = list(replies)
        self.call_count = 0

    async def chat(self, messages, tools=None) -> BrainReply:
        if self.call_count >= len(self._replies):
            raise AssertionError("FakeBrain.chat called more times than scripted")
        reply = self._replies[self.call_count]
        self.call_count += 1
        return reply


@pytest.fixture
def fake_brain():
    """Factory: `fake_brain(replies=[...])` builds a scripted `FakeBrain`."""
    return FakeBrain


class FakeTts:
    """Yields a scripted list of audio chunks; records the text it received."""

    def __init__(self, chunks: Sequence[bytes] = ()) -> None:
        self._chunks = list(chunks)
        self.received_text: list[str] = []

    async def synthesize(self, text_deltas) -> AsyncIterator[bytes]:
        async for delta in text_deltas:
            self.received_text.append(delta)
        for chunk in self._chunks:
            yield chunk


@pytest.fixture
def fake_tts():
    """Factory: `fake_tts(chunks=[...])` builds a scripted `FakeTts`."""
    return FakeTts


class FakeAudioSource:
    """Satisfies the transport protocol: yields a fixed list of PCM16 frames.

    Transport-independent, per D-02 -- a test drives the turn pipeline
    through this fake without caring whether WebSocket or WebRTC is live.
    """

    def __init__(self, frames: Sequence[bytes] = ()) -> None:
        self._frames = list(frames)
        self.sent_audio: list[bytes] = []

    async def frames(self) -> AsyncIterator[bytes]:
        for frame in self._frames:
            yield frame

    async def send_audio(self, chunk: bytes) -> None:
        self.sent_audio.append(chunk)


@pytest.fixture
def fake_audio_source():
    """Factory: `fake_audio_source(frames=[...])` builds a `FakeAudioSource`."""
    return FakeAudioSource


class FakeEnvelopeClient:
    """A fake `instructor`-wrapped client for a `brain_race.TierBrain`.

    Only `.chat.completions.create(...)` exists -- the one method
    `run_triage_tier`/`run_top_tier` actually call. Returns a scripted
    `TierReply`, optionally after a real `asyncio.sleep`, so a test can make
    one tier "return later" than another without a real network call.
    Records every call's keyword arguments, so a test can assert a `tools`
    keyword never reached a triage tier's client.
    """

    def __init__(self, reply: object, delay_s: float = 0.0) -> None:
        self._reply = reply
        self._delay_s = delay_s
        self.calls: list[dict[str, Any]] = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    async def _create(self, **kwargs: Any) -> object:
        self.calls.append(kwargs)
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._reply


@pytest.fixture
def fake_envelope_client():
    """Factory: `fake_envelope_client(reply=..., delay_s=...)` builds one."""
    return FakeEnvelopeClient


# Every entity id below is invented. No real house appears in this file,
# matching the convention `mcp/spire_mcp/safety.py::_demo` already states.
_FAKE_STATES: dict[str, dict] = {
    "switch.example_fan": {
        "entity_id": "switch.example_fan",
        "state": "off",
        "attributes": {},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
    "switch.example_server_socket": {
        "entity_id": "switch.example_server_socket",
        "state": "on",
        "attributes": {},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
    "sensor.example_server_power": {
        "entity_id": "sensor.example_server_power",
        "state": "42.0",
        "attributes": {"unit_of_measurement": "W"},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
    "light.example_lamp": {
        "entity_id": "light.example_lamp",
        "state": "off",
        "attributes": {},
        "last_changed": "2026-01-01T00:00:00+00:00",
        "last_updated": "2026-01-01T00:00:00+00:00",
    },
}


class FakeHomeAssistant:
    """An `httpx.MockTransport` handler plus the `AsyncClient` built on it.

    Serves `GET /api/states`, `GET /api/states/{entity_id}`, and
    `POST /api/services/{domain}/{service}` with the response shapes
    RESEARCH.md section 5 documents. `requests` records every request this
    fake received, so a test can assert its length is 0 for a denied call.
    """

    def __init__(self) -> None:
        self.requests: list[httpx.Request] = []
        self.client = httpx.AsyncClient(
            base_url="http://ha.invalid",
            transport=httpx.MockTransport(self._handle),
        )

    def _handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        path = request.url.path

        if request.method == "GET" and path == "/api/states":
            return httpx.Response(200, json=list(_FAKE_STATES.values()))

        if request.method == "GET" and path.startswith("/api/states/"):
            entity_id = path.removeprefix("/api/states/")
            state = _FAKE_STATES.get(entity_id)
            if state is None:
                return httpx.Response(404, json={"message": "Entity not found"})
            return httpx.Response(200, json=state)

        if request.method == "POST" and path.startswith("/api/services/"):
            _, _, _, domain, service = path.split("/", 4)
            body = request.read()
            payload = json.loads(body) if body else {}
            entity_id = payload.get("entity_id")
            state = _FAKE_STATES.get(entity_id) if entity_id else None
            if state is None:
                return httpx.Response(200, json=[])
            return httpx.Response(200, json=[state])

        return httpx.Response(404, json={"message": "not found"})

    async def aclose(self) -> None:
        await self.client.aclose()


@pytest_asyncio.fixture
async def fake_ha():
    ha = FakeHomeAssistant()
    yield ha
    await ha.aclose()
