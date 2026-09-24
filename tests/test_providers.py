"""Real assertions for the xAI provider wire-contract validation map.

Turned green by plan 01-02. No test here opens a network connection or
reads an environment variable -- URL construction and stream accumulation
are pure functions, tested directly against invented config values and
fake streamed chunks.
"""

import asyncio
import json
from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

import pytest

from atlas.config import SttConfig, TtsConfig
from atlas.providers.base import FinalTranscript, ToolCall
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


def test_stt_url_uses_wire_parameter_names():
    from atlas.providers.stt_xai import XaiStt

    stt = XaiStt(_stt_cfg())
    url = stt.build_url(SourceFormat("pcm", 16000))

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert query["endpointing"] == ["200"]
    assert query["smart_turn"] == ["0.7"]
    assert query["smart_turn_timeout"] == ["1200"]
    assert not any(name.endswith("_ms") for name in query)


def test_stt_url_repeats_keyterm_once_per_configured_term():
    from dataclasses import replace

    from atlas.providers.stt_xai import XaiStt

    stt = XaiStt(replace(_stt_cfg(), keyterms=("Atlas", "example cooler")))
    query = parse_qs(urlparse(stt.build_url(SourceFormat("alaw", 8000))).query)
    assert query["keyterm"] == ["Atlas", "example cooler"]


def test_stt_keyterms_parse_from_a_comma_separated_string_or_a_list():
    assert SttConfig.from_config({"keyterms": "Atlas, example cooler,"}).keyterms == (
        "Atlas",
        "example cooler",
    )
    assert SttConfig.from_config({"keyterms": ["Atlas"]}).keyterms == ("Atlas",)
    assert SttConfig.from_config({}).keyterms == ()


def test_stt_url_renders_the_16khz_pcm_source_it_is_given():
    """`build_url` reads `source_format` rather than assuming Phase 1's one
    source -- a 16 kHz PCM `AudioSource` renders exactly that pair."""
    from atlas.providers.stt_xai import XaiStt

    stt = XaiStt(_stt_cfg())
    query = parse_qs(urlparse(stt.build_url(SourceFormat("pcm", 16000))).query)

    assert query["encoding"] == ["pcm"]
    assert query["sample_rate"] == ["16000"]


def test_stt_url_renders_the_8khz_alaw_source_it_is_given():
    """PROV-07: an 8 kHz A-law `AudioSource` (the camera) must never be
    streamed into a socket opened for 16 kHz PCM -- `build_url` renders the
    A-law source's own sample rate, not a hardcoded 16000."""
    from atlas.providers.stt_xai import XaiStt

    stt = XaiStt(_stt_cfg())
    query = parse_qs(urlparse(stt.build_url(SourceFormat("alaw", 8000))).query)

    assert query["encoding"] == ["alaw"]
    assert query["sample_rate"] == ["8000"]


def test_tts_session_update_requests_browser_playable_codec():
    from atlas.providers.tts_xai import XaiTts

    cfg = TtsConfig(
        url="wss://api.x.ai/v1/tts",
        api_key="test-key",
        voice_id="eve",
        language="en",
        codec="alaw",
        sample_rate=8000,
        browser_codec="pcm",
        browser_sample_rate=24000,
    )
    tts = XaiTts(cfg)
    message = tts.build_session_update()

    assert message["output_format"] == {"codec": "pcm", "sample_rate": 24000}


class _NeverEndingFrames:
    """Mirrors a live mic: keeps streaming past the turn's own end.

    Real production audio sources only stop on the operator's own
    start/stop toggle (`index.html`'s "single on/off toggle only" design),
    never because a reply arrived -- so a faithful frames double for
    testing `XaiStt.stream()` must not stop on its own either.
    """

    async def __call__(self):
        while True:
            yield b"\x00\x00"
            await asyncio.sleep(0.01)


class _FakeXaiWebsocket:
    """A minimal double for `websockets.connect()`'s return value.

    Reproduces the real xAI STT wire shape closely enough to exercise
    `XaiStt.stream()`'s own control flow: an async context manager,
    `recv()` for the `transcript.created` handshake, `send()` for outbound
    audio, and `async for` for inbound JSON events ending in
    `transcript.done`.
    """

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


async def test_stream_ends_promptly_once_final_transcript_arrives_even_if_mic_keeps_streaming(monkeypatch):
    """CR-02 regression.

    `XaiStt.stream()`'s `finally` clause once awaited its own `sender()`
    task to completion, and `sender()` only finishes once `frames` (the mic
    source) is exhausted -- which a live mic never is mid-turn. That made
    `stream()` hang past `transcript.done` instead of reaching
    `StopAsyncIteration`, in turn making the controller's lookahead call in
    `_drain_to_final_transcript` block indefinitely. This reproduces that
    exact shape (a `frames` source that never ends on its own) and asserts
    `stream()` still finishes promptly -- it hangs past the timeout below
    against the pre-fix code, where `finally: await send_task` never
    returns while `frames` keeps yielding.
    """
    from atlas.providers.stt_xai import XaiStt

    fake_ws = _FakeXaiWebsocket([{"type": "transcript.done", "text": "turn on the fan"}])
    monkeypatch.setattr(
        "atlas.providers.stt_xai.websockets.connect",
        lambda *args, **kwargs: fake_ws,
    )

    stt = XaiStt(_stt_cfg())

    async def drain():
        events = []
        async for event in stt.stream(_NeverEndingFrames()(), SourceFormat("pcm", 16000)):
            events.append(event)
        return events

    events = await asyncio.wait_for(drain(), timeout=1.0)

    assert events == [FinalTranscript(text="turn on the fan")]


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tc(index, name=None, arguments=None):
    return SimpleNamespace(index=index, function=SimpleNamespace(name=name, arguments=arguments))


async def test_brain_accumulates_tool_calls_by_index():
    from atlas.providers.brain_xai import accumulate_stream

    async def whole():
        yield _chunk(
            tool_calls=[
                _tc(
                    0,
                    name="ha_call_service",
                    arguments='{"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"}',
                )
            ]
        )

    async def fragmented():
        yield _chunk(tool_calls=[_tc(0, name="ha_call")])
        yield _chunk(tool_calls=[_tc(0, name="_service", arguments='{"domain": "switch",')])
        yield _chunk(tool_calls=[_tc(0, arguments=' "service": "turn_on", "entity_id": "switch.example_fan"}')])

    whole_reply = await accumulate_stream(whole())
    fragmented_reply = await accumulate_stream(fragmented())

    expected = [
        ToolCall(
            name="ha_call_service",
            arguments={"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"},
        )
    ]
    assert whole_reply.tool_calls == expected
    assert fragmented_reply.tool_calls == expected


@pytest.mark.asyncio
async def test_final_transcript_comes_from_speech_final_not_transcript_done():
    """The live API puts the final text on a partial, and `done` is empty.

    VERIFIED AGAINST THE LIVE xAI API, 2026-09-17. One spoken sentence produced:

        transcript.partial  is_final=False speech_final=False  "Is the living room light?"
        transcript.partial  is_final=True  speech_final=False  "Is the living room light sensor on?"
        transcript.partial  is_final=True  speech_final=True   "Is the living room light sensor on?"
        transcript.done                                        ""

    The provider previously read the final text off `transcript.done`, whose
    `text` is always the empty string -- so every real turn threw away a
    perfect transcription and the assistant said "sorry, i didn't catch that".

    The whole test suite stayed green through that, because `FakeStt` emitted
    `transcript.done` WITH text: the fake encoded the same assumption the
    provider did, so the two agreed with each other and neither agreed with
    the API. This test replays the real frame sequence instead.
    """
    import json as _json

    from atlas.providers.base import FinalTranscript, PartialTranscript
    from atlas.providers.stt_xai import XaiStt
    from atlas.config import SttConfig

    wire = [
        {"type": "transcript.partial", "is_final": False, "speech_final": False, "text": "Is the living"},
        {"type": "transcript.partial", "is_final": True, "speech_final": False, "text": "Is the living room light sensor on?"},
        {"type": "transcript.partial", "is_final": True, "speech_final": True, "text": "Is the living room light sensor on?"},
        {"type": "transcript.done", "text": "", "duration": 1.845},
    ]

    class _Ws:
        def __init__(self, frames): self._frames = list(frames); self._sent = []
        async def __aenter__(self): return self
        async def __aexit__(self, *a): return False
        async def recv(self): return _json.dumps({"type": "transcript.created", "id": "x"})
        async def send(self, d): self._sent.append(d)
        def __aiter__(self):
            async def gen():
                for e in self._frames:
                    yield _json.dumps(e)
            return gen()

    import atlas.providers.stt_xai as mod

    def _connect(*a, **k): return _Ws(wire)

    orig = mod.websockets.connect
    mod.websockets.connect = _connect
    try:
        async def frames():
            yield b"\x00\x00" * 160

        stt = XaiStt(SttConfig.from_config({"url": "wss://x.invalid", "api_key": "k"}))
        out = [ev async for ev in stt.stream(frames(), SourceFormat("pcm", 16000))]
    finally:
        mod.websockets.connect = orig

    finals = [e for e in out if isinstance(e, FinalTranscript)]
    assert len(finals) == 1, f"expected exactly one final, got {out!r}"
    assert finals[0].text == "Is the living room light sensor on?", (
        "the final transcript must come from the speech_final partial, not from "
        "transcript.done (whose text is always empty)"
    )
    assert any(isinstance(e, PartialTranscript) for e in out)


@pytest.mark.asyncio
async def test_tts_synthesize_once_posts_rest_and_returns_the_whole_buffer():
    """xAI TTS is REST. VERIFIED AGAINST THE LIVE API, 2026-09-17.

        POST https://api.x.ai/v1/tts
        {"text","voice_id","language","output_format":{"codec","sample_rate"}}
        -> 200, Content-Type: audio/pcm, the whole utterance as raw bytes

    This provider used to open a WebSocket to that same `https://` URL and died
    on `InvalidURI: scheme isn't ws or wss` the first time a real turn reached
    speech -- after a correct transcription and a correct Home Assistant read,
    so everything upstream was working and the operator still heard nothing.

    `wss://api.x.ai/v1/tts` does exist but rejected every parameter shape tried
    against it with HTTP 400. Omitting `language` returns a 422 naming the
    missing field, which is how the required shape above was established.

    Plan 07-02: `synthesize_once` takes a plain string and returns a plain
    buffer -- joining deltas and chunking the response both moved to
    `BatchTtsAdapter` (`tests/test_batch_tts_adapter.py` covers both).
    """
    import httpx as _httpx

    from atlas.config import TtsConfig
    from atlas.providers.tts_xai import XaiTts

    captured = {}

    def handler(request: _httpx.Request) -> _httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return _httpx.Response(200, content=b"\x01\x02" * 1000,
                               headers={"content-type": "audio/pcm"})

    transport = _httpx.MockTransport(handler)
    real_client = _httpx.AsyncClient

    def _client(*a, **k):
        k.pop("timeout", None)
        return real_client(transport=transport)

    import atlas.providers.tts_xai as mod
    mod.httpx.AsyncClient = _client
    try:
        tts = XaiTts(TtsConfig.from_config(
            {"url": "https://api.x.ai/v1/tts", "api_key": "k", "voice_id": "eve"}
        ))

        audio = await tts.synthesize_once("the light is off")
    finally:
        mod.httpx.AsyncClient = real_client

    assert captured["url"].startswith("https://"), "TTS is REST, not a websocket"
    assert captured["body"]["text"] == "the light is off"
    assert captured["body"]["language"], "language is required; omitting it is a 422"
    assert set(captured["body"]["output_format"]) == {"codec", "sample_rate"}
    assert audio == b"\x01\x02" * 1000, "the whole response body must reach the caller"


@pytest.mark.asyncio
async def test_xai_tts_reuses_one_http_client_across_calls():
    """260924-4iu (a): one `XaiTts` instance builds its client once, on the
    first call, and reuses it -- not a fresh client, and a fresh TCP and
    TLS handshake, per call."""
    import httpx as _httpx

    from atlas.config import TtsConfig
    from atlas.providers.tts_xai import XaiTts

    request_count = 0

    def handler(request: _httpx.Request) -> _httpx.Response:
        nonlocal request_count
        request_count += 1
        return _httpx.Response(200, content=b"\x01\x02", headers={"content-type": "audio/pcm"})

    transport = _httpx.MockTransport(handler)
    real_client = _httpx.AsyncClient
    build_count = 0

    def _counting_client(*a, **k):
        nonlocal build_count
        build_count += 1
        k.pop("timeout", None)
        k.pop("limits", None)
        return real_client(transport=transport)

    import atlas.providers.tts_xai as mod

    mod.httpx.AsyncClient = _counting_client
    try:
        tts = XaiTts(TtsConfig.from_config({"url": "https://api.x.ai/v1/tts", "api_key": "k", "voice_id": "eve"}))
        await tts.synthesize_once("one")
        await tts.synthesize_once("two")
    finally:
        mod.httpx.AsyncClient = real_client

    assert build_count == 1, "the client must be built once, not once per call"
    assert request_count == 2, "both calls must reach the transport through the one client"


@pytest.mark.asyncio
async def test_xai_tts_aclose_closes_the_owned_client_and_is_idempotent():
    """260924-4iu (a): `aclose()` closes the client the instance built,
    does nothing on a second call, does nothing when no call was ever
    made, and never closes a client the caller injected."""
    import httpx as _httpx

    from atlas.config import TtsConfig
    from atlas.providers.tts_xai import XaiTts

    def handler(request: _httpx.Request) -> _httpx.Response:
        return _httpx.Response(200, content=b"\x01\x02", headers={"content-type": "audio/pcm"})

    transport = _httpx.MockTransport(handler)
    real_client = _httpx.AsyncClient

    def _client(*a, **k):
        k.pop("timeout", None)
        k.pop("limits", None)
        return real_client(transport=transport)

    import atlas.providers.tts_xai as mod

    mod.httpx.AsyncClient = _client
    try:
        config = TtsConfig.from_config({"url": "https://api.x.ai/v1/tts", "api_key": "k", "voice_id": "eve"})

        tts = XaiTts(config)
        await tts.synthesize_once("hi")
        built_client = tts._client
        await tts.aclose()
        assert built_client.is_closed, "aclose() must close the client the instance built"
        await tts.aclose()  # idempotent: must not raise

        never_called = XaiTts(config)
        await never_called.aclose()  # must not raise, and must build no client
        assert never_called._client is None
    finally:
        mod.httpx.AsyncClient = real_client

    injected = real_client(transport=transport)
    try:
        config = TtsConfig.from_config({"url": "https://api.x.ai/v1/tts", "api_key": "k", "voice_id": "eve"})
        injected_tts = XaiTts(config, http_client=injected)
        await injected_tts.synthesize_once("hi")
        await injected_tts.aclose()
        assert not injected.is_closed, "aclose() must never close a client the caller injected"
    finally:
        await injected.aclose()


@pytest.mark.asyncio
async def test_xai_tts_retries_once_on_a_dropped_keepalive_connection():
    """260924-4iu (a): a pooled keep-alive connection the server closed at
    the moment of reuse raises `httpx.RemoteProtocolError`. A render has
    no side effect, so `synthesize_once` retries the POST exactly once and
    returns the retry's audio. A handler that raises on every request
    still makes `synthesize_once` raise, after exactly two attempts."""
    import httpx as _httpx

    from atlas.config import TtsConfig
    from atlas.providers.tts_xai import XaiTts

    real_client = _httpx.AsyncClient

    def _client_over(transport: _httpx.MockTransport):
        def _build(*a, **k):
            k.pop("timeout", None)
            k.pop("limits", None)
            return real_client(transport=transport)

        return _build

    import atlas.providers.tts_xai as mod

    call_count = 0

    def flaky_handler(request: _httpx.Request) -> _httpx.Response:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise _httpx.RemoteProtocolError("connection dropped")
        return _httpx.Response(200, content=b"\x01\x02", headers={"content-type": "audio/pcm"})

    mod.httpx.AsyncClient = _client_over(_httpx.MockTransport(flaky_handler))
    try:
        tts = XaiTts(TtsConfig.from_config({"url": "https://api.x.ai/v1/tts", "api_key": "k", "voice_id": "eve"}))
        audio = await tts.synthesize_once("hi")
    finally:
        mod.httpx.AsyncClient = real_client

    assert audio == b"\x01\x02"
    assert call_count == 2, "the second attempt is the one whose audio reaches the caller"

    always_fail_count = 0

    def always_fails(request: _httpx.Request) -> _httpx.Response:
        nonlocal always_fail_count
        always_fail_count += 1
        raise _httpx.RemoteProtocolError("connection dropped")

    mod.httpx.AsyncClient = _client_over(_httpx.MockTransport(always_fails))
    try:
        tts2 = XaiTts(TtsConfig.from_config({"url": "https://api.x.ai/v1/tts", "api_key": "k", "voice_id": "eve"}))
        with pytest.raises(_httpx.RemoteProtocolError):
            await tts2.synthesize_once("hi")
    finally:
        mod.httpx.AsyncClient = real_client

    assert always_fail_count == 2, "exactly two attempts: the retry is not retried again"


@pytest.mark.asyncio
async def test_batch_tts_adapter_skips_the_provider_call_entirely_for_empty_text():
    """An empty reply must not bill a synthesis request or emit silence.

    Plan 07-02: this short circuit lives on `BatchTtsAdapter` now, not on
    `XaiTts` -- driven here over the real client so the client's own
    request-building code stays proven to never fire for empty text, not
    just the adapter's fake-provider path (`test_batch_tts_adapter.py`
    covers that same behavior against a fake for the general case)."""
    from atlas.config import TtsConfig
    from atlas.providers.batch_tts_adapter import BatchTtsAdapter
    from atlas.providers.tts_xai import XaiTts

    calls = []
    tts = XaiTts(TtsConfig.from_config({"url": "https://x.invalid", "api_key": "k"}))
    real_synthesize_once = tts.synthesize_once

    async def _counting_synthesize_once(text, sink=None):
        calls.append(text)
        return await real_synthesize_once(text, sink=sink)

    tts.synthesize_once = _counting_synthesize_once
    adapter = BatchTtsAdapter(tts)

    async def nothing():
        yield "   "

    assert [c async for c in adapter.synthesize(nothing())] == []
    assert calls == [], "the provider must not be called at all for whitespace-only text"
