"""Real assertions for the xAI provider wire-contract validation map.

Turned green by plan 01-02. No test here opens a network connection or
reads an environment variable -- URL construction and stream accumulation
are pure functions, tested directly against invented config values and
fake streamed chunks.
"""

from types import SimpleNamespace
from urllib.parse import parse_qs, urlparse

from spire_voice.config import SttConfig, TtsConfig
from spire_voice.providers.base import ToolCall


def test_stt_url_uses_wire_parameter_names():
    from spire_voice.providers.stt_xai import XaiStt

    cfg = SttConfig(
        url="wss://api.x.ai/v1/stt",
        api_key="test-key",
        endpointing_ms=200,
        smart_turn=0.7,
        smart_turn_timeout_ms=1200,
        vad_threshold=0.08,
        interim_results=True,
        language="en",
    )
    stt = XaiStt(cfg)
    url = stt.build_url()

    parsed = urlparse(url)
    query = parse_qs(parsed.query)

    assert query["endpointing"] == ["200"]
    assert query["smart_turn"] == ["0.7"]
    assert query["smart_turn_timeout"] == ["1200"]
    assert not any(name.endswith("_ms") for name in query)


def test_tts_session_update_requests_browser_playable_codec():
    from spire_voice.providers.tts_xai import XaiTts

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


def _chunk(content=None, tool_calls=None):
    return SimpleNamespace(choices=[SimpleNamespace(delta=SimpleNamespace(content=content, tool_calls=tool_calls))])


def _tc(index, name=None, arguments=None):
    return SimpleNamespace(index=index, function=SimpleNamespace(name=name, arguments=arguments))


async def test_brain_accumulates_tool_calls_by_index():
    from spire_voice.providers.brain_xai import accumulate_stream

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
