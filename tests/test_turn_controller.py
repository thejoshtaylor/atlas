"""Real assertions for the turn-controller validation map.

`test_full_turn_happy_path` is turned green by plan 01-02. The remaining
four (the closing-turn and tool-round-cap edges) stay stubs for plan 01-05.
"""

import json
from types import SimpleNamespace

from spire_mcp.ha import handle_call_service, handle_list_entities
from spire_mcp.safety import Denied, Policy

from spire_voice.providers.base import BrainReply, FinalTranscript, ToolCall


class _FakeToolHost:
    """Calls straight into `spire_mcp.ha`'s handler functions against `fake_ha`.

    No subprocess and no MCP wire protocol -- this is a functional double
    for `McpToolHost`, exercising the same safety-gated handlers the real
    child process runs, over the fake HTTP transport instead of a real
    network call. A `Denied` raised by a handler is converted to an
    error-shaped result the same way the real MCP framework converts an
    uncaught tool exception, so the turn controller sees one consistent
    shape either way.
    """

    def __init__(self, ha, policy: Policy) -> None:
        self._ha = ha
        self._policy = policy

    async def call_tool(self, name: str, arguments: dict):
        try:
            if name == "ha_call_service":
                result = await handle_call_service(
                    self._policy, self._ha.client, "http://ha.invalid", "test-token", **arguments
                )
            elif name == "ha_list_entities":
                result = await handle_list_entities(
                    self._policy, self._ha.client, "http://ha.invalid", "test-token"
                )
            else:
                raise AssertionError(f"unknown tool: {name}")
        except Denied as exc:
            return SimpleNamespace(isError=True, content=[SimpleNamespace(text=str(exc))])
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps(result))])


async def test_full_turn_happy_path(fake_audio_source, fake_stt, fake_brain, fake_tts, fake_ha):
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={
                            "domain": "switch",
                            "service": "turn_on",
                            "entity_id": "switch.example_fan",
                        },
                    )
                ]
            ),
            BrainReply(text="turned on the fan"),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02", b"\x03\x04"])
    policy = Policy.from_config(None)
    tool_host = _FakeToolHost(fake_ha, policy)
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert brain.call_count == 2
    assert len(fake_ha.requests) == 1
    assert fake_ha.requests[0].method == "POST"
    assert tts.received_text == ["turned on the fan"]
    assert source.sent_audio == [b"\x01\x02", b"\x03\x04"]
    assert timings.end_of_speech_to_first_audio_ms is not None


def test_empty_transcript_closes_turn():
    raise AssertionError("not implemented: VOICE-08")


def test_silence_timeout_closes_turn():
    raise AssertionError("not implemented: VOICE-08")


def test_next_turn_runs_after_a_closed_turn():
    raise AssertionError("not implemented: VOICE-08")


def test_tool_round_cap_is_enforced():
    raise AssertionError("not implemented: VOICE-01")
