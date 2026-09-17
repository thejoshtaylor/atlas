"""Proves the safety boundary's refusal reaches the spoken reply verbatim.

Turned green by plan 01-03. The one test here exercises the end-to-end shape
`mcp/spire_mcp/ha.py`'s own docstring documents: an uncaught `Denied`
becomes an error-shaped MCP result whose text is `Denied.reason`, and the
turn controller speaks that text directly with no second brain call in
between.
"""

from types import SimpleNamespace

from spire_mcp.ha import handle_call_service
from spire_mcp.safety import Denied, Policy

from spire_voice.providers.base import BrainReply, FinalTranscript, ToolCall


class _RefusingToolHost:
    """Calls straight into `handle_call_service` against `fake_ha`.

    Converts an uncaught `Denied` into the same error-shaped result the real
    MCP framework produces for an uncaught tool exception -- `is_error=True`,
    `content[0].text == str(exc)`, which for a `Denied` is exactly `reason`
    (see `ha.py`'s module docstring). No subprocess and no MCP wire protocol,
    matching `_FakeToolHost` in `tests/test_turn_controller.py`.
    """

    def __init__(self, ha, policy: Policy) -> None:
        self._ha = ha
        self._policy = policy

    async def call_tool(self, name: str, arguments: dict):
        assert name == "ha_call_service", f"unexpected tool: {name}"
        try:
            await handle_call_service(
                self._policy,
                self._ha.client,
                "http://ha.invalid",
                "test-token",
                **arguments,
            )
        except Denied as exc:
            return SimpleNamespace(is_error=True, content=[SimpleNamespace(text=str(exc))])
        raise AssertionError("call was allowed; this test needs a denied entity")


async def test_denied_reason_reaches_the_reply_verbatim(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_ha
):
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the server socket")])
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={
                            "domain": "switch",
                            "service": "turn_off",
                            "entity_id": "switch.example_server_socket",
                        },
                    )
                ]
            ),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    policy = Policy.from_config({"deny_entities": ["switch.example_server_socket"]})
    tool_host = _RefusingToolHost(fake_ha, policy)
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

    # The reason string safety.py actually raises, not a substring or a
    # paraphrase.
    assert tts.received_text == ["that one is off limits"]
    # The refusal does not make a second round trip to the language model.
    assert brain.call_count == 1
    # Nothing left the process for the denied call.
    assert len(fake_ha.requests) == 0
