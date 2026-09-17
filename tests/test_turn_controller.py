"""Real assertions for the turn-controller validation map.

`test_full_turn_happy_path` was turned green by plan 01-02. The remaining
four -- the two VOICE-08 closing-turn guards, the guarantee that a closed
turn does not wedge the next one, and the tool-round cap -- are turned green
by plan 01-05.
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


async def test_empty_transcript_closes_turn(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """A final transcript that decoded to nothing ends the turn immediately.

    No language-model call and no text-to-speech call -- there is nothing to
    reason about, per RESEARCH.md Pitfall 3's first case.
    """
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert brain.call_count == 0
    assert tts.received_text == []
    assert timings.turn_outcome == "empty_transcript"


async def test_silence_timeout_closes_turn(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """A provider that never emits anything is closed by the client-side
    `max_utterance_s` timeout, never by a provider event -- RESEARCH.md
    Pitfall 3's second case, the one with no dedicated server event to key
    off at all.

    The clock is a test-controlled callable, not real elapsed time: each
    call advances the fake clock by 5 simulated seconds, so the configured
    15-second budget is exceeded on the third check with no real sleep
    anywhere near that long.
    """
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(hang=True)
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[])
    timings = TurnTimings()

    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 5.0
        return fake_now[0]

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        max_utterance_s=15,
        clock=clock,
        poll_interval_s=0.01,
    )

    assert brain.call_count == 0
    assert tts.received_text == []
    assert timings.turn_outcome == "timeout"


async def test_next_turn_runs_after_a_closed_turn(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """The half of VOICE-08 a hang would hide: whichever route closed the
    first turn, an ordinary turn right after it still reaches speech.
    """
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    async def _assert_ordinary_turn_reaches_tts() -> None:
        source = fake_audio_source(frames=[b"\x00\x01"])
        stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
        brain = fake_brain(replies=[BrainReply(text="turned on the fan")])
        tts = fake_tts(chunks=[b"\x01\x02"])
        await run_turn(
            source,
            stt,
            brain,
            tts,
            None,
            tools_schema=[],
            system_prompt="you control a home",
            max_tool_rounds=3,
            timings=TurnTimings(),
        )
        assert tts.received_text == ["turned on the fan"]
        assert source.sent_audio == [b"\x01\x02"]

    # Route 1: closed by the empty-transcript guard.
    empty_source = fake_audio_source(frames=[b"\x00\x01"])
    empty_stt = fake_stt(events=[FinalTranscript(text="")])
    await run_turn(
        empty_source,
        empty_stt,
        fake_brain(replies=[]),
        fake_tts(chunks=[]),
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=TurnTimings(),
    )
    await _assert_ordinary_turn_reaches_tts()

    # Route 2: closed by the silence-timeout guard.
    hang_source = fake_audio_source(frames=[b"\x00\x01"])
    hang_stt = fake_stt(hang=True)
    fake_now = [0.0]

    def clock() -> float:
        fake_now[0] += 5.0
        return fake_now[0]

    await run_turn(
        hang_source,
        hang_stt,
        fake_brain(replies=[]),
        fake_tts(chunks=[]),
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=TurnTimings(),
        max_utterance_s=15,
        clock=clock,
        poll_interval_s=0.01,
    )
    await _assert_ordinary_turn_reaches_tts()


async def test_tool_round_cap_is_enforced(fake_audio_source, fake_stt, fake_brain, fake_tts):
    """A command that keeps asking for tools past `max_tool_rounds` stops and
    says so -- never a success confirmation, per CMD-01's transparency
    prohibition.
    """
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import _TOO_MANY_ROUNDS_REPLY, run_turn

    class _AlwaysToolHost:
        """A tool host that always answers successfully -- the brain is what
        keeps asking for another round, not a failing tool call."""

        async def call_tool(self, name: str, arguments: dict) -> SimpleNamespace:
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    max_tool_rounds = 3
    tool_call = ToolCall(
        name="ha_call_service",
        arguments={"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"},
    )
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="do an oddly complicated thing")])
    brain = fake_brain(replies=[BrainReply(tool_calls=[tool_call]) for _ in range(max_tool_rounds)])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        _AlwaysToolHost(),
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=max_tool_rounds,
        timings=timings,
    )

    assert brain.call_count == max_tool_rounds
    assert tts.received_text == [_TOO_MANY_ROUNDS_REPLY]
    assert timings.turn_outcome == "round_cap"


async def test_empty_tool_call_free_reply_falls_back_to_a_spoken_reply(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """WR-01 (phase 01 code review): a `chat.completions` reply with no tool
    calls and no text is a valid, reachable shape (the model stops early
    against `max_tokens`, or simply has nothing to add) -- not an error, but
    left unhandled it reaches text-to-speech as an empty string and the
    operator hears silence with no indication the turn ended. This asserts
    the fallback fires instead, the same way VOICE-08's empty-transcript
    case does.
    """
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import _EMPTY_REPLY, run_turn

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what do you think")])
    brain = fake_brain(replies=[BrainReply(text="")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert brain.call_count == 1
    assert tts.received_text == [_EMPTY_REPLY]
    assert timings.turn_outcome == "empty_reply"
