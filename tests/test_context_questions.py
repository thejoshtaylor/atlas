"""Turn-level proofs for CMD-03, CMD-04, and CMD-05 (04-CONTEXT.md D-01
through D-04).

Each test here asserts on where a call went or whether one happened at
all, never on how the spoken sentence read -- reading only the answer
would pass a turn that got the words right after a pointless tool round,
which is exactly the failure CMD-03/CMD-04 forbid (D-04).

Task 1 (CMD-03): a time or date question is answered from the per-turn
state message with zero calls to the tool host at all -- the production
turn always starts a state fetch before the transcript is drained (see the
test's own docstring below for the caveat this repository has already had
to correct twice), but that fetch is not a call made *in service of*
answering the time; a `get_time` tool would be.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from spire_mcp.ha import handle_call_service
from spire_mcp.safety import Denied, Policy

from spire_voice.mcp_client import McpToolHostLookup
from spire_voice.providers.base import BrainReply, FinalTranscript, ToolCall
from spire_voice.providers.tier_reply import FillerPhrase, TierReply
from spire_voice.timing import TurnTimings
from spire_voice.turn import brain_race
from spire_voice.turn.controller import run_turn


class _RecordingToolHost:
    """Records every `(name, arguments)` pair `call_tool` receives, in
    order, and returns a scripted result keyed by tool name.

    Unlike the `_RaisingToolHost` fixture `test_turn_controller.py` uses
    for its own zero-tool-call proof, this host does not raise on a call
    -- it records one, so a test can assert the recorded list is empty
    (or, for the weather-routing proof in a later task, assert exactly
    which host received the call), rather than relying on an uncalled
    `AssertionError` to prove the same thing indirectly.
    """

    def __init__(self, results_by_tool: dict[str, Any] | None = None) -> None:
        self._results = dict(results_by_tool or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return self._results.get(name)


async def test_a_time_question_is_answered_with_zero_tool_calls(
    fake_audio_source, fake_stt, fake_tts, fake_envelope_client
):
    """CMD-03: a confident triage tier answers a spoken time question from
    the state message `run_turn` already injects, with zero calls
    recorded on the tool host.

    Caveat, stated plainly per this plan's own instruction (and the two
    prior corrections this project has already had to make on exactly
    this kind of claim): a production turn always starts a `state_fetch`
    before the transcript is drained (`run_turn`'s own dispatch order) --
    that fetch is itself one call to the tool host (`ha_list_entities`),
    made on every turn regardless of what was asked. This test supplies
    `state_fetch` as a fake that does not go through the tool host at all
    (mirroring `test_criterion_4_a_confident_triage_tier_answers_with_zero_tool_calls`
    in `tests/test_turn_controller.py`), so the zero this test asserts is
    a clean zero: no call was made in service of answering the time
    question itself, which is the claim CMD-03 is actually about -- not
    that a turn makes no tool call ever.
    """
    triage_reply = TierReply(
        answer="it's a quarter past two",
        confident=True,
        needs_tool=False,
        filler=FillerPhrase.ONE_MOMENT,
    )
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=triage_reply),
        calls_tools=False,
    )

    async def _state_fetch() -> list[dict[str, str]]:
        return [{"entity_id": "light.example_lamp", "friendly_name": "the lamp", "state": "on"}]

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what time is it")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    tool_host = _RecordingToolHost()
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        None,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier],
        state_fetch=_state_fetch,
    )

    assert tts.received_text == ["it's a quarter past two"]
    assert tool_host.calls == []
    assert len(triage_tier.envelope_client.calls) == 1


# --- Task 3: CMD-04, the weather pair (routed correctly, and degraded honestly) ---


class _FakeAdvertisedTool:
    """A `.name`-bearing stand-in for `mcp.types.Tool` -- all
    `McpToolHostLookup.__init__` reads off a host's `.tools` list to build
    its name-to-host map."""

    def __init__(self, name: str) -> None:
        self.name = name


class _RecordingLookupHost:
    """A `_ToolHost`-shaped fake carrying its own advertised tool names
    (what `McpToolHostLookup` routes by) and a scripted result per tool
    name. Records every call it actually receives, in order -- the
    weather-routing tests below assert on which host received a call
    (D-04), never only on what the spoken answer said.
    """

    def __init__(self, tool_names: list[str], results_by_tool: dict[str, Any] | None = None) -> None:
        self.tools = [_FakeAdvertisedTool(name) for name in tool_names]
        self._results = dict(results_by_tool or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return self._results.get(name)


async def test_a_weather_question_reaches_the_weather_host_and_never_home_assistant(
    fake_audio_source, fake_stt, fake_tts, fake_brain
):
    """CMD-04, D-04: a spoken weather question drives exactly one tool
    call, and the two-host lookup routes it to the weather host -- the
    Home Assistant host records none. Asserted on where the call went,
    not on what the sentence said: asserting only on the spoken text
    would pass a turn that answered correctly after a pointless round
    trip to the house, which is the exact thing CMD-04 forbids.
    """
    ha_host = _RecordingLookupHost(["ha_call_service", "ha_get_state", "ha_list_entities"])
    weather_host = _RecordingLookupHost(
        ["weather_current", "weather_forecast"],
        results_by_tool={
            "weather_current": SimpleNamespace(
                is_error=False,
                content=[SimpleNamespace(text="18.5 degrees and cloudy")],
            )
        },
    )
    lookup = McpToolHostLookup([ha_host, weather_host])

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what's the weather like outside")])
    brain = fake_brain(
        replies=[
            BrainReply(tool_calls=[ToolCall(name="weather_current", arguments={})]),
            BrainReply(text="it's 18.5 degrees and cloudy outside"),
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        lookup,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == ["it's 18.5 degrees and cloudy outside"]
    assert weather_host.calls == [("weather_current", {})]
    # The assertion CMD-04 is actually about: nothing reached the house.
    assert ha_host.calls == []


async def test_an_unreachable_weather_upstream_reaches_speech_in_the_childs_own_words(
    fake_audio_source, fake_stt, fake_tts, fake_brain
):
    """An unreachable weather upstream produces a spoken sentence carrying
    the child's own words (the same error-shaped-result-carries-its-own-
    text discipline `mcp_client.py`'s own docstring states for a `Denied`),
    and the turn ends normally -- no crash, no second brain round to
    compose a paraphrase (D-14)."""
    unreachable_sentence = "i can't reach the weather service right now"
    ha_host = _RecordingLookupHost(["ha_call_service"])
    weather_host = _RecordingLookupHost(
        ["weather_current"],
        results_by_tool={
            "weather_current": SimpleNamespace(
                is_error=True,
                content=[SimpleNamespace(text=unreachable_sentence)],
            )
        },
    )
    lookup = McpToolHostLookup([ha_host, weather_host])

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="what's the weather like outside")])
    brain = fake_brain(replies=[BrainReply(tool_calls=[ToolCall(name="weather_current", arguments={})])])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        lookup,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
    )

    assert tts.received_text == [unreachable_sentence]
    # No second brain.chat round composed a paraphrase of the failure.
    assert brain.call_count == 1
    assert weather_host.calls == [("weather_current", {})]
    assert ha_host.calls == []


# --- Task 3: CMD-05, a denied entity still answers a question about its
# power use, and is still refused for control (D-03) ---


class _RealHomeAssistantToolHost:
    """Calls straight into the real `handle_call_service` against
    `fake_ha` and a real `Policy` -- the same shape
    `tests/test_safety_integration.py`'s own `_RefusingToolHost` uses,
    duplicated here rather than imported (matching `test_multi_action.py`'s
    own stated reason for doing the same: no import-order dependency
    between test modules) so a `Denied` raised by the real safety boundary
    reaches this test exactly as it would reach a real turn.
    """

    def __init__(self, ha: Any, policy: Policy) -> None:
        self._ha = ha
        self._policy = policy

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
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


async def test_a_control_denied_entitys_power_reading_still_answers_and_control_is_still_refused(
    fake_audio_source, fake_stt, fake_tts, fake_brain, fake_envelope_client, fake_ha
):
    """CMD-05, D-03: `allow_read` never blocks and live state is already
    injected -- `04-CONTEXT.md`'s own position is that CMD-05 needs no new
    code, only a turn-level proof against an entity the running policy
    denies for control. Both facts must hold together in one file: a read
    succeeding proves nothing about a control call still being refused,
    and a control call being refused proves nothing about a read still
    working -- CMD-05 is the two facts holding together, and a test
    proving only one would not notice a regression in the other.
    """
    example_entity = "switch.example_server_socket"
    policy = Policy.from_config({"deny_entities": [example_entity]})

    # Half one: the read answers, at the turn level, from injected state
    # -- the same zero-tool-call shape CMD-03's own test above uses, since
    # this is exactly how a live-state question is already answered.
    triage_reply = TierReply(
        answer="it's drawing 42 watts",
        confident=True,
        needs_tool=False,
        filler=FillerPhrase.ONE_MOMENT,
    )
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=triage_reply),
        calls_tools=False,
    )

    async def _state_fetch() -> list[dict[str, str]]:
        return [{"entity_id": example_entity, "friendly_name": "the server socket", "state": "42.0"}]

    read_source = fake_audio_source(frames=[b"\x00\x01"])
    read_stt = fake_stt(events=[FinalTranscript(text="how much power is the server socket using")])
    read_tts = fake_tts(chunks=[b"\x01\x02"])
    read_tool_host = _RecordingLookupHost(["ha_call_service", "ha_get_state"])
    read_timings = TurnTimings()

    await run_turn(
        read_source,
        read_stt,
        None,
        read_tts,
        read_tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=read_timings,
        tiers=[triage_tier],
        state_fetch=_state_fetch,
    )

    assert read_tts.received_text == ["it's drawing 42 watts"]
    assert read_tool_host.calls == []

    # Half two: the same entity, asked to be switched, is still refused --
    # through the real allow_call boundary, not a fake that assumes it.
    control_source = fake_audio_source(frames=[b"\x00\x01"])
    control_stt = fake_stt(events=[FinalTranscript(text="turn off the server socket")])
    control_brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    ToolCall(
                        name="ha_call_service",
                        arguments={
                            "domain": "switch",
                            "service": "turn_off",
                            "entity_id": example_entity,
                        },
                    )
                ]
            ),
        ]
    )
    control_tts = fake_tts(chunks=[b"\x01\x02"])
    control_tool_host = _RealHomeAssistantToolHost(fake_ha, policy)
    control_timings = TurnTimings()

    await run_turn(
        control_source,
        control_stt,
        control_brain,
        control_tts,
        control_tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=control_timings,
    )

    assert control_tts.received_text == ["that one is off limits"]
    # Nothing left the process for the denied call.
    assert len(fake_ha.requests) == 0
