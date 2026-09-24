"""CMD-06/CMD-07 at three or more actions -- the model-issued multi-action
path, not the macro path.

`tests/test_macros.py`'s two partial-failure tests drive `fire_macro`, which
is deliberately sequential and stops on the first failure -- a different,
correct contract for a fixed set of actions an operator authored. Nothing in
this file drives that path, and nothing here matches `fire_macro`'s
stop-early behaviour: a command a person spoke in one sentence, that the
language model turned into three or more separate tool calls, is a
different situation, and CMD-06/CMD-07 are named for it specifically.

The first test below (`test_a_three_action_command_with_a_denied_first_action_runs_all_three`)
was run against the unfixed `_run_tool_rounds` before any production code in
this plan changed -- its failure output is recorded in this plan's SUMMARY
under the "CMD-07 finding" heading. `.planning/REQUIREMENTS.md`'s CMD-07
entry carries the dated correction this test's failure proved was owed.

The tests from `test_a_mixed_batch_carries_the_failed_action_s_reason_verbatim`
onward exercise Task 2's `_compose_mixed_outcome_reply`: the per-action
summary composed in code, never through a second `brain.chat` round.
"""

from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from typing import Any

import pytest
from mcp.types import CallToolResult, TextContent

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn


class _RecordingToolHost:
    """Records every `(name, arguments)` pair `call_tool` receives, in the
    order it received them, and returns a scripted result keyed by
    `arguments["entity_id"]` -- never by arrival order -- so a caller can
    delay one action's return without changing which result it gets back.
    That is what makes the out-of-order test below prove something: the
    delayed call still resolves to its OWN scripted result, and
    `_run_tool_rounds` still reports it against the `call_{i}` its position
    in `reply.tool_calls` says it should, never against whichever call
    happened to finish first (D-05).
    """

    def __init__(
        self,
        results_by_entity: dict[str, Any],
        delays_by_entity: dict[str, float] | None = None,
    ) -> None:
        self._results = dict(results_by_entity)
        self._delays = dict(delays_by_entity or {})
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        entity_id = arguments["entity_id"]
        delay = self._delays.get(entity_id, 0.0)
        if delay:
            await asyncio.sleep(delay)
        return self._results[entity_id]


class _RecordingBrain:
    """A `_BrainProvider`-shaped double recording every `messages` list it
    was called with -- the same double `tests/test_turn_controller.py`
    already names `_RecordingBrain` for the same reason (01.1-06 read_first
    pointer), duplicated here rather than imported so this file has no
    import-order dependency on that test module.
    """

    def __init__(self, replies: list[BrainReply]) -> None:
        self._replies = list(replies)
        self.received_messages: list[list[dict[str, Any]]] = []
        self.call_count = 0

    async def chat(self, messages, tools=None) -> BrainReply:
        self.received_messages.append([dict(m) for m in messages])
        if not self._replies:
            raise AssertionError("_RecordingBrain.chat called more times than scripted")
        self.call_count += 1
        return self._replies.pop(0)


def _service_call(entity_id: str) -> ToolCall:
    return ToolCall(
        name="ha_call_service",
        arguments={"domain": "switch", "service": "turn_off", "entity_id": entity_id},
    )


def _ok(text: str = "{}") -> SimpleNamespace:
    return SimpleNamespace(isError=False, content=[SimpleNamespace(text=text)])


def _denied(reason: str) -> SimpleNamespace:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=reason)])


# 260924-4it: helpers for the first-round done shortcut's own tests.


def _changed(entity_id: str, structured: bool = True) -> CallToolResult:
    """A real `handle_call_service`-shaped 2xx reply: `{"changed": [...]}`,
    with or without `structured_content` set -- both are what the real MCP
    framework can hand back, and the shortcut must fire either way."""
    payload = {"changed": [{"entity_id": entity_id, "state": "off"}]}
    return CallToolResult(
        content=[TextContent(type="text", text=json.dumps(payload, indent=2))],
        structured_content=payload if structured else None,
    )


def _text_result(text: str) -> CallToolResult:
    return CallToolResult(content=[TextContent(type="text", text=text)])


class _ByNameToolHost:
    """Records every `(name, arguments)` call and returns a scripted result
    keyed by tool name."""

    def __init__(self, results_by_name: dict[str, Any]) -> None:
        self._results = dict(results_by_name)
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return self._results[name]


async def test_a_three_action_command_with_a_denied_first_action_runs_all_three(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """THE CMD-07 FINDING.

    Three tool calls arrive in one model round: the first is denied, the
    second and third would succeed. `_run_tool_rounds` returned the instant
    the first call came back error-shaped -- the second and third actions in
    a three-action sentence never ran at all. This is the gap CMD-07's tick
    was never actually tested against: the existing partial-failure tests
    (`tests/test_macros.py`) drive `fire_macro`, the macro path, whose
    stop-on-first-failure is a different, deliberate contract.
    """
    tool_host = _RecordingToolHost(
        {
            "switch.example_server_socket": _denied("that one is off limits"),
            "switch.example_fan": _ok(),
            "light.example_lamp": _ok(),
        }
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the server socket, the fan, and the lamp")])
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    _service_call("switch.example_server_socket"),
                    _service_call("switch.example_fan"),
                    _service_call("light.example_lamp"),
                ]
            )
        ]
    )
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    # The tool host recorded three calls, not one: every action ran, even
    # though the first came back error-shaped.
    assert len(tool_host.calls) == 3, (
        f"expected all three actions to run; the tool host recorded {len(tool_host.calls)} call(s): "
        f"{tool_host.calls!r}"
    )
    assert [call[1]["entity_id"] for call in tool_host.calls] == [
        "switch.example_server_socket",
        "switch.example_fan",
        "light.example_lamp",
    ]
    # The first action's refusal reaches speech byte-for-byte -- compared
    # against the FULL reason string with `in`, never a substring of it
    # (a paraphrase that softened the words but kept one of them would still
    # pass a single-word check, and is exactly what this comparison exists
    # to catch).
    assert "that one is off limits" in tts.received_text[-1]
    # The reply is not composed of success clauses alone: three actions ran,
    # so a reply that is nothing but the fixed success phrase would need it
    # three times -- it can appear at most twice, since the first action was
    # denied.
    assert tts.received_text[-1].count("succeeded") < 3
    # No second `brain.chat` call was made for this batch (D-14): the
    # composed sentence never passed through a second inference round.
    assert brain.call_count == 1


async def test_results_correlate_to_call_index_never_completion_order(fake_audio_source, fake_stt, fake_tts):
    """The slowest-to-resolve call is the FIRST-dispatched one -- proving the
    reported result for each action is tied to its position in
    `reply.tool_calls`, never to which call happened to finish first (D-05).
    """
    tool_host = _RecordingToolHost(
        {
            "switch.example_fan": _ok('{"ok":"fan"}'),
            "light.example_lamp": _ok('{"ok":"lamp"}'),
            "switch.example_server_socket": _ok('{"ok":"socket"}'),
        },
        delays_by_entity={"switch.example_fan": 0.05},
    )
    brain = _RecordingBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    _service_call("switch.example_fan"),
                    _service_call("light.example_lamp"),
                    _service_call("switch.example_server_socket"),
                ]
            ),
            BrainReply(text="all done"),
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the fan, the lamp, and the socket")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    assert len(brain.received_messages) == 2
    second_round_messages = brain.received_messages[1]
    tool_messages = [m for m in second_round_messages if m["role"] == "tool"]
    assert [m["tool_call_id"] for m in tool_messages] == ["call_0", "call_1", "call_2"]
    assert [m["content"] for m in tool_messages] == [
        '{"ok":"fan"}',
        '{"ok":"lamp"}',
        '{"ok":"socket"}',
    ]


async def test_three_action_all_success_batch_is_byte_identical_to_before(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """Behaviour #4: three tool calls that all succeed still cost a second
    `brain.chat` round, and the reply text is that round's own text -- the
    concurrent dispatch this plan adds must not change the all-success shape
    at all.

    260924-4it: its `_ok()` results carry "{}", not Home Assistant's own
    `{"changed": [...]}` reply, so the done shortcut does not apply here and
    the second round still runs.
    """
    tool_host = _RecordingToolHost(
        {
            "switch.example_fan": _ok(),
            "light.example_lamp": _ok(),
            "switch.example_server_socket": _ok(),
        }
    )
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    _service_call("switch.example_fan"),
                    _service_call("light.example_lamp"),
                    _service_call("switch.example_server_socket"),
                ]
            ),
            BrainReply(text="all three are off now"),
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the fan, the lamp, and the socket")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    assert len(tool_host.calls) == 3
    assert brain.call_count == 2
    assert tts.received_text == ["all three are off now"]


async def test_a_mixed_batch_carries_the_failed_action_s_reason_verbatim(
    fake_audio_source, fake_stt, fake_tts
):
    """The byte-for-byte assertion this test exists to be: compared against
    the FULL refusal reason string, not a single word of it -- a paraphrase
    that softened "that one is off limits" to, say, "that one is
    restricted" would still contain the word "that" and would pass a
    substring check against a single word. It must not pass this one.
    """
    reason = "that one is off limits"
    tool_host = _RecordingToolHost(
        {
            "switch.example_server_socket": _denied(reason),
            "switch.example_fan": _ok(),
            "light.example_lamp": _ok(),
        }
    )
    brain = _RecordingBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    _service_call("switch.example_server_socket"),
                    _service_call("switch.example_fan"),
                    _service_call("light.example_lamp"),
                ]
            )
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the server socket, the fan, and the lamp")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    reply_text = tts.received_text[-1]
    assert reason in reply_text
    # Not merely a single word of the reason -- the whole sentence, unbroken.
    assert reply_text.count(reason) == 1
    # No second `brain.chat` call for this batch: the composed sentence
    # never passed through a second inference round (D-14).
    assert brain.call_count == 1
    assert len(brain.received_messages) == 1


async def test_a_mixed_batch_s_clause_order_follows_tool_calls_order_not_completion_order(
    fake_audio_source, fake_stt, fake_tts
):
    """The denied action is dispatched LAST and resolves FIRST (no delay);
    the two successful actions are dispatched first and resolve later. The
    composed sentence must still read in `reply.tool_calls` order -- the
    fan's clause first, the lamp's second, the denied socket's third -- never
    in the order the fake host happened to finish them.
    """
    tool_host = _RecordingToolHost(
        {
            "switch.example_fan": _ok(),
            "light.example_lamp": _ok(),
            "switch.example_server_socket": _denied("that one is off limits"),
        },
        delays_by_entity={"switch.example_fan": 0.05, "light.example_lamp": 0.03},
    )
    brain = _RecordingBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    _service_call("switch.example_fan"),
                    _service_call("light.example_lamp"),
                    _service_call("switch.example_server_socket"),
                ]
            )
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the fan, the lamp, and the server socket")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    reply_text = tts.received_text[-1]
    # The fan and lamp clauses (both "succeeded") come before the denied
    # socket's clause -- `reply.tool_calls` order, even though the fake host
    # actually resolved the socket (no delay) before the fan or the lamp
    # (delayed 0.05s and 0.03s respectively).
    assert reply_text.index("succeeded") < reply_text.index("that one is off limits")
    assert reply_text.count("succeeded") == 2


async def test_a_raised_tool_call_is_reported_as_a_failed_clause_not_swallowed_or_raised(
    fake_audio_source, fake_stt, fake_tts
):
    """One action's `call_tool` raises instead of returning a result. The
    turn does not crash, the raised action is named in the spoken reply, and
    its wording is distinguishable from an ordinary boundary refusal -- "it
    never even ran" is a different fact from "it ran and was refused."
    """

    class _RaisingToolHost:
        def __init__(self) -> None:
            self.calls: list[tuple[str, dict]] = []

        async def call_tool(self, name, arguments):
            self.calls.append((name, dict(arguments)))
            if arguments["entity_id"] == "switch.example_fan":
                raise RuntimeError("the stdio pipe closed mid-call")
            return _ok()

    tool_host = _RaisingToolHost()
    brain = _RecordingBrain(
        replies=[
            BrainReply(
                tool_calls=[
                    _service_call("switch.example_fan"),
                    _service_call("light.example_lamp"),
                ]
            )
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the fan and the lamp")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    # Both actions still ran -- a raised exception in the gather does not
    # cancel its siblings (return_exceptions=True, never a TaskGroup).
    assert len(tool_host.calls) == 2
    reply_text = tts.received_text[-1]
    assert "did not complete" in reply_text
    # Never worded the same as a boundary refusal -- these are different
    # facts about the house.
    assert "off limits" not in reply_text
    assert brain.call_count == 1


# 260924-4it: the first-round done shortcut.


@pytest.mark.parametrize("structured", (True, False))
async def test_an_all_action_first_round_speaks_done_with_no_second_round(
    structured, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """Behaviour item 1: three ha_call_service calls in round one, each a
    real `{"changed": [...]}` Home Assistant reply, speak "done" after
    exactly one `brain.chat` call -- with `structured_content` set, and
    with only the JSON text block.
    """
    tool_host = _RecordingToolHost(
        {
            "switch.example_fan": _changed("switch.example_fan", structured=structured),
            "light.example_lamp": _changed("light.example_lamp", structured=structured),
            "switch.example_server_socket": _changed("switch.example_server_socket", structured=structured),
        }
    )
    brain = fake_brain(
        replies=[
            BrainReply(
                tool_calls=[
                    _service_call("switch.example_fan"),
                    _service_call("light.example_lamp"),
                    _service_call("switch.example_server_socket"),
                ]
            )
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the fan, the lamp, and the socket")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    assert tts.received_text == ["done"]
    assert brain.call_count == 1
    assert len(tool_host.calls) == 3
    assert timings.turn_outcome == "completed"


_NOT_PLAIN_ACTION_SUCCESS_CASES = (
    pytest.param(
        [ToolCall(name="ha_list_entities", arguments={}), _service_call("switch.example_fan")],
        {"ha_list_entities": _text_result("[]"), "ha_call_service": _changed("switch.example_fan")},
        "turn off the fan",
        id="read_plus_action",
    ),
    pytest.param(
        [
            ToolCall(
                name="example__ha_call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
            )
        ],
        {"example__ha_call_service": _changed("switch.example_fan")},
        "turn off the fan",
        id="collision_prefixed_name",
    ),
    pytest.param(
        [_service_call("switch.example_fan")],
        {"ha_call_service": _text_result(json.dumps({"error": "home assistant returned 500: example failure"}))},
        "turn off the fan",
        id="error_payload",
    ),
    pytest.param(
        [_service_call("switch.example_fan")],
        {
            "ha_call_service": CallToolResult(
                content=[
                    TextContent(
                        type="text",
                        text=json.dumps({"changed": [], "response": {"todo.example_list": {"items": []}}}),
                    )
                ],
                structured_content={"changed": [], "response": {"todo.example_list": {"items": []}}},
            )
        },
        "turn off the fan",
        id="response_payload",
    ),
    pytest.param(
        [_service_call("switch.example_fan")],
        {"ha_call_service": _text_result("ok")},
        "turn off the fan",
        id="text_only_result",
    ),
    pytest.param(
        [_service_call("switch.example_fan")],
        {"ha_call_service": _changed("switch.example_fan")},
        "turn off the fan and what's the temperature",
        id="asks_for_information_transcript",
    ),
)


@pytest.mark.parametrize("tool_calls, results_by_name, transcript", _NOT_PLAIN_ACTION_SUCCESS_CASES)
async def test_a_round_that_is_not_plain_action_success_still_reaches_the_model(
    tool_calls, results_by_name, transcript, fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """Behaviour item 3: a read next to the action, a collision-prefixed
    name, a Home Assistant `{"error": ...}` payload, a service-response
    payload, an unparseable result, and a question in the transcript all
    still get the second `brain.chat` round -- none of these is plain
    action success.
    """
    tool_host = _ByNameToolHost(results_by_name)
    brain = fake_brain(
        replies=[BrainReply(tool_calls=tool_calls), BrainReply(text="phrased by the model")]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text=transcript)])
    tts = fake_tts(chunks=[b"\x01\x02"])
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
    assert tts.received_text == ["phrased by the model"]


async def test_model_text_next_to_the_tool_call_still_gets_a_phrasing_round(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """Behaviour item 3's seventh case: the model attached text of its own
    next to the tool call, so the shortcut must not fire even though the
    round is otherwise plain action success."""
    tool_host = _RecordingToolHost({"switch.example_fan": _changed("switch.example_fan")})
    brain = fake_brain(
        replies=[
            BrainReply(tool_calls=[_service_call("switch.example_fan")], text="working on it"),
            BrainReply(text="phrased by the model"),
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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
    assert tts.received_text == ["phrased by the model"]


async def test_an_action_after_a_read_round_still_gets_a_phrasing_round(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """Behaviour item 4: round one reads state, round two calls a service
    with a `{"changed"}` payload, round three carries text -- the shortcut
    only ever looks at round 0, so this three-round turn still costs three
    `brain.chat` calls and speaks the third round's own text."""
    tool_host = _ByNameToolHost(
        {
            "ha_get_state": _text_result(json.dumps({"entity_id": "sensor.example_server_power", "state": "42.0"})),
            "ha_call_service": _changed("switch.example_fan"),
        }
    )
    brain = fake_brain(
        replies=[
            BrainReply(tool_calls=[ToolCall(name="ha_get_state", arguments={"entity_id": "sensor.example_server_power"})]),
            BrainReply(tool_calls=[_service_call("switch.example_fan")]),
            BrainReply(text="power draw checked and the fan is off"),
        ]
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="check the server power then turn off the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])
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

    assert brain.call_count == 3
    assert tts.received_text == ["power draw checked and the fan is off"]


async def test_only_the_top_tier_reaches_the_tool_host_when_the_shortcut_fires(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_envelope_client
):
    """D-05, 260924-4it: the done shortcut fires on the top tier's own tool
    round, and only the top tier ever reaches the tool host -- a triage
    tier's envelope client still gets exactly one call, carrying no `tools`
    keyword, and the top tier's own envelope client gets none (it has no
    envelope client at all, per Task 1)."""
    from atlas.providers.tier_reply import FillerPhrase, TierReply
    from atlas.turn import brain_race

    tool_host = _RecordingToolHost({"switch.example_fan": _changed("switch.example_fan")})

    triage_reply = TierReply(answer="", confident=False, needs_tool=True, filler=FillerPhrase.STILL_LOOKING)
    triage_tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=fake_envelope_client(reply=triage_reply, delay_s=0.0),
        calls_tools=False,
    )

    top_answer = "turned it off"
    top_reply = TierReply(answer=top_answer, confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
    top_brain = fake_brain(replies=[BrainReply(tool_calls=[_service_call("switch.example_fan")])])
    top_tier = brain_race.TierBrain(
        index=1,
        model="top-model",
        brain=top_brain,
        envelope_client=fake_envelope_client(reply=top_reply, delay_s=0.0),
        calls_tools=True,
    )

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn off the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        top_brain,
        tts,
        tool_host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        tiers=[triage_tier, top_tier],
        filler_after_ms=0,
        filler_cache=None,
    )

    assert len(tool_host.calls) == 1
    assert tts.received_text == ["done"]
    assert len(triage_tier.envelope_client.calls) == 1
    assert "tools" not in triage_tier.envelope_client.calls[0]
    assert top_tier.envelope_client.calls == []
