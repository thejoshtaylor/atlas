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

Task 2 extends this file with the mixed-outcome composer's own tests, once
the composer exists.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from typing import Any

from spire_voice.providers.base import BrainReply, FinalTranscript, ToolCall
from spire_voice.timing import TurnTimings
from spire_voice.turn.controller import run_turn


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
    # The first action's refusal reaches speech byte-for-byte.
    assert "that one is off limits" in tts.received_text[-1]
    # The reply is not composed of success clauses alone -- Task 1 does not
    # yet build the mixed-outcome composer (that is Task 2), so this only
    # proves the refusal text still reaches speech rather than being
    # silently replaced by a confirmation.
    assert tts.received_text[-1] != "done, done, done"


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
