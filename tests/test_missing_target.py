"""261001-ibf: a command that names no device calls no tool and no brain.

Two live turns (2026-10-01) paused after "Turn off." and "Turn on the.". The
brain guessed a target and switched real devices. `run_turn` now asks which
device, in code, and opens a clarification window whose answer may reach
only `ha_call_service`. Every name below is invented.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from atlas.app import _catalog_prompt
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import AnswerScope, FollowUpChannel, FollowUpRequest

from brain_fakes import RecordingFakeBrain
from conftest import FakeAudioSource, FakeStt, FakeTts

_HA_SCHEMA = [
    {"type": "function", "function": {"name": "weather_now"}},
    {"type": "function", "function": {"name": "ha_call_service"}},
]


class _CountingToolHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])


async def _state_fetch() -> list[dict[str, Any]]:
    return [{"entity_id": "switch.example_fan", "friendly_name": "Example Fan", "state": "on"}]


async def _run(
    transcript: str,
    brain: RecordingFakeBrain,
    *,
    incoming: FollowUpRequest | None = None,
    channel: bool = True,
    schema: list[dict[str, Any]] | None = None,
) -> tuple[TurnTimings, FakeTts, FakeAudioSource, _CountingToolHost]:
    source = FakeAudioSource(frames=[b"\x00\x01"])
    if channel:
        source.follow_up = FollowUpChannel(incoming=incoming)
    host = _CountingToolHost()
    tts = FakeTts(chunks=[b"\x01"])
    timings = TurnTimings()
    await run_turn(
        source,
        FakeStt(events=[FinalTranscript(text=transcript)]),
        brain,
        tts,
        host,
        tools_schema=_HA_SCHEMA if schema is None else schema,
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        state_fetch=_state_fetch,
        local_intents=True,
    )
    return timings, tts, source, host


@pytest.mark.parametrize(("transcript", "question"), [("Turn off.", "turn off what?"), ("Turn on the.", "turn on what?")])
async def test_a_bare_verb_asks_which_device_and_calls_nothing(transcript, question):
    brain = RecordingFakeBrain()
    timings, tts, _source, host = await _run(transcript, brain)

    assert timings.turn_outcome == "missing_target"
    assert tts.received_text == [question]
    assert brain.call_count == 0
    assert host.calls == []


async def test_the_question_opens_a_clarification_window_that_reaches_only_ha_call_service():
    brain = RecordingFakeBrain()
    _timings, _tts, source, _host = await _run("Turn off.", brain)

    request = source.follow_up.requested
    assert request is not None
    assert request.kind == "clarification"
    assert request.question == "turn off what?"
    assert request.original_transcript == "Turn off."
    assert request.answer_scope == AnswerScope(tool_names=frozenset({"ha_call_service"}), entity_ids=None)


async def test_the_window_scope_has_no_tool_when_the_schema_offers_no_ha_call_service():
    brain = RecordingFakeBrain()
    _timings, _tts, source, _host = await _run(
        "Turn off.", brain, schema=[{"type": "function", "function": {"name": "weather_now"}}]
    )

    assert source.follow_up.requested.answer_scope == AnswerScope(tool_names=frozenset())


async def test_the_window_scope_is_narrowed_by_the_scope_of_the_turn():
    brain = RecordingFakeBrain()
    incoming = FollowUpRequest(
        kind="clarification",
        chain_depth=1,
        original_transcript="which one",
        question="which one?",
        answer_scope=AnswerScope(
            tool_names=frozenset({"ha_call_service"}), entity_ids=frozenset({"switch.example_fan"})
        ),
        playback_ends_at=0.0,
    )
    _timings, _tts, source, _host = await _run("turn on", brain, incoming=incoming)

    assert source.follow_up.requested.answer_scope == AnswerScope(
        tool_names=frozenset({"ha_call_service"}), entity_ids=frozenset({"switch.example_fan"})
    )
    assert source.follow_up.requested.chain_depth == 2


async def test_a_source_with_no_follow_up_channel_asks_and_opens_no_window():
    brain = RecordingFakeBrain()
    timings, tts, _source, host = await _run("Turn off.", brain, channel=False)

    assert timings.turn_outcome == "missing_target"
    assert tts.received_text == ["turn off what?"]
    assert brain.call_count == 0
    assert host.calls == []


async def test_a_pronoun_with_nothing_to_refer_to_asks_which_device():
    brain = RecordingFakeBrain()
    timings, tts, _source, _host = await _run("turn it on", brain)

    assert timings.turn_outcome == "missing_target"
    assert tts.received_text == ["turn on what?"]
    assert brain.call_count == 0


async def test_the_answer_to_the_question_runs_the_brain_with_the_earlier_exchange_first():
    first = RecordingFakeBrain()
    _timings, _tts, source, _host = await _run("Turn off.", first)
    request = source.follow_up.requested

    call = ToolCall(
        name="ha_call_service",
        arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
    )
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[call]), BrainReply(text="Done.")])
    timings, _tts2, _source2, host = await _run("the example fan", brain, incoming=request)

    assert brain.call_count == 2
    assert brain.calls[0].messages[-3:] == [
        {"role": "user", "content": "Turn off."},
        {"role": "assistant", "content": "turn off what?"},
        {"role": "user", "content": "the example fan"},
    ]
    assert [name for name, _arguments in host.calls] == ["ha_call_service"]
    assert timings.turn_outcome == "completed"


@pytest.mark.parametrize("transcript", ["lights off", "turn off the example fan", "stop"])
async def test_a_command_with_a_target_is_not_caught(transcript):
    brain = RecordingFakeBrain(replies=[BrainReply(text="ok")])
    timings, _tts, _source, _host = await _run(transcript, brain)

    assert timings.turn_outcome != "missing_target"


async def test_turn_it_off_in_a_window_whose_scope_offers_a_tool_still_reaches_the_brain():
    incoming = FollowUpRequest(
        kind="answer",
        chain_depth=1,
        original_transcript="what is the weather",
        question="It is sunny.",
        answer_scope=AnswerScope(tool_names=frozenset({"weather_now"})),
        playback_ends_at=0.0,
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="Done.")])
    timings, _tts, _source, _host = await _run("turn it off", brain, incoming=incoming)

    assert timings.turn_outcome == "completed"
    assert brain.call_count == 1


def test_the_catalog_prompt_forbids_a_tool_call_when_no_device_is_named():
    prompt = _catalog_prompt([])

    assert "names no device" in prompt
    assert "do not call a tool" in prompt
    assert "Ask which device" in prompt
