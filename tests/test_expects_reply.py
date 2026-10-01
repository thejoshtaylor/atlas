"""261001-dlp: the brain's own "I asked a question" flag.

The top tier's tool loop sends a strict `spoken_reply` json_schema on every
round and reads the final round as `{"answer", "expects_reply"}`. The flag
travels on `HandoffSlot.expects_reply` into `TierReply.expects_reply`. A reply
that code composes never sets it. Every name and entity id here is invented.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.providers.base import BrainReply, ToolCall
from atlas.providers.tier_reply import FillerPhrase, TierReply
from atlas.timing import TurnTimings
from atlas.turn import brain_race, controller
from atlas.turn.controller import _EMPTY_REPLY, _SPOKEN_REPLY_FORMAT, _parse_spoken_reply, _run_tool_rounds
from atlas.turn.handoff import HandoffSlot

from brain_fakes import RecordingFakeBrain

_SCHEMA = [
    {"type": "function", "function": {"name": "weather_now"}},
    {"type": "function", "function": {"name": "lights_set"}},
]
_WEATHER = ToolCall(name="weather_now", arguments={"place": "home"})
_QUESTION = json.dumps({"answer": "what do you want me to turn on?", "expects_reply": True})


class _Host:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])


async def _rounds(brain: Any, slot: "HandoffSlot | None", **kwargs: Any) -> tuple[str, TurnTimings]:
    timings = TurnTimings()
    text = await _run_tool_rounds(
        brain,
        _Host(),
        _SCHEMA,
        [{"role": "system", "content": "you answer"}, {"role": "user", "content": "turn on the"}],
        kwargs.pop("max_tool_rounds", 3),
        timings,
        handoff_slot=slot,
        **kwargs,
    )
    return text, timings


# --- the spoken_reply format on every round -------------------------------------


async def test_every_tool_round_sends_the_spoken_reply_format_with_the_tools_on_offer():
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_WEATHER]), BrainReply(text=_QUESTION)])

    text, _ = await _rounds(brain, HandoffSlot())

    assert text == "what do you want me to turn on?"
    assert [call.response_format for call in brain.calls] == [_SPOKEN_REPLY_FORMAT, _SPOKEN_REPLY_FORMAT]
    assert all(call.tools == _SCHEMA for call in brain.calls)


def test_the_spoken_reply_format_is_the_strict_two_field_object():
    spec = _SPOKEN_REPLY_FORMAT["json_schema"]
    assert _SPOKEN_REPLY_FORMAT["type"] == "json_schema"
    assert spec["name"] == "spoken_reply" and spec["strict"] is True
    assert spec["schema"]["required"] == ["answer", "expects_reply"]
    assert spec["schema"]["additionalProperties"] is False
    assert {k: v["type"] for k, v in spec["schema"]["properties"].items()} == {
        "answer": "string",
        "expects_reply": "boolean",
    }


async def test_a_final_round_question_sets_the_flag_on_the_handoff_slot():
    slot = HandoffSlot()
    text, _ = await _rounds(RecordingFakeBrain(replies=[BrainReply(text=_QUESTION)]), slot)

    assert text == "what do you want me to turn on?"
    assert slot.expects_reply is True


async def test_a_final_round_with_the_flag_false_leaves_the_slot_false():
    slot = HandoffSlot()
    reply = json.dumps({"answer": "Paris.", "expects_reply": False})
    text, _ = await _rounds(RecordingFakeBrain(replies=[BrainReply(text=reply)]), slot)

    assert text == "Paris."
    assert slot.expects_reply is False


# --- the parse and its fallback -------------------------------------------------


def test_plain_text_is_spoken_verbatim_with_one_warning(caplog):
    with caplog.at_level(logging.WARNING, logger="atlas.turn.controller"):
        assert _parse_spoken_reply("it is teatime", TurnTimings()) == ("it is teatime", False)
    assert [r.levelno for r in caplog.records if r.name == "atlas.turn.controller"] == [logging.WARNING]


@pytest.mark.parametrize("text", ['["a"]', '{"expects_reply": true}', '{"answer": 3, "expects_reply": true}', "   "])
def test_off_schema_output_is_spoken_raw_and_never_sets_the_flag(text):
    assert _parse_spoken_reply(text, TurnTimings()) == (text, False)


def test_a_flag_that_is_not_a_json_boolean_is_false():
    assert _parse_spoken_reply('{"answer": "hi", "expects_reply": "yes"}', TurnTimings()) == ("hi", False)
    assert _parse_spoken_reply('{"answer": "hi", "expects_reply": 1}', TurnTimings()) == ("hi", False)
    assert _parse_spoken_reply('{"answer": "hi"}', TurnTimings()) == ("hi", False)


def test_a_blank_json_answer_is_the_empty_reply_case():
    timings = TurnTimings()
    assert _parse_spoken_reply('{"answer": "  ", "expects_reply": true}', timings) == (_EMPTY_REPLY, False)
    assert timings.turn_outcome == "empty_reply"


# --- code-composed replies never set the flag -----------------------------------


async def _top_tier(brain: Any, tool_host: Any, slot: "HandoffSlot | None", **kwargs: Any) -> TierReply:
    tier = brain_race.TierBrain(index=0, model="", brain=brain, envelope_client=None, calls_tools=True)
    return await brain_race.run_top_tier(
        tier,
        tool_host,
        _SCHEMA,
        kwargs.pop("messages", [{"role": "user", "content": "turn on the"}]),
        kwargs.pop("max_tool_rounds", 3),
        TurnTimings(),
        handoff_slot=slot,
        **kwargs,
    )


async def test_the_round_zero_done_shortcut_leaves_the_flag_false():
    schema = [{"type": "function", "function": {"name": "ha_call_service"}}]
    call = ToolCall(name="ha_call_service", arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"})
    payload = {"changed": [{"entity_id": "switch.example_fan", "state": "off"}]}

    class _Changed:
        async def call_tool(self, name: str, arguments: dict) -> Any:
            return SimpleNamespace(
                isError=False, content=[SimpleNamespace(text=json.dumps(payload))], structured_content=payload
            )

    tier = brain_race.TierBrain(
        index=0, model="", brain=RecordingFakeBrain(replies=[BrainReply(tool_calls=[call])]), envelope_client=None, calls_tools=True
    )
    slot = HandoffSlot()
    reply = await brain_race.run_top_tier(
        tier,
        _Changed(),
        schema,
        [{"role": "user", "content": "turn off the fan"}],
        3,
        TurnTimings(),
        handoff_slot=slot,
    )

    assert reply.answer == controller._DONE_REPLY
    assert reply.expects_reply is False and slot.expects_reply is False


async def test_a_mixed_outcome_with_a_denied_call_leaves_the_flag_false():
    class _Mixed:
        async def call_tool(self, name: str, arguments: dict) -> Any:
            if name == "lights_set":
                return SimpleNamespace(isError=True, content=[SimpleNamespace(text="denied: off limits")])
            return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")])

    brain = RecordingFakeBrain(
        replies=[BrainReply(tool_calls=[_WEATHER, ToolCall(name="lights_set", arguments={"entity_id": "light.example_lamp"})])]
    )
    slot = HandoffSlot()
    reply = await _top_tier(brain, _Mixed(), slot)

    assert reply.expects_reply is False and slot.expects_reply is False


async def test_the_round_cap_reply_leaves_the_flag_false():
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_WEATHER])])
    slot = HandoffSlot()
    reply = await _top_tier(brain, _Host(), slot, max_tool_rounds=1)

    assert reply.answer == controller._TOO_MANY_ROUNDS_REPLY
    assert reply.expects_reply is False and slot.expects_reply is False


def test_the_flag_has_exactly_one_write_site_and_the_parse_one_caller():
    lines = Path(controller.__file__).read_text().splitlines()
    code = "\n".join(line for line in lines if not line.strip().startswith("#"))
    assert len(re.findall(r"handoff_slot\.expects_reply\s*=[^=]", code)) == 1
    assert code.count("_parse_spoken_reply(") == 2  # the definition and one call


# --- run_top_tier copies the flag -----------------------------------------------


async def test_run_top_tier_copies_the_flag_into_the_tier_reply():
    slot = HandoffSlot()
    reply = await _top_tier(RecordingFakeBrain(replies=[BrainReply(text=_QUESTION)]), _Host(), slot)

    assert reply.expects_reply is True and reply.confident is True


async def test_run_top_tier_gives_false_for_a_plain_final_round_and_for_no_slot():
    plain = await _top_tier(RecordingFakeBrain(replies=[BrainReply(text="Paris.")]), _Host(), HandoffSlot())
    no_slot = await _top_tier(RecordingFakeBrain(replies=[BrainReply(text=_QUESTION)]), _Host(), None)

    assert plain.expects_reply is False
    assert no_slot.expects_reply is False
    assert no_slot.answer == "what do you want me to turn on?"


# --- the TierReply field --------------------------------------------------------


def test_tier_reply_flag_defaults_to_false_and_parses_from_json():
    base = {"answer": "x", "confident": True, "needs_tool": False, "filler": "let_me_check"}
    reply = TierReply(answer="x", confident=True, needs_tool=False, filler=FillerPhrase.LET_ME_CHECK)
    assert reply.expects_reply is False
    assert reply.model_dump()["expects_reply"] is False
    assert TierReply.model_validate_json(json.dumps(base)).expects_reply is False
    assert TierReply.model_validate_json(json.dumps({**base, "expects_reply": True})).expects_reply is True
