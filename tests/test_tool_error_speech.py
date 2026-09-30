"""Quick task 260930-e3r (D-03, D-14): a raw tool error is never spoken.

The live error was a pydantic `ValidationError` from `schedule_workflow`. It
was spoken whole, about 37 s of TTS. These tests prove three things. The
speech gate turns such text into a short apology and logs the full text. An
argument error goes back to the model as one short line and the model gets
another round. A policy refusal is still spoken verbatim with no extra round.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.timing import TurnTimings
from atlas.turn.controller import _TOOL_ERROR_REPLY, _spoken_error_text, run_turn
from atlas.turn.tool_errors import (
    ARGUMENT_ERROR_PREFIX,
    condense_argument_error,
    is_argument_error,
    is_speakable_error,
)


def _error(text: str) -> SimpleNamespace:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=text)])


def _live_error_text() -> str:
    """The text a `schedule_workflow` call with a `wait` step and no time field raised."""
    from atlas.workflow.tool import ScheduleWorkflowRequest

    with pytest.raises(Exception) as caught:
        ScheduleWorkflowRequest.model_validate(
            {"steps": [{"kind": "wait", "arguments": {"duration_s": 5}}], "summary": "wait five seconds"}
        )
    return str(caught.value)


def _timer_error_text() -> str:
    from atlas.timers.core import TimerSpec

    with pytest.raises(Exception) as caught:
        TimerSpec.model_validate({"duration_seconds": 0, "label": "pa\nsta"})
    return str(caught.value)


# --- the speech gate -----------------------------------------------------------


def test_the_live_validation_error_is_spoken_as_an_apology_and_logged_in_full(caplog):
    text = _live_error_text()
    assert "validation error" in text

    with caplog.at_level(logging.WARNING, logger="atlas.turn.controller"):
        spoken = _spoken_error_text("schedule_workflow", _error(text))

    assert spoken == "sorry, something went wrong with that"
    assert spoken == _TOOL_ERROR_REPLY
    assert any(text in record.getMessage() for record in caplog.records if record.levelno == logging.WARNING)


def test_an_sdk_wrapped_validation_error_is_an_argument_error_and_is_apologised():
    text = "Error executing tool set_timer: " + _timer_error_text()

    assert _spoken_error_text("set_timer", _error(text)) == _TOOL_ERROR_REPLY


def test_speakable_text_is_unchanged():
    assert _spoken_error_text("ha_call_service", _error("that one is off limits")) == "that one is off limits"
    assert _spoken_error_text("ha_call_service", _error("")) == ""
    assert _spoken_error_text("ha_call_service", _error("Error executing tool ha_call_service")) == ""


@pytest.mark.parametrize(
    "text",
    [
        "x" * 301,
        "first line\nsecond line",
        "see https://example.invalid/docs for more",
        "Traceback (most recent call last)",
    ],
)
def test_unfit_text_becomes_the_apology(text):
    assert not is_speakable_error(text)
    assert _spoken_error_text("some_tool", _error(text)) == _TOOL_ERROR_REPLY


def test_a_300_character_single_line_is_still_speakable():
    assert is_speakable_error("x" * 300)
    assert is_speakable_error("")


# --- the condensed message for the model ---------------------------------------


def test_the_live_error_condenses_to_one_short_line_the_model_can_act_on():
    condensed = condense_argument_error(_live_error_text())

    assert condensed.startswith(ARGUMENT_ERROR_PREFIX)
    assert "\n" not in condensed
    assert "https://" not in condensed
    assert "input_value" not in condensed
    assert "[type=" not in condensed
    assert "delay_seconds or at" in condensed
    assert len(condensed) <= 500
    assert is_argument_error(condensed)


def test_a_field_level_error_names_each_field_next_to_its_message():
    condensed = condense_argument_error(_timer_error_text())

    assert condensed.startswith(ARGUMENT_ERROR_PREFIX)
    assert "duration_seconds: Input should be greater than or equal to 1" in condensed
    assert "label: a label may not contain a line break or a tab" in condensed
    assert "\n" not in condensed


def test_condense_is_idempotent_and_capped():
    once = condense_argument_error("invalid arguments: " + "y" * 900)

    assert len(once) <= 500
    assert once.endswith("...")
    assert condense_argument_error(once) == once


# --- the retry round -----------------------------------------------------------


class _RecordingBrain:
    def __init__(self, replies: list[BrainReply]) -> None:
        self._replies = list(replies)
        self.received_messages: list[list[dict[str, Any]]] = []
        self.call_count = 0

    async def chat(self, messages, tools=None) -> BrainReply:
        self.received_messages.append([dict(m) for m in messages])
        self.call_count += 1
        return self._replies.pop(0)


class _Host:
    def __init__(self, result: Any) -> None:
        self._result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, dict(arguments)))
        return self._result


def _bad_call() -> ToolCall:
    return ToolCall(name="schedule_workflow", arguments={"steps": [{"kind": "wait"}]})


async def _run(brain, host, fake_audio_source, fake_stt, fake_tts, *, rounds=3):
    tts = fake_tts(chunks=[b"\x01\x02"])
    await run_turn(
        fake_audio_source(frames=[b"\x00\x01"]),
        fake_stt(events=[FinalTranscript(text="remind me later")]),
        brain,
        tts,
        host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=rounds,
        timings=TurnTimings(),
    )
    return tts


async def test_an_argument_error_goes_back_to_the_model_and_the_next_round_is_spoken(
    fake_audio_source, fake_stt, fake_tts
):
    brain = _RecordingBrain(
        [BrainReply(tool_calls=[_bad_call()]), BrainReply(text="your timer is set")]
    )
    host = _Host(_error("Error executing tool schedule_workflow: " + _live_error_text()))

    tts = await _run(brain, host, fake_audio_source, fake_stt, fake_tts)

    assert brain.call_count == 2
    last = brain.received_messages[1][-1]
    assert last["role"] == "tool"
    assert last["content"].startswith("invalid arguments: ")
    assert "https://" not in last["content"]
    assert tts.received_text[-1] == "your timer is set"


async def test_the_retry_is_bounded_and_the_final_reply_is_the_apology(fake_audio_source, fake_stt, fake_tts):
    brain = _RecordingBrain([BrainReply(tool_calls=[_bad_call()]) for _ in range(3)])
    host = _Host(_error("Error executing tool schedule_workflow: " + _live_error_text()))

    tts = await _run(brain, host, fake_audio_source, fake_stt, fake_tts, rounds=3)

    assert brain.call_count == 3
    assert tts.received_text[-1] == _TOOL_ERROR_REPLY
    spoken = " ".join(tts.received_text).lower()
    assert "validation" not in spoken
    assert "pydantic" not in spoken


async def test_a_refusal_is_spoken_verbatim_with_no_second_model_round(fake_audio_source, fake_stt, fake_tts):
    brain = _RecordingBrain([BrainReply(tool_calls=[_bad_call()]), BrainReply(text="should never be asked")])
    host = _Host(_error("that one is off limits"))

    tts = await _run(brain, host, fake_audio_source, fake_stt, fake_tts)

    assert brain.call_count == 1
    assert tts.received_text[-1] == "that one is off limits"
