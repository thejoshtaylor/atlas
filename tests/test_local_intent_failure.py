"""261001-ibf: a failed local on/off command is never silent.

Live evidence (2026-10-01): "turn on the swamp cooler." ended as
`local_intent_failed` after 5004 ms with no Home Assistant request and no log
line. 5004 ms is the per-plugin MCP deadline. These tests pin the parent side:

- every error-shaped local-intent result writes one WARNING line;
- a parent-side timeout still ends as `local_intent_failed`, with no second
  action, because it cannot prove that the request never reached Home Assistant;
- only the child's `HA_UNREACHABLE_REASON`, which the child raises when no
  request left the process, falls back to the brain.

Every name below is invented.
"""

from __future__ import annotations

import asyncio
import json
import logging
from types import SimpleNamespace
from typing import Any

from atlas_mcp.ha_names import HA_UNREACHABLE_REASON

from atlas.providers.base import BrainReply, FinalTranscript
from atlas.timing import TurnTimings
from atlas.turn.controller import run_turn

from brain_fakes import RecordingFakeBrain
from conftest import FakeAudioSource, FakeStt, FakeTts

_FILLER_CACHE = {None: {"done": b"done-bytes", "i can't do that one": b"cant-bytes"}}
_TIMEOUT_TEXT = "ha_call_service did not respond within 5000ms"


class _Host:
    def __init__(self, result: Any, *, delay_s: float = 0.0) -> None:
        self._result = result
        self._delay_s = delay_s
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, dict(arguments)))
        if self._delay_s:
            await asyncio.sleep(self._delay_s)
        return self._result


def _error(text: str, **extra: Any) -> SimpleNamespace:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=text)], **extra)


_OK = SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps({"changed": []}))])


async def _state_fetch() -> list[dict[str, Any]]:
    return [{"entity_id": "switch.example_fan", "friendly_name": "Example Fan", "state": "off"}]


async def _turn(host: _Host, brain: RecordingFakeBrain, **kwargs: Any) -> tuple[TurnTimings, FakeAudioSource, FakeTts]:
    source = FakeAudioSource(frames=[b"\x00\x01"])
    tts = FakeTts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    await run_turn(
        source,
        FakeStt(events=[FinalTranscript(text="turn off the example fan")]),
        brain,
        tts,
        host,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        state_fetch=_state_fetch,
        filler_cache=_FILLER_CACHE,
        local_intents=True,
        **kwargs,
    )
    return timings, source, tts


def _controller_warnings(caplog) -> list[logging.LogRecord]:
    return [
        record
        for record in caplog.records
        if record.name == "atlas.turn.controller" and record.levelno == logging.WARNING
    ]


async def test_a_parent_timeout_ends_as_local_intent_failed_and_is_logged(caplog):
    host = _Host(_error(_TIMEOUT_TEXT))
    brain = RecordingFakeBrain()

    with caplog.at_level(logging.WARNING, logger="atlas.turn.controller"):
        timings, source, tts = await _turn(host, brain)

    assert timings.turn_outcome == "local_intent_failed"
    assert source.sent_audio == [b"cant-bytes"]
    assert brain.call_count == 0
    assert len(host.calls) == 1  # no second action
    warnings = _controller_warnings(caplog)
    assert len(warnings) == 1
    message = warnings[0].getMessage()
    assert timings.turn_id in message
    assert "switch.turn_off" in message
    assert "switch.example_fan" in message
    assert _TIMEOUT_TEXT in message


async def test_a_wrapped_refusal_is_logged_with_the_wrapper_removed(caplog):
    host = _Host(_error("Error executing tool ha_call_service: that switch is off limits"))

    with caplog.at_level(logging.WARNING, logger="atlas.turn.controller"):
        timings, _source, _tts = await _turn(host, RecordingFakeBrain())

    assert timings.turn_outcome == "local_intent_failed"
    warnings = _controller_warnings(caplog)
    assert len(warnings) == 1
    assert "that switch is off limits" in warnings[0].getMessage()
    assert "Error executing tool" not in warnings[0].getMessage()


async def test_a_not_sent_result_logs_and_falls_back_to_the_brain(caplog):
    host = _Host(_error(f"Error executing tool ha_call_service: {HA_UNREACHABLE_REASON}"), delay_s=0.05)
    brain = RecordingFakeBrain(replies=[BrainReply(text="Done.")])
    fetched: list[bool] = []
    cancelled: list[bool] = []

    async def pending_runs_fetch() -> tuple[Any, ...]:
        try:
            await asyncio.sleep(0.01)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        fetched.append(True)
        return ()

    with caplog.at_level(logging.WARNING, logger="atlas.turn.controller"):
        timings, source, _tts = await _turn(host, brain, pending_runs_fetch=pending_runs_fetch)

    assert timings.turn_outcome == "completed"
    assert brain.call_count == 1
    assert b"cant-bytes" not in source.sent_audio
    # The pending-runs fetch is still read on the brain path.
    assert fetched == [True]
    assert cancelled == []
    messages = [record.getMessage() for record in _controller_warnings(caplog)]
    assert any("falls back to the brain" in message for message in messages)
    assert any(HA_UNREACHABLE_REASON in message for message in messages)


async def test_a_success_speaks_done_with_no_warning(caplog):
    host = _Host(_OK)
    brain = RecordingFakeBrain()

    with caplog.at_level(logging.WARNING, logger="atlas.turn.controller"):
        timings, source, _tts = await _turn(host, brain)

    assert timings.turn_outcome == "local_intent"
    assert source.sent_audio == [b"done-bytes"]
    assert brain.call_count == 0
    assert _controller_warnings(caplog) == []


async def test_a_claim_refusal_still_speaks_live_and_never_reaches_the_brain():
    host = _Host(_error("someone just changed the example fan", claim_refusal=True))
    brain = RecordingFakeBrain()

    timings, _source, tts = await _turn(host, brain)

    assert timings.turn_outcome == "claim_refused"
    assert tts.received_text == ["someone just changed the example fan"]
    assert brain.call_count == 0


async def test_a_raised_call_still_ends_as_local_intent_failed():
    class _Raising:
        async def call_tool(self, name: str, arguments: dict) -> Any:
            raise RuntimeError("boom")

    brain = RecordingFakeBrain()
    timings, source, _tts = await _turn(_Raising(), brain)  # type: ignore[arg-type]

    assert timings.turn_outcome == "local_intent_failed"
    assert source.sent_audio == [b"cant-bytes"]
    assert brain.call_count == 0


def test_an_unreachable_home_assistant_is_a_failed_workflow_step_not_a_denied_one():
    from atlas.workflow.steps import _is_policy_refusal

    wrapped = f"Error executing tool ha_call_service: {HA_UNREACHABLE_REASON}"

    assert _is_policy_refusal("ha_call_service", wrapped) is False
    assert _is_policy_refusal("ha_call_service", "Error executing tool ha_call_service: that switch is off limits")
