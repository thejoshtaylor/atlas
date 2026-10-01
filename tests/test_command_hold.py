"""261001-ibf: an unfinished command, and a bare wake phrase in an answer
window, are held on the same xAI socket.

Live edge turn (2026-10-01): the operator said "Turn off." and paused before
"the swamp cooler". The first half reached the brain alone. `WakeHold` now
holds a final that looks unfinished, and `_drain_to_final_transcript` waits a
short time for the rest. Every name below is invented.
"""

from __future__ import annotations

import asyncio
import json
import time
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.providers.base import BrainReply
from atlas.timing import TurnTimings
from atlas.providers.stt_xai import XaiStt
from atlas.transports.edge import SpeechSignals
from atlas.turn.follow_up import AnswerScope, FollowUpChannel, FollowUpRequest
from atlas.turn.wake_echo import WakeHold

from brain_fakes import RecordingFakeBrain
from test_early_finalize import _EventCapturingSource, _partial, _patch_connect, _ScriptedXaiSocket, _xai_cfg


# --- WakeHold alone ----------------------------------------------------------


def test_an_unfinished_command_is_held_and_the_continuation_joins_it():
    hold = WakeHold("hey atlas", verify=True, unfinished=True)

    assert hold("Hey Atlas, turn off.") is True
    assert hold.awaiting_continuation is True
    assert hold.fragments == ["Hey Atlas, turn off"]
    assert hold("the swamp cooler.") is False
    assert hold.awaiting_continuation is False
    assert hold.heard_text("the swamp cooler.") == "Hey Atlas, turn off the swamp cooler."


def test_a_hold_that_times_out_keeps_the_held_words():
    hold = WakeHold("hey atlas", verify=True, unfinished=True)
    hold("Hey Atlas, turn off.")

    assert hold.heard_text("") == "Hey Atlas, turn off"


def test_an_empty_final_while_awaiting_a_continuation_is_held_and_changes_nothing():
    hold = WakeHold("hey atlas", verify=True, unfinished=True)
    hold("Hey Atlas, turn off.")

    assert hold("") is True
    assert hold.awaiting_continuation is True
    assert hold.fragments == ["Hey Atlas, turn off"]


def test_a_dangling_continuation_keeps_holding():
    hold = WakeHold("hey atlas", verify=True, unfinished=True)
    hold("Hey Atlas, turn on the")

    assert hold("lamp in the") is True
    assert hold.heard_text("kitchen.") == "Hey Atlas, turn on the lamp in the kitchen."


def test_the_wake_then_unfinished_command_composes_after_the_held_wake_phrase():
    hold = WakeHold("hey atlas", verify=True, unfinished=True)

    assert hold("Hey Atlas.") is True
    assert hold("turn off.") is True
    assert hold.heard_text("the swamp cooler.") == "Hey Atlas. turn off the swamp cooler."


def test_with_needs_wake_a_text_without_the_phrase_is_never_held_as_unfinished():
    hold = WakeHold("hey atlas", verify=True, unfinished=True, unfinished_needs_wake=True)

    assert hold("Turn off.") is False
    assert hold.fragments == []


def test_without_needs_wake_an_answer_window_text_is_held():
    hold = WakeHold("hey atlas", verify=True, unfinished=True, unfinished_needs_wake=False)

    assert hold("Turn off.") is True
    assert hold.heard_text("the swamp cooler.") == "Turn off the swamp cooler."


@pytest.mark.parametrize(("gate", "expected"), [(False, False), (True, True)])
def test_the_wake_gate_decides_whether_a_bare_wake_phrase_is_held(gate, expected):
    hold = WakeHold("hey atlas", verify=True, unfinished=True, unfinished_needs_wake=False, wake_gate=lambda: gate)

    assert hold("Hey Atlas.") is expected


def test_a_hold_with_no_unfinished_flag_behaves_as_before():
    hold = WakeHold("hey atlas", verify=True)

    assert hold("Hey Atlas, turn off.") is False
    assert hold("Hey Atlas.") is True
    assert hold.heard_text("turn on the lights") == "Hey Atlas. turn on the lights"


# --- run_turn on one xAI socket ----------------------------------------------

_ENTITIES = [{"entity_id": "switch.example_swamp_cooler", "friendly_name": "Example Swamp Cooler", "state": "on"}]


class _ToolHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps({"ok": True}))])


async def _state_fetch() -> list[dict[str, Any]]:
    return _ENTITIES


async def _turn(
    source: Any,
    brain: RecordingFakeBrain,
    tts: Any,
    *,
    incoming: FollowUpRequest | None = None,
    wake_heard: bool = False,
    timeout_s: float = 4.0,
    **kwargs: Any,
) -> tuple[TurnTimings, _ToolHost]:
    from atlas.turn.controller import run_turn

    if incoming is not None:
        source.follow_up = FollowUpChannel(incoming=incoming, wake_heard=wake_heard)
    host = _ToolHost()
    timings = TurnTimings()
    await asyncio.wait_for(
        run_turn(
            source,
            XaiStt(_xai_cfg()),
            brain,
            tts,
            host,
            tools_schema=[{"type": "function", "function": {"name": "ha_call_service"}}],
            system_prompt="you control a home",
            max_tool_rounds=3,
            timings=timings,
            wake_phrase="hey atlas",
            verify_wake=True,
            state_fetch=_state_fetch,
            local_intents=True,
            **kwargs,
        ),
        timeout=timeout_s,
    )
    return timings, host


def _answer_window(**overrides: Any) -> FollowUpRequest:
    fields: dict[str, Any] = {
        "kind": "answer",
        "chain_depth": 1,
        "original_transcript": "what is the weather",
        "question": "It is sunny.",
        "answer_scope": AnswerScope(tool_names=frozenset({"ha_call_service"})),
        "playback_ends_at": 0.0,
    }
    fields.update(overrides)
    return FollowUpRequest(**fields)


async def test_a_wake_turn_joins_a_command_split_by_a_pause_on_one_socket(monkeypatch, fake_tts):
    socket = _ScriptedXaiSocket(
        [_partial("Hey Atlas, turn off.", final=True), _partial("the example swamp cooler.", final=True)]
    )
    connects = _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)

    timings, host = await _turn(source, RecordingFakeBrain(), fake_tts(chunks=[b"\x01"]))

    assert len(connects) == 1
    assert source.final_text() == "turn off the example swamp cooler."
    assert timings.turn_outcome == "local_intent"
    assert [name for name, _arguments in host.calls] == ["ha_call_service"]
    assert host.calls[0][1]["entity_id"] == "switch.example_swamp_cooler"


async def test_an_unfinished_command_with_nothing_after_it_ends_with_the_held_text(monkeypatch, fake_tts):
    monkeypatch.setattr("atlas.turn.controller._UNFINISHED_COMMAND_HOLD_S", 0.2)
    socket = _ScriptedXaiSocket([_partial("Hey Atlas, turn off.", final=True)])
    connects = _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)
    brain = RecordingFakeBrain()
    tts = fake_tts(chunks=[b"\x01"])

    started = time.monotonic()
    timings, host = await _turn(source, brain, tts)

    assert time.monotonic() - started < 2.0
    assert len(connects) == 1
    assert timings.turn_outcome == "missing_target"
    assert tts.received_text == ["turn off what?"]
    assert brain.call_count == 0
    assert host.calls == []


async def test_a_worded_partial_after_the_hold_cancels_the_timer(monkeypatch, fake_tts):
    monkeypatch.setattr("atlas.turn.controller._UNFINISHED_COMMAND_HOLD_S", 0.3)
    more = asyncio.Event()
    last = asyncio.Event()
    socket = _ScriptedXaiSocket(
        [
            _partial("Hey Atlas, turn off.", final=True),
            more,
            _partial("the", final=False),
            last,
            _partial("the example swamp cooler.", final=True),
        ]
    )
    _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)

    async def _speak_on() -> None:
        await asyncio.sleep(0.1)
        more.set()
        await asyncio.sleep(0.5)  # past the point where the timer would have fired
        last.set()

    asyncio.ensure_future(_speak_on())
    timings, host = await _turn(source, RecordingFakeBrain(), fake_tts(chunks=[b"\x01"]))

    assert source.final_text() == "turn off the example swamp cooler."
    assert timings.turn_outcome == "local_intent"


async def test_the_turn_budget_returns_the_held_words_not_a_timeout(monkeypatch, fake_tts):
    monkeypatch.setattr("atlas.turn.controller._UNFINISHED_COMMAND_HOLD_S", 60.0)
    socket = _ScriptedXaiSocket([_partial("Hey Atlas, turn off.", final=True)])
    _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)
    brain = RecordingFakeBrain()
    tts = fake_tts(chunks=[b"\x01"])

    timings, _host = await _turn(source, brain, tts, max_utterance_s=0.3, poll_interval_s=0.01)

    assert timings.turn_outcome == "missing_target"
    assert tts.received_text == ["turn off what?"]
    assert brain.call_count == 0


async def test_a_held_wake_phrase_with_nothing_after_it_still_times_out(monkeypatch, fake_tts):
    socket = _ScriptedXaiSocket([_partial("Hey Atlas.", final=True)])
    _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)

    timings, _host = await _turn(
        source, RecordingFakeBrain(), fake_tts(chunks=[b"\x01"]), max_utterance_s=0.3, poll_interval_s=0.01
    )

    assert timings.turn_outcome == "timeout"


async def test_a_continuation_that_xai_holds_until_finalized_is_finalized_in_an_answer_window(
    monkeypatch, fake_tts
):
    monkeypatch.setattr("atlas.turn.controller._UNFINISHED_COMMAND_HOLD_S", 0.5)
    monkeypatch.setattr("atlas.turn.controller._WORDLESS_SEGMENT_GRACE_S", 0.1)
    signals = SpeechSignals(hangover_s=0.0)
    first = asyncio.Event()
    socket = _ScriptedXaiSocket(
        [
            first,
            _partial("Turn off.", final=True),
            ("finalize", 1),
            _partial("", final=True),
            ("finalize", 2),
            _partial("the example swamp cooler.", final=True),
        ]
    )
    connects = _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(signals)
    brain = RecordingFakeBrain(replies=[BrainReply(text="Done.")])

    async def _speech() -> None:
        await asyncio.sleep(0.02)
        signals.publish({"type": "vad.start", "seq": 1})
        first.set()
        await asyncio.sleep(0.05)
        signals.publish({"type": "vad.end", "seq": 1})
        await asyncio.sleep(0.3)  # the pause, longer than the grace
        signals.publish({"type": "vad.start", "seq": 2})
        await asyncio.sleep(0.05)
        signals.publish({"type": "vad.end", "seq": 2})

    asyncio.ensure_future(_speech())
    await _turn(source, brain, fake_tts(chunks=[b"\x01"]), incoming=_answer_window())

    assert len(connects) == 1
    assert socket.finalize_count == 2
    assert brain.calls[0].messages[-1] == {"role": "user", "content": "Turn off the example swamp cooler."}


async def test_a_bare_wake_phrase_in_a_wake_heard_window_is_held_and_the_command_runs(monkeypatch, fake_tts):
    socket = _ScriptedXaiSocket([_partial("Hey Atlas.", final=True), _partial("turn off the example swamp cooler.", final=True)])
    connects = _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)

    timings, host = await _turn(
        source, RecordingFakeBrain(), fake_tts(chunks=[b"\x01"]), incoming=_answer_window(), wake_heard=True
    )

    assert len(connects) == 1
    assert timings.turn_outcome == "local_intent"
    assert host.calls[0][1]["entity_id"] == "switch.example_swamp_cooler"


async def test_a_bare_wake_phrase_with_no_wake_hit_holds_nothing(monkeypatch, fake_tts):
    socket = _ScriptedXaiSocket([_partial("Hey Atlas.", final=True)])
    _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)
    brain = RecordingFakeBrain()

    timings, _host = await _turn(
        source, brain, fake_tts(chunks=[b"\x01"]), incoming=_answer_window(), wake_heard=False, timeout_s=2.0
    )

    assert timings.turn_outcome == "no_command"
    assert brain.call_count == 0


async def test_an_unfinished_answer_in_a_window_is_held_and_joined(monkeypatch, fake_tts):
    socket = _ScriptedXaiSocket([_partial("Turn off.", final=True), _partial("the example swamp cooler.", final=True)])
    connects = _patch_connect(monkeypatch, socket)
    source = _EventCapturingSource(None)
    brain = RecordingFakeBrain(replies=[BrainReply(text="Done.")])

    await _turn(source, brain, fake_tts(chunks=[b"\x01"]), incoming=_answer_window())

    assert len(connects) == 1
    assert brain.calls[0].messages[-1] == {"role": "user", "content": "Turn off the example swamp cooler."}
