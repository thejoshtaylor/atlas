"""261001-dlp: a question the model asks opens a window that can answer it.

On the edge source, a turn that dispatched no tool and ended on a model
question ("what do you want me to turn on?") leaves an answer window scoped to
the full catalog that turn offered. The operator then says "the lamp" with no
wake word, and the window turn can call the lamp tool. Every other case keeps
the Phase 13 scope. Every name and entity id here is invented.
"""

from __future__ import annotations

import json
from typing import Any

from atlas.config import GateConfig, WakeConfig
from atlas.providers.base import BrainReply
from atlas.providers.tier_reply import FillerPhrase, TierReply
from atlas.sources.runner import SourceRunner
from atlas.timing import TurnTimings
from atlas.turn import brain_race
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import AnswerScope, FollowUpRequest

from brain_fakes import RecordingFakeBrain
from conftest import FakeAudioSource, FakeEnvelopeClient, FakeTts
from test_follow_up_answer_window import (
    _LIGHTS,
    _SCHEMA,
    _AlwaysHitWakeDetector,
    _Edge,
    _RecordingToolHost,
    _text,
    _tool_names,
)

_FULL = frozenset({"weather_now", "lights_set", "gmail_fetch_body"})


def _json(answer: str, expects_reply: bool) -> BrainReply:
    return BrainReply(text=json.dumps({"answer": answer, "expects_reply": expects_reply}))


async def test_a_no_tool_question_opens_a_window_that_reaches_the_full_offered_catalog():
    brain = RecordingFakeBrain(
        replies=[
            _json("what do you want me to turn on?", True),
            BrainReply(tool_calls=[_LIGHTS]),
            BrainReply(text="Done."),
        ]
    )
    edge = _Edge([_text("turn on the"), _text("the lamp"), _text("")], brain)

    await edge.run()

    first = edge.requests[0]
    assert first is not None and first.kind == "answer"
    assert first.answer_scope == AnswerScope(tool_names=_FULL, entity_ids=None)
    assert edge.tts[0].received_text[-1] == "what do you want me to turn on?"
    # The window turn was offered all three tools, and its lamp call reached the host.
    assert _tool_names(brain.calls[1]) == _FULL
    assert [name for name, _ in edge.tool_host.calls] == ["lights_set"]
    assert [incoming.kind if incoming else None for incoming in edge.incomings] == [None, "answer", "answer"]


async def test_a_confident_triage_question_opens_the_same_full_catalog_window():
    """A triage tier's confident reply with `expects_reply` counts like a top-tier one."""
    question = TierReply(
        answer="what do you want me to turn on?",
        confident=True,
        needs_tool=False,
        filler=FillerPhrase.LET_ME_CHECK,
        expects_reply=True,
    )
    tier = brain_race.TierBrain(
        index=0,
        model="triage-model",
        brain=None,
        envelope_client=FakeEnvelopeClient(reply=question),
        calls_tools=False,
    )
    source = FakeAudioSource(frames=[b"\x00"])
    requests: list[FollowUpRequest | None] = []
    turns = [_text("turn on the"), _text("")]

    async def run_turn_fn(turn_source: Any) -> None:
        channel = getattr(turn_source, "follow_up", None)
        await run_turn(
            turn_source,
            turns[len(requests)],
            None,
            FakeTts(chunks=[b"\x01\x02"]),
            _RecordingToolHost(),
            tools_schema=_SCHEMA,
            system_prompt="you answer questions",
            max_tool_rounds=3,
            timings=TurnTimings(),
            tiers=[tier],
            answer_windows=True,
        )
        requests.append(channel.requested if channel is not None else None)

    runner = SourceRunner(
        "edge",
        source,
        _AlwaysHitWakeDetector(),
        lambda chunk: chunk,
        run_turn_fn,
        wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
        gate_config=GateConfig(),
        follow_up_window_s=lambda: 0.3,
        follow_up_echo_tail_s=0.0,
    )
    await runner.run()

    first = requests[0]
    assert first is not None and first.kind == "answer"
    assert first.answer_scope == AnswerScope(tool_names=_FULL, entity_ids=None)
