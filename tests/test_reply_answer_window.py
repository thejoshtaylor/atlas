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

import pytest

from atlas.config import GateConfig, WakeConfig
from atlas.providers.base import BrainReply
from atlas.providers.tier_reply import FillerPhrase, TierReply
from atlas.sources.runner import SourceRunner
from atlas.timing import TurnTimings
from atlas.turn import brain_race
from atlas.turn.controller import run_turn
from atlas.timers.ring_stop import is_stop_command
from atlas.turn.answer_window import build_answer_request, is_affirmative, silent_answer_outcome
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, AnswerScope, FollowUpChannel, FollowUpRequest

from brain_fakes import RecordingFakeBrain
from conftest import FakeAudioSource, FakeEnvelopeClient, FakeTts
from test_follow_up_answer_window import (
    _LIGHTS,
    _SCHEMA,
    _WEATHER,
    _AlwaysHitWakeDetector,
    _answer_request,
    _Edge,
    _RecordingToolHost,
    _text,
    _tool_names,
    _window_turn,
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
    edge = _Edge([_text("turn something on"), _text("the lamp"), _text("")], brain)

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
    turns = [_text("turn something on"), _text("")]

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


# --- the scope rules around the flag --------------------------------------------


async def test_a_no_tool_answer_with_the_flag_false_keeps_the_empty_scope():
    """13-CONTEXT D-11, unchanged."""
    brain = RecordingFakeBrain(replies=[_json("Paris.", False)])
    edge = _Edge([_text("capital of france"), _text("")], brain)

    await edge.run()

    request = edge.requests[0]
    assert request is not None
    assert request.answer_scope == AnswerScope(tool_names=frozenset())


async def test_a_plain_text_no_tool_answer_keeps_the_empty_scope():
    brain = RecordingFakeBrain(replies=[BrainReply(text="Paris.")])
    edge = _Edge([_text("capital of france"), _text("")], brain)

    await edge.run()

    request = edge.requests[0]
    assert request is not None
    assert request.answer_scope == AnswerScope(tool_names=frozenset())


async def test_a_question_after_a_tool_call_opens_the_full_offered_catalog():
    """A 2026-10-01 user decision amends 13-CONTEXT D-10: a question asked
    after a state read must still let the answer act."""
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_WEATHER]), _json("which place?", True)])
    edge = _Edge([_text("what is the weather"), _text("")], brain)

    await edge.run()

    request = edge.requests[0]
    assert request is not None
    assert request.answer_scope == AnswerScope(tool_names=_FULL, entity_ids=None)


async def test_a_turn_that_used_a_tool_and_asked_nothing_keeps_the_dispatched_scope():
    """13-CONTEXT D-10, unchanged without the flag."""
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_WEATHER]), _json("sunny.", False)])
    edge = _Edge([_text("what is the weather"), _text("")], brain)

    await edge.run()

    request = edge.requests[0]
    assert request is not None
    assert request.answer_scope == AnswerScope(tool_names=frozenset({"weather_now"}))


async def test_a_source_without_answer_windows_opens_no_window_for_a_question():
    brain = RecordingFakeBrain(replies=[_json("what do you want me to turn on?", True)])
    edge = _Edge([_text("turn something on")], brain, answer_windows=False)

    await edge.run()

    assert edge.requests == [None]
    assert len(edge.timings) == 1


async def _chained_turn(incoming: FollowUpRequest, brain: Any) -> FollowUpRequest | None:
    source = FakeAudioSource(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming)
    await run_turn(
        source,
        _text("and the other one"),
        brain,
        FakeTts(chunks=[b"\x01"]),
        _RecordingToolHost(),
        tools_schema=_SCHEMA,
        system_prompt="you answer questions",
        max_tool_rounds=3,
        timings=TurnTimings(),
        answer_windows=True,
    )
    return source.follow_up.requested


async def test_a_chained_question_never_widens_the_incoming_scope():
    incoming_scope = AnswerScope(
        tool_names=frozenset({"weather_now", "lights_set"}), entity_ids=frozenset({"light.example_lamp"})
    )
    brain = RecordingFakeBrain(replies=[_json("which one?", True)])

    requested = await _chained_turn(_answer_request(chain_depth=1, answer_scope=incoming_scope), brain)

    assert _tool_names(brain.calls[0]) == {"weather_now", "lights_set"}
    assert requested is not None and requested.kind == "answer"
    assert requested.chain_depth == 2
    assert requested.answer_scope == incoming_scope


async def test_a_question_at_the_chain_cap_opens_no_window():
    brain = RecordingFakeBrain(replies=[_json("which one?", True)])

    requested = await _chained_turn(_answer_request(chain_depth=MAX_CHAINED_FOLLOW_UPS), brain)

    assert requested is None


def _scope(**overrides: Any) -> AnswerScope | None:
    fields: dict[str, Any] = {
        "incoming": None,
        "final_text": "turn on the",
        "reply_text": "what do you want me to turn on?",
        "called_tools": frozenset(),
        "answer_scope": None,
        "proposals_only": False,
        "prior_exchange": None,
        "playback_ends_at": None,
        "answer_only_from": None,
        "expects_reply": True,
        "offered_tools": frozenset({"a", "b"}),
    }
    fields.update(overrides)
    request = build_answer_request(**fields)
    return request.answer_scope if request is not None else None


def test_no_tool_and_a_question_uses_the_offered_catalog():
    assert _scope() == AnswerScope(tool_names=frozenset({"a", "b"}))


def test_a_question_widens_a_dispatched_scope_to_the_offered_catalog():
    assert _scope(called_tools=frozenset({"a"}), called_entities=frozenset({"light.example_lamp"})) == AnswerScope(
        tool_names=frozenset({"a", "b"})
    )


def test_a_dispatched_tool_without_a_question_keeps_its_scope():
    assert _scope(called_tools=frozenset({"a"}), expects_reply=False) == AnswerScope(tool_names=frozenset({"a"}))


def test_no_question_keeps_the_empty_scope_even_with_an_offered_catalog():
    assert _scope(expects_reply=False) == AnswerScope(tool_names=frozenset())


def test_the_incoming_scope_narrows_the_offered_catalog():
    assert _scope(answer_scope=AnswerScope(tool_names=frozenset({"a"}))) == AnswerScope(tool_names=frozenset({"a"}))


# --- "yeah" and "okay" after a model question (261001-dlp-04) -------------------


def test_the_request_records_that_its_reply_asked_a_question():
    request = build_answer_request(
        incoming=None,
        final_text="turn on the",
        reply_text="what do you want me to turn on?",
        called_tools=frozenset(),
        answer_scope=None,
        proposals_only=False,
        prior_exchange=None,
        playback_ends_at=None,
        answer_only_from=None,
        expects_reply=True,
        offered_tools=frozenset({"a"}),
    )
    assert request is not None and request.expects_reply is True
    assert _answer_request().expects_reply is False


@pytest.mark.parametrize("text", ["yes", "yeah", "Yep.", "yup", "ok", "Okay.", "sure", "yeah sure"])
def test_a_whole_utterance_affirmative_is_not_silent_after_a_question(text):
    assert is_affirmative(text)
    assert silent_answer_outcome(object(), text, "hey atlas", expects_reply=True) is None


@pytest.mark.parametrize("text", ["stop", "never mind", "be quiet", "that's enough", "thanks", "cancel"])
def test_other_stop_phrases_still_end_the_window_after_a_question(text):
    assert silent_answer_outcome(object(), text, "hey atlas", expects_reply=True) == "stopped"


@pytest.mark.parametrize(("text", "expected"), [("okay", "stopped"), ("Okay.", "stopped"), ("yeah", "no_command")])
def test_an_affirmative_after_a_plain_answer_ends_the_window_silently(text, expected):
    assert silent_answer_outcome(object(), text, "hey atlas") == expected
    assert silent_answer_outcome(object(), text, "hey atlas", expects_reply=False) == expected


def test_an_affirmative_with_other_words_is_not_whole_utterance():
    assert not is_affirmative("yeah turn it off")
    assert not is_affirmative("")


def test_the_ring_stop_still_takes_okay():
    assert is_stop_command("okay") and is_stop_command("Ok.")


@pytest.mark.parametrize("transcript", ["yeah", "ok", "Okay."])
async def test_an_affirmative_after_a_question_reaches_the_brain(transcript):
    brain = RecordingFakeBrain(replies=[BrainReply(text="Got it.")])

    timings, tts, _source, _host = await _window_turn(
        transcript, brain, incoming=_answer_request(expects_reply=True)
    )

    assert brain.call_count == 1
    assert timings.turn_outcome == "completed"
    assert tts.received_text == ["Got it."]
    assert brain.calls[0].messages[-1] == {"role": "user", "content": transcript}


@pytest.mark.parametrize("transcript", ["yeah", "ok", "Okay."])
async def test_an_affirmative_after_a_plain_answer_ends_silently(transcript):
    brain = RecordingFakeBrain()

    timings, tts, _source, _host = await _window_turn(transcript, brain, incoming=_answer_request())

    assert brain.call_count == 0
    assert timings.turn_outcome in {"stopped", "no_command"}
    assert tts.received_text == []


@pytest.mark.parametrize("transcript", ["stop", "never mind"])
async def test_a_stop_phrase_after_a_question_still_ends_silently(transcript):
    brain = RecordingFakeBrain()

    timings, tts, _source, _host = await _window_turn(
        transcript, brain, incoming=_answer_request(expects_reply=True)
    )

    assert brain.call_count == 0
    assert timings.turn_outcome == "stopped"
    assert tts.received_text == []


async def test_a_spoken_yeah_answers_a_model_question_through_the_edge():
    brain = RecordingFakeBrain(replies=[_json("do you want the lamp on?", True), BrainReply(text="Sure thing.")])
    edge = _Edge([_text("turn on the light"), _text("yeah"), _text("")], brain)

    await edge.run()

    assert brain.call_count == 2
    assert edge.tts[1].received_text == ["Sure thing."]
