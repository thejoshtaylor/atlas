"""Plan 13-04: after an ordinary answer on the edge source, the microphone
stays open with no wake word (VOICE-21, D-09 to D-15).

The window fails closed. Every word heard in it is untrusted, because a
television can say any sentence. So the window turn may reach only the tools
the answered turn dispatched, and none after a chat-only answer.

Three layers:

- `answer_window.py`: pure unit tests.
- `run_turn` with a hand-built `FollowUpChannel(incoming=...)`: the window
  turn alone (stop phrases, filler, macros, chains).
- A real `SourceRunner` with the real `run_turn` and `answer_windows=True`:
  the whole flow, the way the edge runs it.

Tool names are generic and no house data appears here.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.config import GateConfig, MacroActionConfig, MacroConfig, WakeConfig
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.sources.runner import SourceRunner
from atlas.timing import TurnTimings
from atlas.turn.answer_window import (
    ANSWER_WINDOW_SILENT,
    answer_speaker_mismatch,
    answer_turn_scope,
    build_answer_request,
    silent_answer_outcome,
)
from atlas.turn.controller import run_turn
from atlas.turn.follow_up import MAX_CHAINED_FOLLOW_UPS, AnswerScope, FollowUpChannel, FollowUpRequest
from atlas.turn.handoff import AMENDED_CONTINUATION_REFUSAL

from brain_fakes import RecordingFakeBrain
from conftest import FakeAudioSource, FakeStt, FakeTts

_SCHEMA = [
    {"type": "function", "function": {"name": "weather_now"}},
    {"type": "function", "function": {"name": "lights_set"}},
    {"type": "function", "function": {"name": "gmail_fetch_body"}},
]

_WEATHER = ToolCall(name="weather_now", arguments={"place": "home"})
_LIGHTS = ToolCall(name="lights_set", arguments={"entity_id": "light.example_lamp", "on": True})


class _RecordingToolHost:
    def __init__(self) -> None:
        self.calls: list[tuple[str, dict]] = []

    async def call_tool(self, name: str, arguments: dict) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text=json.dumps({"ok": True}))])


class _AlwaysHitWakeDetector:
    def process(self, chunk: bytes) -> Any:
        from conftest import FakeWakeHit

        return FakeWakeHit(score=1.0)

    def close(self) -> None:
        pass


def _text(text: str) -> FakeStt:
    return FakeStt(events=[FinalTranscript(text=text)])


def _answer_request(**overrides: Any) -> FollowUpRequest:
    fields: dict[str, Any] = {
        "kind": "answer",
        "chain_depth": 1,
        "original_transcript": "what is the weather",
        "question": "It is sunny.",
        "answer_scope": AnswerScope(tool_names=frozenset({"weather_now"})),
        "playback_ends_at": 0.0,
    }
    fields.update(overrides)
    return FollowUpRequest(**fields)


class _Edge:
    """One real `SourceRunner` whose turn function runs the real `run_turn`
    with the edge's `answer_windows` flag. `stts` scripts each turn's own
    transcript, in order; `requests` and `incomings` record what each turn
    left and what each turn answered."""

    def __init__(
        self,
        stts: list[FakeStt],
        brain: Any,
        *,
        answer_windows: bool = True,
        macros: tuple[MacroConfig, ...] = (),
        window_s: float = 0.3,
    ) -> None:
        self.source = FakeAudioSource(frames=[b"\x00"])
        self.tool_host = _RecordingToolHost()
        self.brain = brain
        self.timings: list[TurnTimings] = []
        self.incomings: list[FollowUpRequest | None] = []
        self.requests: list[FollowUpRequest | None] = []
        self.tts: list[FakeTts] = []

        async def run_turn_fn(turn_source: Any) -> None:
            index = len(self.timings)
            channel = getattr(turn_source, "follow_up", None)
            self.incomings.append(channel.incoming if channel is not None else None)
            timings = TurnTimings()
            self.timings.append(timings)
            tts = FakeTts(chunks=[b"\x01\x02"])
            self.tts.append(tts)
            await run_turn(
                turn_source,
                stts[index],
                brain,
                tts,
                self.tool_host,
                tools_schema=_SCHEMA,
                system_prompt="you answer questions",
                max_tool_rounds=3,
                timings=timings,
                macros=macros,
                answer_windows=answer_windows,
            )
            self.requests.append(channel.requested if channel is not None else None)

        self.runner = SourceRunner(
            "edge",
            self.source,
            _AlwaysHitWakeDetector(),
            lambda chunk: chunk,
            run_turn_fn,
            wake_config=WakeConfig(engine="vosk", refractory_s=0.0),
            gate_config=GateConfig(),
            follow_up_window_s=lambda: window_s,
            follow_up_echo_tail_s=0.0,
        )

    async def run(self) -> None:
        await self.runner.run()


def _tool_names(call: Any) -> set[str]:
    return {entry["function"]["name"] for entry in (call.tools or [])}


# --- the pure rules ------------------------------------------------------------


def test_build_answer_request_carries_the_exchange_and_the_scope():
    request = build_answer_request(
        incoming=None,
        final_text="what is the weather",
        reply_text="It is sunny.",
        called_tools={"weather_now"},
        answer_scope=None,
        proposals_only=True,
        prior_exchange=[{"role": "user", "content": "earlier"}],
        playback_ends_at=12.5,
        asked_by_speaker="1",
    )
    assert request == FollowUpRequest(
        kind="answer",
        chain_depth=1,
        original_transcript="what is the weather",
        question="It is sunny.",
        prior_messages=({"role": "user", "content": "earlier"},),
        playback_ends_at=12.5,
        proposals_only=True,
        answer_scope=AnswerScope(tool_names=frozenset({"weather_now"})),
        asked_by_speaker="1",
    )


def test_build_answer_request_narrows_by_the_scope_it_ran_under_and_never_widens():
    request = build_answer_request(
        incoming=_answer_request(chain_depth=2),
        final_text="and tomorrow",
        reply_text="Rain.",
        called_tools={"weather_now", "lights_set"},
        answer_scope=AnswerScope(tool_names=frozenset({"weather_now"})),
        proposals_only=False,
        prior_exchange=None,
        playback_ends_at=None,
        asked_by_speaker=None,
    )
    assert request is not None
    assert request.chain_depth == 3
    assert request.answer_scope == AnswerScope(tool_names=frozenset({"weather_now"}))


def test_build_answer_request_opens_no_window_past_the_chain_cap_or_after_a_blank_reply():
    common: dict[str, Any] = {
        "final_text": "x",
        "called_tools": set(),
        "answer_scope": None,
        "proposals_only": False,
        "prior_exchange": None,
        "playback_ends_at": None,
        "asked_by_speaker": None,
    }
    capped = build_answer_request(
        incoming=_answer_request(chain_depth=MAX_CHAINED_FOLLOW_UPS), reply_text="Fine.", **common
    )
    blank = build_answer_request(incoming=None, reply_text="  ", **common)
    assert capped is None
    assert blank is None


def test_an_answer_request_with_no_recorded_scope_offers_no_tool():
    assert answer_turn_scope(_answer_request(answer_scope=None)) == AnswerScope(tool_names=frozenset())


@pytest.mark.parametrize(
    ("final", "text", "expected"),
    [
        (None, "", ANSWER_WINDOW_SILENT),
        (object(), "   ", ANSWER_WINDOW_SILENT),
        (object(), "never mind", "stopped"),
        (object(), "okay", "stopped"),
        (object(), "um", "no_command"),
        (object(), "turn it off", None),
        (object(), "and tomorrow", None),
    ],
)
def test_silent_answer_outcome(final, text, expected):
    assert silent_answer_outcome(final, text, "hey atlas") == expected


def test_the_asker_only_rule_applies_in_enforce_mode_only():
    request = _answer_request(asked_by_speaker="1")
    assert answer_speaker_mismatch(request, speaker_event={"speaker_id": 2}, effective_mode="enforce")
    assert answer_speaker_mismatch(request, speaker_event={}, effective_mode="enforce")
    assert not answer_speaker_mismatch(request, speaker_event={"speaker_id": 1}, effective_mode="enforce")
    assert not answer_speaker_mismatch(request, speaker_event={"speaker_id": 2}, effective_mode="record")
    assert not answer_speaker_mismatch(_answer_request(), speaker_event={"speaker_id": 2}, effective_mode="enforce")


# --- the whole flow, through a real SourceRunner -----------------------------


async def test_a_window_opens_after_an_answer_and_is_scoped_to_the_tools_it_used():
    brain = RecordingFakeBrain(
        replies=[
            BrainReply(tool_calls=[_WEATHER]),
            BrainReply(text="It is sunny."),
            # The window turn tries a light as well as the weather. The
            # mixed result settles the reply without another model call.
            BrainReply(tool_calls=[_WEATHER, _LIGHTS]),
        ]
    )
    edge = _Edge([_text("what is the weather"), _text("and tomorrow"), _text("")], brain)

    await edge.run()

    assert [incoming.kind if incoming else None for incoming in edge.incomings] == [None, "answer", "answer"]
    first_request = edge.requests[0]
    assert first_request is not None and first_request.kind == "answer"
    assert first_request.answer_scope == AnswerScope(tool_names=frozenset({"weather_now"}))
    # The window turn was offered only the weather tool, and the light call never reached the host.
    window_call = brain.calls[2]
    assert _tool_names(window_call) == {"weather_now"}
    assert [name for name, _ in edge.tool_host.calls] == ["weather_now", "weather_now"]
    # The previous exchange came first, so "and tomorrow" has its meaning.
    assert window_call.messages[-3:] == [
        {"role": "user", "content": "what is the weather"},
        {"role": "assistant", "content": "It is sunny."},
        {"role": "user", "content": "and tomorrow"},
    ]
    assert AMENDED_CONTINUATION_REFUSAL in edge.tts[1].received_text[-1]


async def test_a_conversation_only_answer_opens_a_window_that_offers_no_tool():
    brain = RecordingFakeBrain(replies=[BrainReply(text="Paris."), BrainReply(tool_calls=[_LIGHTS]), BrainReply(text="No.")])
    edge = _Edge([_text("capital of france"), _text("turn on the lamp"), _text("")], brain)

    await edge.run()

    request = edge.requests[0]
    assert request is not None
    assert request.answer_scope == AnswerScope(tool_names=frozenset())
    assert not _tool_names(brain.calls[1])
    assert edge.tool_host.calls == []


async def test_a_source_without_the_flag_runs_exactly_one_turn():
    """The camera and browser shape: `answer_windows` stays False."""
    brain = RecordingFakeBrain(replies=[BrainReply(text="Paris.")])
    edge = _Edge([_text("capital of france")], brain, answer_windows=False)

    await edge.run()

    assert len(edge.timings) == 1
    assert edge.requests == [None]


async def test_a_window_that_hears_nothing_speaks_nothing():
    brain = RecordingFakeBrain(replies=[BrainReply(text="Paris.")])
    edge = _Edge([_text("capital of france"), _text("")], brain)

    await edge.run()

    assert edge.timings[1].turn_outcome == ANSWER_WINDOW_SILENT
    assert edge.tts[1].received_text == []
    assert len(edge.source.sent_audio) == 1  # the reply's own chunk, none from the window
    assert brain.call_count == 1


async def test_a_window_with_no_speech_at_all_times_out_silently():
    brain = RecordingFakeBrain(replies=[BrainReply(text="Paris.")])
    edge = _Edge([_text("capital of france"), FakeStt(hang=True)], brain, window_s=0.05)

    await edge.run()

    assert edge.timings[1].turn_outcome == ANSWER_WINDOW_SILENT
    assert edge.tts[1].received_text == []


async def test_a_refused_call_and_a_code_only_call_never_widen_the_next_scope():
    brain = RecordingFakeBrain(
        replies=[
            # Turn 1 asks for a code-only tool and a real one. The mixed
            # result settles the reply without another model call.
            BrainReply(tool_calls=[ToolCall(name="gmail_fetch_body", arguments={}), _WEATHER]),
            # The window turn asks for a tool outside its scope as well.
            BrainReply(tool_calls=[_WEATHER, _LIGHTS]),
        ]
    )
    edge = _Edge([_text("weather"), _text("and tomorrow"), _text("")], brain)

    await edge.run()

    assert edge.requests[0].answer_scope == AnswerScope(tool_names=frozenset({"weather_now"}))
    assert edge.requests[1] is not None
    assert edge.requests[1].answer_scope == AnswerScope(tool_names=frozenset({"weather_now"}))


async def test_a_chain_only_narrows_and_stops_at_the_cap():
    brain = RecordingFakeBrain(
        replies=[
            BrainReply(tool_calls=[_WEATHER, _LIGHTS]),
            BrainReply(text="One."),
            BrainReply(tool_calls=[_WEATHER]),
            BrainReply(text="Two."),
            BrainReply(text="Three."),
            BrainReply(text="Four."),
        ]
    )
    edge = _Edge([_text("first"), _text("second"), _text("third"), _text("fourth")], brain)

    await edge.run()

    scopes = [request.answer_scope.tool_names for request in edge.requests if request is not None]
    assert scopes == [frozenset({"weather_now", "lights_set"}), frozenset({"weather_now"}), frozenset()]
    assert all(later <= earlier for earlier, later in zip(scopes, scopes[1:]))
    # Three windows open (four turns in all), and the last window's turn
    # requests no fourth.
    assert [request.chain_depth for request in edge.requests if request is not None] == [1, 2, 3]
    assert len(edge.timings) == 4
    assert edge.requests[3] is None


# --- the window turn alone ---------------------------------------------------


async def _window_turn(
    transcript: str, brain: Any, *, incoming: FollowUpRequest | None = None, macros: tuple[MacroConfig, ...] = ()
) -> tuple[TurnTimings, FakeTts, FakeAudioSource, _RecordingToolHost]:
    source = FakeAudioSource(frames=[b"\x00\x01"])
    source.follow_up = FollowUpChannel(incoming=incoming or _answer_request())
    tool_host = _RecordingToolHost()
    tts = FakeTts(chunks=[b"\x01"])
    timings = TurnTimings()
    await run_turn(
        source,
        _text(transcript),
        brain,
        tts,
        tool_host,
        tools_schema=_SCHEMA,
        system_prompt="you answer questions",
        max_tool_rounds=3,
        timings=timings,
        macros=macros,
        local_intents=True,
    )
    return timings, tts, source, tool_host


@pytest.mark.parametrize("transcript", ["never mind", "Never mind.", "be quiet", "that's enough", "cancel", "stop"])
async def test_a_stop_phrase_ends_the_window_silently(transcript):
    brain = RecordingFakeBrain()
    timings, tts, source, _host = await _window_turn(transcript, brain)

    assert timings.turn_outcome == "stopped"
    assert tts.received_text == []
    assert source.sent_audio == []
    assert brain.call_count == 0


async def test_a_filler_word_ends_the_window_as_no_command():
    brain = RecordingFakeBrain()
    timings, tts, _source, _host = await _window_turn("um", brain)

    assert timings.turn_outcome == "no_command"
    assert tts.received_text == []
    assert brain.call_count == 0


async def test_turn_it_off_in_a_window_is_a_command_not_a_stop():
    brain = RecordingFakeBrain(replies=[BrainReply(tool_calls=[_WEATHER]), BrainReply(text="Done.")])
    timings, tts, _source, host = await _window_turn("turn it off", brain)

    assert timings.turn_outcome == "completed"
    assert brain.call_count == 2
    assert tts.received_text


async def test_a_window_never_runs_a_macro_whose_phrase_it_hears():
    macro = MacroConfig(
        phrase="good night",
        reply="good night",
        actions=(MacroActionConfig(tool="lights_set", arguments={"entity_id": "light.example_lamp"}),),
    )
    brain = RecordingFakeBrain(replies=[BrainReply(text="Sleep well.")])
    timings, _tts, _source, host = await _window_turn("good night", brain, macros=(macro,))

    assert host.calls == []
    assert brain.call_count == 1
    assert timings.turn_outcome == "completed"


async def test_a_window_with_a_real_request_runs_the_brain_with_the_previous_exchange_first():
    brain = RecordingFakeBrain(replies=[BrainReply(text="Rain.")])
    _timings, tts, _source, _host = await _window_turn("and tomorrow", brain)

    messages = brain.calls[0].messages
    assert messages[-3:] == [
        {"role": "user", "content": "what is the weather"},
        {"role": "assistant", "content": "It is sunny."},
        {"role": "user", "content": "and tomorrow"},
    ]
    assert tts.received_text == ["Rain."]
