"""Per-member home control (quick task 260929-p12).

An identified member whose `can_control_home` is false can talk to Atlas but
cannot run a home write. These tests prove the classifier, the guard host, the
speaker gate result, and every command path in `run_turn`: the model, the local
on/off intent, and a macro. A parallel group proves each turn gets its own
permission.

Every name, id, and phrase here is a generic placeholder.
"""

from __future__ import annotations

import dataclasses
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from atlas.config import MacroActionConfig, MacroConfig, SessionConfig
from atlas.providers.base import BrainReply, FinalTranscript, ToolCall
from atlas.session.recorder import SessionRecorder
from atlas.speaker_id.matching import MatchResult, ReferenceSet
from atlas.speaker_id.turn_gate import SpeakerIdTurnContext, evaluate_turn_speaker
from atlas.timing import TurnTimings
from atlas.turn import controller as controller_module
from atlas.turn.controller import run_turn
from atlas.turn.entity_claims import ClaimingToolHost, ClaimRegistry, bare_tool_name, is_home_write
from atlas.turn.home_control import (
    HOME_CONTROL_REFUSED_EVENT,
    HomeControlGuardHost,
    home_control_refusal_text,
    is_home_control_refusal,
    restrict_home_writes,
    workflow_has_home_write,
)
from atlas.turn.pending_action import EXECUTING_TOOL_BY_ACTION
from atlas_mcp.ha_names import HA_WRITE_TOOL_NAMES
from atlas.turn.reply_group import GroupSpeaker
from atlas.turn.turn_context import TurnContext
from tests.test_turn_claims import _FakeHomeAssistant
from tests.test_turn_group_speech import _Turn, _run_both
from tests.test_turn_speaker_gate import _measurement_for, _StubSpan, _StubTracker

_LAMP = {"domain": "light", "service": "turn_on", "entity_id": "light.example_lamp"}
_OTHER_LAMP = {"domain": "light", "service": "turn_on", "entity_id": "light.other_lamp"}
_STATES = [{"entity_id": "light.example_lamp", "friendly_name": "lamp", "state": "off"}]
_ALEX_REFUSAL = "Alex, you can't control the house."


def _alex_match() -> MatchResult:
    return MatchResult(best_speaker_id=1, best_name="Alex", best_score=0.9, second_score=0.1, margin=0.8, scores={1: 0.9})


def _sam_match() -> MatchResult:
    return MatchResult(best_speaker_id=2, best_name="Sam", best_score=0.9, second_score=0.1, margin=0.8, scores={2: 0.9})


def _references() -> ReferenceSet:
    references = ReferenceSet()
    references.upsert_speaker(1, "Alex", [(1.0, 0.0)])
    references.upsert_speaker(2, "Sam", [(0.0, 1.0)])
    return references


def _context(match: MatchResult | None, *, mode="enforce", denied=frozenset({1})) -> SpeakerIdTurnContext:
    span = _StubSpan(_measurement_for(match))
    return SpeakerIdTurnContext(
        tracker=_StubTracker(span),
        references=_references(),
        mode=mode,
        threshold=0.5,
        model_id="model",
        worker=object(),
        home_control_denied=set(denied),
    )


class _RecordingHost:
    """Records every call and answers with success."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.tools = object()

    async def call_tool(self, name: str, arguments: Any) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")], structured_content={})


# ---- the classifier --------------------------------------------------------


def test_is_home_write_classifies_a_service_call_as_a_write() -> None:
    assert is_home_write("ha_call_service", dict(_LAMP)) is True


def test_is_home_write_treats_a_get_service_as_a_read() -> None:
    assert is_home_write("ha_call_service", {"domain": "todo", "service": "get_items"}) is False


def test_is_home_write_treats_the_spotify_playlist_tool_as_a_write() -> None:
    assert is_home_write("ha_play_spotify_playlist", {"playlist": "example"}) is True


def test_is_home_write_is_false_for_a_read_tool_name() -> None:
    assert is_home_write("ha_get_states", {}) is False
    assert is_home_write("set_speaker_volume", {"level": 3}) is False


def test_is_home_write_fails_closed_on_non_dict_arguments() -> None:
    assert is_home_write("ha_call_service", "not a dict") is True
    assert is_home_write("ha_call_service", None) is True


def test_is_home_write_counts_a_service_call_with_no_target() -> None:
    assert is_home_write("ha_call_service", {"domain": "light", "service": "turn_on"}) is True


def test_bare_tool_name_strips_a_plugin_prefix() -> None:
    assert bare_tool_name("slug__ha_call_service") == "ha_call_service"
    assert bare_tool_name("ha_call_service") == "ha_call_service"


# ---- the guard host --------------------------------------------------------


async def test_the_guard_refuses_a_home_write_and_the_inner_host_gets_no_call() -> None:
    inner = _RecordingHost()
    events: list[dict] = []
    guard = HomeControlGuardHost(inner, name="Alex", record_event=events.append)

    result = await guard.call_tool("ha_call_service", dict(_LAMP))

    assert inner.calls == []
    assert result.isError is True
    assert is_home_control_refusal(result) == _ALEX_REFUSAL
    assert events == [{"type": HOME_CONTROL_REFUSED_EVENT, "tool": "ha_call_service"}]
    assert "Alex" not in str(events)


async def test_the_guard_refuses_a_prefixed_write_name() -> None:
    inner = _RecordingHost()
    guard = HomeControlGuardHost(inner, name="Alex")

    result = await guard.call_tool("slug__ha_call_service", dict(_LAMP))

    assert inner.calls == []
    assert is_home_control_refusal(result) == _ALEX_REFUSAL


def test_the_refusal_has_no_name_when_the_name_is_none() -> None:
    assert home_control_refusal_text(None) == "you can't control the house."
    assert home_control_refusal_text("Alex") == _ALEX_REFUSAL


async def test_the_guard_forwards_reads_and_other_tools_unchanged() -> None:
    inner = _RecordingHost()
    guard = HomeControlGuardHost(inner, name="Alex")

    calls = [
        ("ha_get_states", {}),
        ("ha_call_service", {"domain": "todo", "service": "get_items"}),
        ("set_speaker_volume", {"level": 3}),
        ("cancel_workflow_run", {"run_id": "run-1"}),
    ]
    for name, arguments in calls:
        result = await guard.call_tool(name, arguments)
        assert is_home_control_refusal(result) is None

    assert inner.calls == calls
    assert guard.tools is inner.tools


async def test_a_speaker_argument_does_not_change_the_refusal() -> None:
    inner = _RecordingHost()
    guard = HomeControlGuardHost(inner, name="Alex")

    result = await guard.call_tool("ha_call_service", {**_LAMP, "speaker": "Sam"})

    assert inner.calls == []
    assert is_home_control_refusal(result) == _ALEX_REFUSAL


def test_call_tool_takes_only_a_name_and_arguments() -> None:
    parameters = list(inspect.signature(HomeControlGuardHost.call_tool).parameters)
    assert parameters == ["self", "name", "arguments"]


def test_restrict_home_writes_returns_the_host_itself_when_allowed() -> None:
    host = _RecordingHost()
    assert restrict_home_writes(host, allowed=True, name="Alex") is host


def test_restrict_home_writes_returns_none_for_no_host() -> None:
    assert restrict_home_writes(None, allowed=False, name="Alex") is None


def test_restrict_home_writes_wraps_the_host_when_not_allowed() -> None:
    host = _RecordingHost()
    guarded = restrict_home_writes(host, allowed=False, name="Alex")
    assert isinstance(guarded, HomeControlGuardHost)
    assert guarded.inner is host


# ---- the gate result -------------------------------------------------------


async def test_the_gate_denies_an_identified_restricted_member_in_enforce_mode() -> None:
    outcome = await evaluate_turn_speaker(
        _context(_alex_match()), _StubSpan(_measurement_for(_alex_match())), timings=TurnTimings()
    )
    assert outcome.can_control_home is False
    assert outcome.speaker_name == "Alex"


async def test_the_gate_permits_another_member_in_enforce_mode() -> None:
    outcome = await evaluate_turn_speaker(
        _context(_sam_match()), _StubSpan(_measurement_for(_sam_match())), timings=TurnTimings()
    )
    assert outcome.can_control_home is True


async def test_the_gate_does_no_check_in_record_mode() -> None:
    outcome = await evaluate_turn_speaker(
        _context(_alex_match(), mode="record"), _StubSpan(_measurement_for(_alex_match())), timings=TurnTimings()
    )
    assert outcome.can_control_home is True


async def test_the_gate_permits_when_there_is_no_context_or_the_mode_is_off() -> None:
    assert (await evaluate_turn_speaker(None, None, timings=TurnTimings())).can_control_home is True
    off = SpeakerIdTurnContext(
        tracker=None, references=None, mode="off", threshold=0.5, model_id=None, worker=None, home_control_denied={1}
    )
    assert (await evaluate_turn_speaker(off, None, timings=TurnTimings())).can_control_home is True


def test_turn_context_carries_no_home_control_permission() -> None:
    names = [field.name for field in dataclasses.fields(TurnContext)]
    assert not [name for name in names if "home" in name]


# ---- run_turn --------------------------------------------------------------


def _lamp_call_reply(arguments: dict | None = None) -> BrainReply:
    return BrainReply(tool_calls=[ToolCall(name="ha_call_service", arguments=dict(arguments or _LAMP))])


async def _state_fetch():
    return list(_STATES)


async def _run_single(
    tmp_path,
    fake_audio_source,
    fake_stt,
    fake_brain,
    fake_tts,
    *,
    speaker_id,
    tool_host,
    transcript="turn on the lamp",
    replies=None,
    **run_kwargs,
):
    source = fake_audio_source(frames=[b"\x00\x01"] * 3)
    stt = fake_stt(events=[FinalTranscript(text=transcript)])
    brain = fake_brain(replies=replies if replies is not None else [_lamp_call_reply(), BrainReply(text="done")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()
    recorder = SessionRecorder(SessionConfig(dir=str(tmp_path)), timings)
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
        session_recorder=recorder,
        speaker_id=speaker_id,
        **run_kwargs,
    )
    return tts, timings


async def test_the_model_path_refuses_a_restricted_members_write(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
) -> None:
    home = _FakeHomeAssistant()

    tts, _ = await _run_single(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        speaker_id=_context(_alex_match()), tool_host=home,
    )

    assert home.calls == []
    assert tts.received_text == [_ALEX_REFUSAL]


async def test_the_local_intent_path_refuses_a_restricted_members_write(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
) -> None:
    home = _FakeHomeAssistant()

    tts, timings = await _run_single(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        speaker_id=_context(_alex_match()), tool_host=home, local_intents=True, state_fetch=_state_fetch,
    )

    assert home.calls == []
    assert tts.received_text == [_ALEX_REFUSAL]
    assert "i can't do that one" not in tts.received_text[0]
    assert timings.turn_outcome == "home_control_refused"


async def test_the_macro_path_refuses_a_restricted_members_write_through_the_guard_over_the_unclaimed_host(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, monkeypatch
) -> None:
    macro = MacroConfig(
        phrase="lamp time",
        aliases=(),
        reply="lamp done",
        actions=(MacroActionConfig(tool="ha_call_service", arguments=dict(_LAMP)),),
    )
    home = _FakeHomeAssistant()
    claiming = ClaimingToolHost(home, registry=ClaimRegistry(), owner="src:1", label=lambda: "Alex")
    seen: list[Any] = []
    real_fire_macro = controller_module.fire_macro

    async def spy(macro_arg, host, **kwargs):
        seen.append(host)
        return await real_fire_macro(macro_arg, host, **kwargs)

    monkeypatch.setattr(controller_module, "fire_macro", spy)

    tts, timings = await _run_single(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        speaker_id=_context(_alex_match()), tool_host=claiming, transcript="lamp time", macros=(macro,),
    )

    assert home.calls == []
    assert tts.received_text == [_ALEX_REFUSAL]
    assert timings.turn_outcome == "macro_failed"
    (host,) = seen
    assert isinstance(host, HomeControlGuardHost)
    assert host.inner is home
    assert not isinstance(host.inner, ClaimingToolHost)


@pytest.mark.parametrize(
    ("match", "mode"),
    [(_sam_match, "enforce"), (_alex_match, "record")],
)
async def test_a_permitted_member_and_record_mode_still_reach_home_assistant(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts, match, mode
) -> None:
    home = _FakeHomeAssistant()

    await _run_single(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        speaker_id=_context(match(), mode=mode), tool_host=home,
    )

    assert len(home.calls) == 1


# ---- a parallel group ------------------------------------------------------


async def test_each_turn_in_a_group_gets_its_own_permission_and_the_refusal_names_the_member_once(
    tmp_path, fake_stt, fake_brain, fake_tts
) -> None:
    speaker = GroupSpeaker(merge_wait_s=0.3)
    live_tts = fake_tts(chunks=[b"\x01\x02"])
    common = dict(
        live_tts=live_tts,
        fake_stt=fake_stt,
        fake_brain=fake_brain,
        speaker_mode="enforce",
        home_control_denied={1},
    )
    home = _FakeHomeAssistant()
    alex = _Turn(
        tmp_path, speaker, key="src:1", order_frame=100, member=(1, "Alex"), transcript="turn on the lamp",
        brain=fake_brain(replies=[_lamp_call_reply(_LAMP)]), tool_host=home, **common,
    )
    sam = _Turn(
        tmp_path, speaker, key="src:2", order_frame=300, member=(2, "Sam"), transcript="turn on the other lamp",
        brain=fake_brain(replies=[_lamp_call_reply(_OTHER_LAMP), BrainReply(text="done")]), tool_host=home, **common,
    )

    await _run_both(alex, sam)

    assert [arguments["entity_id"] for _, arguments in home.calls] == ["light.other_lamp"]
    (spoken,) = live_tts.received_text
    assert spoken.count(_ALEX_REFUSAL) == 1
    assert "Alex, Alex" not in spoken


# ---- scheduled workflows ---------------------------------------------------

_WAIT_STEP = {"kind": "wait", "arguments": {"duration_s": 60}}
_SPEAK_STEP = {"kind": "speak", "arguments": {"text": "time is up"}}
_WRITE_STEP = {"kind": "call_service", "arguments": {"domain": "light", "service": "turn_off", "entity_id": "light.example_lamp"}}
_READ_STEP = {"kind": "call_service", "arguments": {"domain": "todo", "service": "get_items"}}


def _schedule(steps) -> dict:
    return {"steps": steps, "delay_seconds": 60, "summary": "lamp off later"}


@pytest.mark.parametrize("tool_name", ["schedule_workflow", "append_workflow_steps"])
@pytest.mark.parametrize(
    "arguments",
    [
        _schedule([_WAIT_STEP, _WRITE_STEP]),
        {"kind": "call_service", "arguments": _WRITE_STEP["arguments"], "delay_seconds": 5, "summary": "x"},
        _schedule("not a list"),
        _schedule([_WAIT_STEP, "not a dict"]),
        _schedule([{"kind": "loop", "arguments": {}}]),
        _schedule([{"kind": "call_service", "arguments": "not a dict"}]),
        "not a dict",
    ],
)
async def test_the_guard_refuses_a_workflow_with_a_home_write_or_unreadable_steps(tool_name, arguments) -> None:
    inner = _RecordingHost()
    guard = HomeControlGuardHost(inner, name="Alex")

    result = await guard.call_tool(tool_name, arguments)

    assert inner.calls == []
    assert is_home_control_refusal(result) == _ALEX_REFUSAL


async def test_the_guard_refuses_an_appended_home_write_step() -> None:
    inner = _RecordingHost()
    guard = HomeControlGuardHost(inner, name="Alex")

    result = await guard.call_tool("append_workflow_steps", {"run_id": 4, "steps": [_WRITE_STEP]})

    assert inner.calls == []
    assert is_home_control_refusal(result) == _ALEX_REFUSAL


async def test_the_guard_forwards_workflows_without_a_home_write() -> None:
    inner = _RecordingHost()
    guard = HomeControlGuardHost(inner, name="Alex")

    calls = [
        ("schedule_workflow", _schedule([_WAIT_STEP, _SPEAK_STEP])),
        ("schedule_workflow", _schedule([_READ_STEP])),
        ("append_workflow_steps", {"run_id": 4, "steps": [_SPEAK_STEP]}),
        ("cancel_workflow_run", {"run_id": 4}),
    ]
    for name, arguments in calls:
        result = await guard.call_tool(name, arguments)
        assert is_home_control_refusal(result) is None

    assert inner.calls == calls


async def test_a_permitted_member_schedules_a_call_service_workflow() -> None:
    inner = _RecordingHost()
    host = restrict_home_writes(inner, allowed=True, name="Alex")

    await host.call_tool("schedule_workflow", _schedule([_WRITE_STEP]))

    assert len(inner.calls) == 1


def test_workflow_has_home_write_reads_steps_and_the_legacy_flat_form() -> None:
    assert workflow_has_home_write(_schedule([_WRITE_STEP])) is True
    assert workflow_has_home_write(_schedule([_WAIT_STEP, _SPEAK_STEP])) is False
    assert workflow_has_home_write({"kind": "call_service", "arguments": _WRITE_STEP["arguments"]}) is True
    assert workflow_has_home_write({"kind": "wait", "arguments": {"duration_s": 5}}) is False


class _RecordingWorkflowHost:
    """A workflow tool host double that records every call."""

    def __init__(self) -> None:
        self.calls: list[tuple[str, Any]] = []
        self.tools = object()

    async def call_tool(self, name: str, arguments: Any) -> Any:
        self.calls.append((name, arguments))
        return SimpleNamespace(isError=False, content=[SimpleNamespace(text="{}")], structured_content={})


async def test_run_turn_refuses_a_restricted_members_scheduled_home_write(
    tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts
) -> None:
    workflow = _RecordingWorkflowHost()
    reply = BrainReply(
        tool_calls=[ToolCall(name="schedule_workflow", arguments=_schedule([_WAIT_STEP, _WRITE_STEP]))]
    )

    tts, _ = await _run_single(
        tmp_path, fake_audio_source, fake_stt, fake_brain, fake_tts,
        speaker_id=_context(_alex_match()), tool_host=workflow,
        transcript="turn the lamp off in a minute", replies=[reply],
    )

    assert workflow.calls == []
    assert tts.received_text == [_ALEX_REFUSAL]


def test_a_pending_action_confirmation_never_runs_a_home_write() -> None:
    assert set(EXECUTING_TOOL_BY_ACTION.values()).isdisjoint(HA_WRITE_TOOL_NAMES)
