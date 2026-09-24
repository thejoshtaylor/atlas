"""Step-level evidence for FLOW-02 (all three step kinds execute) and
FLOW-03 (a light fade's `transition`, and its refusal off a light) --
plan 05-03.

Against the real `execute_step` with a recording tool host and a
recording speech callable, no database needed: this file's subject is one
function's own dispatch and side-effect discipline, not durable storage
(which has its own tests in plan 05-01/05-02) or the fire-time policy
composition SAFE-08 claims (`test_workflow_fire_time_policy.py`'s own
job). `WorkflowStepRow` is constructed directly, in memory, with no
session attached -- this project's own `execute_step` takes the ORM row
itself (05-01 SUMMARY's own decision), and nothing here ever flushes or
commits one.

The last two tests are `turn/controller.py::_speak`'s own `speech_lock`
evidence (T-05-18): two concurrent utterances sharing one lock never
interleave on a shared sink, proven by chunk ordering, never by timing --
and, so that claim is not trivially true, the same two calls sharing no
lock at all genuinely do interleave against the identical scripted TTS.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

import pytest
from mcp.types import CallToolResult, TextContent

from atlas.config import WorkflowConfig
from atlas.db.models import WorkflowStepRow
from atlas.timing import TurnTimings
from atlas.turn.controller import _speak
from atlas.workflow.steps import (
    StepOutcome,
    UnwiredStepKindError,
    compose_lateness_sentence,
    execute_step,
)

from tests.conftest import FakeAudioSource

# A step's own `due_at`/`now` when lateness is not the thing under test --
# equal wall-clock values (one naive, one aware UTC), so `late_by_s` comes
# out to exactly 0.0 and never crosses `WorkflowConfig.late_threshold_s`.
_DUE_AT = datetime(2026, 1, 1, 12, 0, 0)
_NOW_ON_TIME = datetime(2026, 1, 1, 12, 0, 0, tzinfo=timezone.utc)
# 300s (5 minutes) past `_DUE_AT` -- comfortably past the 60s default
# threshold, and a round number `compose_lateness_sentence` renders as an
# exact, assertable sentence.
_NOW_LATE = datetime(2026, 1, 1, 12, 5, 0, tzinfo=timezone.utc)


def _make_step(
    *,
    kind: str,
    arguments: dict,
    due_at: datetime = _DUE_AT,
    position: int = 0,
) -> WorkflowStepRow:
    return WorkflowStepRow(
        id=1,
        run_id=1,
        position=position,
        kind=kind,
        arguments=arguments,
        due_at=due_at,
        status="pending",
        attempts=0,
        result_detail=None,
        fired_at=None,
    )


class _RecordingToolHost:
    """Records every call it receives; returns a fixed result, or raises
    a fixed exception -- never both, matching `_ToolHost`'s own one-method
    shape (`workflow/steps.py`)."""

    def __init__(self, *, result: CallToolResult | None = None, raises: Exception | None = None) -> None:
        self.calls: list[tuple[str, dict]] = []
        self._result = result
        self._raises = raises

    async def call_tool(self, name: str, arguments: dict) -> CallToolResult:
        self.calls.append((name, dict(arguments)))
        if self._raises is not None:
            raise self._raises
        assert self._result is not None
        return self._result


class _RecordingSpeak:
    """The injected `speak(text)` callable `_execute_speak` (and
    `execute_step`'s own lateness/refusal composer) calls -- records every
    text it was asked to say, or raises a fixed exception once, standing
    in for a synthesis failure."""

    def __init__(self, *, raises: Exception | None = None) -> None:
        self.spoken: list[str] = []
        self._raises = raises

    async def __call__(self, text: str) -> None:
        if self._raises is not None:
            raise self._raises
        self.spoken.append(text)


async def test_wait_step_completes_instantly_and_calls_nothing():
    step = _make_step(kind="wait", arguments={})
    tool_host = _RecordingToolHost()
    speak = _RecordingSpeak()

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME, speak=speak)

    assert outcome.status == "completed"
    assert tool_host.calls == []
    assert speak.spoken == []


async def test_speak_step_passes_its_exact_text_to_the_speech_callable():
    step = _make_step(kind="speak", arguments={"text": "the lights are on"})
    tool_host = _RecordingToolHost()
    speak = _RecordingSpeak()

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME, speak=speak)

    assert outcome.status == "completed"
    assert speak.spoken == ["the lights are on"]
    assert tool_host.calls == []


async def test_a_speak_step_with_no_speak_callable_raises_rather_than_completing():
    step = _make_step(kind="speak", arguments={"text": "hello"})
    tool_host = _RecordingToolHost()

    with pytest.raises(UnwiredStepKindError):
        await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME, speak=None)


async def test_a_failing_speech_callable_produces_a_retryable_outcome():
    step = _make_step(kind="speak", arguments={"text": "hello"})
    tool_host = _RecordingToolHost()
    speak = _RecordingSpeak(raises=RuntimeError("synthesis failed"))

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME, speak=speak)

    assert outcome.status == "failed"
    assert outcome.retry is True


async def test_a_call_service_failure_does_not_retry():
    """PA-D3 (05-01): a service call whose outcome is unknown must not be
    repeated -- unlike `speak`'s own retryable failure above, this kind's
    `retry` is `False` even when the call itself never reached a verdict."""
    step = _make_step(
        kind="call_service",
        arguments={"domain": "switch", "service": "turn_on", "entity_id": "switch.example_fan"},
    )
    tool_host = _RecordingToolHost(raises=RuntimeError("home assistant unreachable"))
    speak = _RecordingSpeak()

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME, speak=speak)

    assert outcome.status == "failed"
    assert outcome.retry is False


async def test_a_call_service_step_carrying_transition_off_a_light_is_denied_before_any_call():
    """FLOW-03's second layer (Task 1): the same domain restriction
    `mcp/atlas_mcp/ha.py` enforces at the process boundary, applied again
    here so a step authored with a bad `transition` never reaches that
    boundary at all -- and is spoken, verbatim, the same as any other
    fire-time refusal (D-14)."""
    step = _make_step(
        kind="call_service",
        arguments={
            "domain": "switch",
            "service": "turn_on",
            "entity_id": "switch.example_fan",
            "transition": 5.0,
        },
    )
    tool_host = _RecordingToolHost()
    speak = _RecordingSpeak()

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME, speak=speak)

    assert outcome.status == "denied"
    assert tool_host.calls == []
    assert speak.spoken == [outcome.detail["reason"]]


def test_compose_lateness_sentence_is_deterministic_and_exact():
    assert compose_lateness_sentence(60.0) == "sorry, this was about 1 minute late."
    assert compose_lateness_sentence(120.0) == "sorry, this was about 2 minutes late."
    assert compose_lateness_sentence(3600.0) == "sorry, this was about 1 hour late."
    assert compose_lateness_sentence(7200.0) == "sorry, this was about 2 hours late."
    # Same input, called twice -- no clock, no random source, same output.
    assert compose_lateness_sentence(300.0) == compose_lateness_sentence(300.0)


async def test_a_late_step_at_position_zero_records_and_speaks_the_lateness_sentence():
    step = _make_step(kind="wait", arguments={}, position=0)
    tool_host = _RecordingToolHost()
    speak = _RecordingSpeak()

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_LATE, speak=speak)

    assert outcome.detail["late_by_s"] == 300.0
    assert speak.spoken == [compose_lateness_sentence(300.0)]


async def test_a_late_step_at_position_one_records_without_speaking():
    step = _make_step(kind="wait", arguments={}, position=1)
    tool_host = _RecordingToolHost()
    speak = _RecordingSpeak()

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_LATE, speak=speak)

    assert outcome.detail["late_by_s"] == 300.0
    assert speak.spoken == []


async def test_a_step_not_yet_late_records_nothing():
    step = _make_step(kind="wait", arguments={}, position=0)
    tool_host = _RecordingToolHost()
    speak = _RecordingSpeak()

    outcome = await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME, speak=speak)

    assert "late_by_s" not in outcome.detail
    assert speak.spoken == []


async def test_an_unknown_kind_raises_rather_than_completing():
    step = _make_step(kind="branch", arguments={})
    tool_host = _RecordingToolHost()

    with pytest.raises(ValueError):
        await execute_step(step, tool_host, WorkflowConfig(), _NOW_ON_TIME)


class _SteppedTts:
    """Yields its scripted chunks with a cooperative yield point
    (`asyncio.sleep(0)`) between each one, so two concurrent `_speak`
    calls against two instances of this class have a genuine opportunity
    to interleave on a shared sink if nothing stops them -- the mechanism
    the lock tests below prove by chunk ordering, never by timing."""

    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def synthesize(self, text_deltas, sink=None):
        async for _ in text_deltas:
            pass
        for chunk in self._chunks:
            await asyncio.sleep(0)
            yield chunk


_CONTIGUOUS_ORDERS = (
    [b"A1", b"A2", b"A3", b"B1", b"B2", b"B3"],
    [b"B1", b"B2", b"B3", b"A1", b"A2", b"A3"],
)


async def test_two_concurrent_speak_calls_sharing_one_lock_never_interleave():
    source = FakeAudioSource()
    lock = asyncio.Lock()

    await asyncio.gather(
        _speak(source, _SteppedTts([b"A1", b"A2", b"A3"]), TurnTimings(), "a", kind="answer", speech_lock=lock),
        _speak(source, _SteppedTts([b"B1", b"B2", b"B3"]), TurnTimings(), "b", kind="answer", speech_lock=lock),
    )

    assert source.sent_audio in _CONTIGUOUS_ORDERS


async def test_the_same_two_calls_sharing_no_lock_do_interleave():
    """The control: proves the test above's own mechanism is capable of
    catching the failure it guards against, not merely trivially true of
    two sequential calls that never race at all."""
    source = FakeAudioSource()

    await asyncio.gather(
        _speak(source, _SteppedTts([b"A1", b"A2", b"A3"]), TurnTimings(), "a", kind="answer"),
        _speak(source, _SteppedTts([b"B1", b"B2", b"B3"]), TurnTimings(), "b", kind="answer"),
    )

    assert source.sent_audio not in _CONTIGUOUS_ORDERS
