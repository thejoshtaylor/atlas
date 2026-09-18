"""Plan 05-05: everything the operator does to a scheduled plan with their
voice -- making one with more than one ordered step (FLOW-01), cancelling
one without knowing its id (FLOW-04), adding to one (FLOW-05), and being
asked which one they meant when the words fit more than one (FLOW-06).

Task 1 drives the widened `WorkflowToolHost` directly, against
`tests/conftest.py`'s `FakeWorkflowRepository` -- no database, the same
Postgres-free discipline `test_workflow_repo.py` already establishes for
everything but its own one `integration`-marked test. Task 2 adds the
pending-run fetch's own turn-level evidence; Task 3 adds the end-to-end
`needs_clarification` shape for a pending run. Every entity id below is
invented, per `tests/test_repo_hygiene.py`'s own rule.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from spire_voice.workflow.tool import WorkflowToolHost


def _host(repo, *, zone=None, now=None):
    clock_time = [now or datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)]
    host = WorkflowToolHost(repo, zone=zone, clock=lambda: clock_time[0])
    return host, clock_time


# ---------------------------------------------------------------------------
# schedule_workflow: an ordered multi-step list, and the two time fields
# ---------------------------------------------------------------------------


async def test_a_three_step_call_stores_three_steps_in_spoken_order_with_kinds_intact(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)

    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [
                {"kind": "wait", "arguments": {"duration_s": 60}},
                {
                    "kind": "call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": "switch.example_porch_light",
                    },
                },
                {"kind": "speak", "arguments": {"text": "the porch light is off now"}},
            ],
            "delay_seconds": 300,
            "summary": "turn off the porch light in a bit",
        },
    )
    assert not getattr(result, "is_error", False), result

    runs = await repo.list_runs()
    assert len(runs) == 1
    run = runs[0]
    assert run.summary == "turn off the porch light in a bit"
    assert [step.kind for step in run.steps] == ["wait", "call_service", "speak"]
    assert [step.position for step in run.steps] == [0, 1, 2]


async def test_a_wait_step_pushes_only_the_later_steps_due_at(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)

    await host.call_tool(
        "schedule_workflow",
        {
            "steps": [
                {
                    "kind": "call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_on",
                        "entity_id": "switch.example_kettle",
                    },
                },
                {"kind": "wait", "arguments": {"duration_s": 120}},
                {
                    "kind": "call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": "switch.example_kettle",
                    },
                },
            ],
            "delay_seconds": 0,
            "summary": "run the kettle for two minutes",
        },
    )

    run = (await repo.list_runs())[0]
    first, wait, last = run.steps
    assert last.due_at == first.due_at + timedelta(seconds=120)
    assert wait.due_at == first.due_at


async def test_both_time_fields_given_is_rejected_with_its_own_text(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)

    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [{"kind": "speak", "arguments": {"text": "hello"}}],
            "delay_seconds": 60,
            "at": "2027-01-01T13:00:00+00:00",
            "summary": "say hello",
        },
    )
    assert result.is_error
    assert "exactly one" in result.content[0].text


async def test_neither_time_field_given_is_rejected_with_distinguishable_text(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)

    both_given = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [{"kind": "speak", "arguments": {"text": "hello"}}],
            "delay_seconds": 60,
            "at": "2027-01-01T13:00:00+00:00",
            "summary": "say hello",
        },
    )
    neither_given = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [{"kind": "speak", "arguments": {"text": "hello"}}],
            "summary": "say hello",
        },
    )
    assert both_given.is_error and neither_given.is_error
    assert both_given.content[0].text != neither_given.content[0].text


async def test_an_at_value_with_no_offset_and_no_configured_zone_is_rejected(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    host, _clock = _host(repo, zone=None)

    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [{"kind": "speak", "arguments": {"text": "hello"}}],
            "at": "2027-01-01T13:00:00",
            "summary": "say hello",
        },
    )
    assert result.is_error
    assert "server.timezone" in result.content[0].text


async def test_an_ambiguous_local_at_value_is_refused_with_its_own_distinguishable_text(
    fake_workflow_repository,
):
    from zoneinfo import ZoneInfo

    repo = fake_workflow_repository()
    zone = ZoneInfo("America/Los_Angeles")
    host, _clock = _host(repo, zone=zone)

    # 2026-11-01T01:30 local occurs twice in America/Los_Angeles (DST ends).
    ambiguous = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [{"kind": "speak", "arguments": {"text": "hello"}}],
            "at": "2026-11-01T01:30:00",
            "summary": "say hello",
        },
    )
    both_given = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [{"kind": "speak", "arguments": {"text": "hello"}}],
            "delay_seconds": 5,
            "at": "2026-11-01T01:30:00",
            "summary": "say hello",
        },
    )
    assert ambiguous.is_error
    assert "ambiguous" in ambiguous.content[0].text
    assert ambiguous.content[0].text != both_given.content[0].text


async def test_a_step_with_an_unknown_kind_is_rejected(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)

    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [{"kind": "loop_forever", "arguments": {}}],
            "delay_seconds": 60,
            "summary": "do something forbidden",
        },
    )
    assert result.is_error


async def test_a_step_with_an_extra_field_is_rejected(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)

    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [
                {
                    "kind": "speak",
                    "arguments": {"text": "hello"},
                    "condition": "the door is locked",
                }
            ],
            "delay_seconds": 60,
            "summary": "say hello conditionally",
        },
    )
    assert result.is_error


async def test_a_transition_value_on_a_non_light_domain_is_rejected(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)

    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [
                {
                    "kind": "call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": "switch.example_fan",
                        "transition": 5,
                    },
                }
            ],
            "delay_seconds": 60,
            "summary": "turn off the fan",
        },
    )
    assert result.is_error
    assert "transition" in result.content[0].text


# ---------------------------------------------------------------------------
# cancel_workflow_run (FLOW-04, D-09)
# ---------------------------------------------------------------------------


async def _schedule_one(host) -> int:
    result = await host.call_tool(
        "schedule_workflow",
        {
            "steps": [
                {
                    "kind": "call_service",
                    "arguments": {
                        "domain": "switch",
                        "service": "turn_off",
                        "entity_id": "switch.example_fan",
                    },
                }
            ],
            "delay_seconds": 300,
            "summary": "turn off the fan later",
        },
    )
    return result.structured_content["run_id"]


async def test_cancel_on_a_pending_run_succeeds(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)
    run_id = await _schedule_one(host)

    result = await host.call_tool("cancel_workflow_run", {"run_id": run_id})

    assert not getattr(result, "is_error", False)
    run = await repo.get_run(run_id)
    assert run.status == "cancelled"


async def test_cancel_on_an_already_terminal_run_is_refused_by_name(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)
    run_id = await _schedule_one(host)
    await host.call_tool("cancel_workflow_run", {"run_id": run_id})  # now cancelled

    result = await host.call_tool("cancel_workflow_run", {"run_id": run_id})

    assert result.is_error
    assert str(run_id) in result.content[0].text


async def test_cancel_on_a_missing_id_is_refused_with_different_text_than_a_terminal_run(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)
    run_id = await _schedule_one(host)
    await host.call_tool("cancel_workflow_run", {"run_id": run_id})  # now cancelled
    terminal_result = await host.call_tool("cancel_workflow_run", {"run_id": run_id})

    missing_result = await host.call_tool("cancel_workflow_run", {"run_id": run_id + 999})

    assert missing_result.is_error
    # Both refusals share the same "not pending or firing" wording in this
    # host (the repository does not itself distinguish missing from
    # terminal for cancel) -- what must differ is the id each names.
    assert str(run_id + 999) in missing_result.content[0].text
    assert missing_result.content[0].text != terminal_result.content[0].text


# ---------------------------------------------------------------------------
# append_workflow_steps (FLOW-05, D-11)
# ---------------------------------------------------------------------------


async def test_append_on_a_pending_run_lands_after_the_existing_tail(fake_workflow_repository):
    repo = fake_workflow_repository()
    host, _clock = _host(repo)
    run_id = await _schedule_one(host)

    result = await host.call_tool(
        "append_workflow_steps",
        {
            "run_id": run_id,
            "steps": [{"kind": "speak", "arguments": {"text": "the fan is off"}}],
        },
    )

    assert not getattr(result, "is_error", False)
    run = await repo.get_run(run_id)
    assert [step.kind for step in run.steps] == ["call_service", "speak"]
    assert run.steps[1].position == 1
    assert run.steps[1].due_at >= run.steps[0].due_at


async def test_append_on_a_firing_run_is_refused_distinguishably_from_a_missing_run(
    fake_workflow_repository,
):
    repo = fake_workflow_repository()
    host, clock_time = _host(repo)
    run_id = await _schedule_one(host)

    # Move the run into "firing" the same way the fake repository's own
    # claim path does -- directly, since this file's subject is the tool
    # host's own dispatch, not the poller (`test_workflow_scheduler_*`'s
    # job).
    repo._runs[run_id]["status"] = "firing"

    firing_result = await host.call_tool(
        "append_workflow_steps",
        {
            "run_id": run_id,
            "steps": [{"kind": "speak", "arguments": {"text": "too late"}}],
        },
    )
    missing_result = await host.call_tool(
        "append_workflow_steps",
        {
            "run_id": run_id + 999,
            "steps": [{"kind": "speak", "arguments": {"text": "too late"}}],
        },
    )

    assert firing_result.is_error
    assert missing_result.is_error
    assert firing_result.content[0].text != missing_result.content[0].text
    assert "does not exist" in missing_result.content[0].text


# ---------------------------------------------------------------------------
# Task 2: the pending-runs fetch, turn-level (D-09, T-01.1-17's own posture
# extended to the second injected fetch)
# ---------------------------------------------------------------------------


async def test_a_raising_pending_runs_fetch_still_reaches_speech(
    fake_audio_source, fake_stt, fake_brain, fake_tts
):
    """A pending-runs fetch that raises is logged and treated as nothing
    scheduled known rather than ending the turn -- the identical
    T-01.1-17 posture `state_fetch` already carries, applied to D-09's
    own second fetch."""
    from spire_voice.providers.base import BrainReply, FinalTranscript
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    async def _raising_pending_runs_fetch():
        raise RuntimeError("workflow repository unreachable")

    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="turn on the fan")])
    brain = fake_brain(replies=[BrainReply(text="turned on the fan")])
    tts = fake_tts(chunks=[b"\x01\x02"])
    timings = TurnTimings()

    await run_turn(
        source,
        stt,
        brain,
        tts,
        None,
        tools_schema=[],
        system_prompt="you control a home",
        max_tool_rounds=3,
        timings=timings,
        pending_runs_fetch=_raising_pending_runs_fetch,
    )

    assert tts.received_text == ["turned on the fan"]
    assert timings.turn_outcome == "completed"


async def test_a_macro_turn_cancels_a_started_pending_runs_fetch_without_leaving_it_dangling(
    fake_audio_source, fake_stt, fake_brain, fake_tts, fake_ha
):
    """A macro turn never builds the message list its result would have
    joined -- but the fetch was already started before the macro check
    ran, so it must be cancelled and its cancellation awaited, never left
    dangling, the identical shape `state_task` already follows."""
    import asyncio

    from spire_mcp.safety import Policy
    from spire_voice.config import MacroActionConfig, MacroConfig
    from spire_voice.providers.base import FinalTranscript
    from spire_voice.timing import TurnTimings
    from spire_voice.turn.controller import run_turn

    from test_turn_controller import _FakeToolHost

    macro = MacroConfig(
        phrase="good night",
        aliases=(),
        reply="good night",
        actions=(
            MacroActionConfig(
                tool="ha_call_service",
                arguments={"domain": "switch", "service": "turn_off", "entity_id": "switch.example_fan"},
            ),
        ),
    )
    policy = Policy.from_config(None)
    source = fake_audio_source(frames=[b"\x00\x01"])
    stt = fake_stt(events=[FinalTranscript(text="good night")])
    brain = fake_brain(replies=[])
    tts = fake_tts(chunks=[])
    tool_host = _FakeToolHost(fake_ha, policy)
    timings = TurnTimings()

    cancelled: list[bool] = []

    async def _hanging_pending_runs_fetch():
        try:
            await asyncio.sleep(10)
        except asyncio.CancelledError:
            cancelled.append(True)
            raise
        return ()

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
        macros=(macro,),
        filler_cache={"good night": b"\x01\x02"},
        pending_runs_fetch=_hanging_pending_runs_fetch,
    )

    assert timings.turn_outcome == "macro"
    assert cancelled == [True]
