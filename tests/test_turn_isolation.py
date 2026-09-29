"""Phase 12 D-12: nothing crosses between two turns that run at the same time.

Three pieces of per-source state used to be shared by accident:

- the workflow run-id scope on `WorkflowToolHost` (Research Pitfall 6),
- the pending-action supersede key (Research Pitfall 5),
- the local speech models (Research Pitfall 11).

Each section below covers one of them.
"""

from __future__ import annotations

import asyncio
from datetime import datetime, timezone

from atlas.workflow.tool import WorkflowToolHost


def _host(repo) -> WorkflowToolHost:
    fixed_now = datetime(2027, 1, 1, 12, 0, tzinfo=timezone.utc)
    return WorkflowToolHost(repo, zone=None, clock=lambda: fixed_now)


async def _schedule_one(host: WorkflowToolHost) -> int:
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


# ---------------------------------------------------------------------------
# Task 1: the workflow run-id scope is per task
# ---------------------------------------------------------------------------


async def test_two_concurrent_turns_each_keep_their_own_run_id_scope(fake_workflow_repository):
    repo = fake_workflow_repository()
    host = _host(repo)
    run_a = await _schedule_one(host)
    run_b = await _schedule_one(host)

    a_has_set = asyncio.Event()
    b_has_set = asyncio.Event()

    async def turn_a():
        host.set_current_turn_run_ids({run_a})
        a_has_set.set()
        await b_has_set.wait()  # turn B has now set its own ids
        own = await host.call_tool("cancel_workflow_run", {"run_id": run_a})
        other = await host.call_tool("cancel_workflow_run", {"run_id": run_b})
        return own, other

    async def turn_b():
        await a_has_set.wait()
        host.set_current_turn_run_ids({run_b})
        b_has_set.set()
        return await host.call_tool("cancel_workflow_run", {"run_id": run_a})

    (a_own, a_other), b_foreign = await asyncio.gather(
        asyncio.create_task(turn_a()), asyncio.create_task(turn_b())
    )

    assert not getattr(a_own, "is_error", False)
    assert a_other.is_error
    assert "not in this turn's own pending-run list" in a_other.content[0].text
    assert b_foreign.is_error
    assert "not in this turn's own pending-run list" in b_foreign.content[0].text
    # Turn B never lost its own run to turn A's cancel.
    assert (await repo.get_run(run_b)).status == "pending"


async def test_a_task_that_never_set_run_ids_refuses_every_run_id(fake_workflow_repository):
    repo = fake_workflow_repository()
    host = _host(repo)
    run_id = await _schedule_one(host)

    async def setter():
        host.set_current_turn_run_ids({run_id})

    await asyncio.create_task(setter())  # another task's scope must not leak here

    cancel = await host.call_tool("cancel_workflow_run", {"run_id": run_id})
    append = await host.call_tool(
        "append_workflow_steps",
        {"run_id": run_id, "steps": [{"kind": "speak", "arguments": {"text": "hi"}}]},
    )

    assert cancel.is_error
    assert append.is_error


async def test_a_child_task_sees_the_ids_its_parent_turn_set(fake_workflow_repository):
    repo = fake_workflow_repository()
    host = _host(repo)
    run_id = await _schedule_one(host)
    other_id = await _schedule_one(host)
    host.set_current_turn_run_ids({run_id})

    async def child(target: int):
        return await host.call_tool("cancel_workflow_run", {"run_id": target})

    shown, unshown = await asyncio.gather(child(run_id), child(other_id))

    assert not getattr(shown, "is_error", False)
    assert unshown.is_error
