"""`summarise_pending_runs`: what the house is about to do, rendered into
one deterministic block for the model's per-turn context (plan 05-05
Task 2, D-09).

Composed here, in code, exactly the same discipline
`turn/controller.py`'s own `_compose_clarifying_question`/
`_compose_mixed_outcome_reply` already carry and `workflow/steps.py`'s
`compose_lateness_sentence` already establishes for a workflow-adjacent
sentence specifically: the same input and the same `now` always produce
the same string, and nothing in here reads a clock, a random source, or
any other per-call state of its own -- `now` always arrives as a
parameter, from the one caller (`app.py`'s `_make_pending_runs_fetch`
factory, via `turn/controller.py`'s own injected-fetch discipline) that
already has an aware instant to hand it.

This module turns two already-absolute `datetime` values (`due_at`,
`now`) into a relative English phrase -- ordinary duration arithmetic on
two instants that already exist (subtracting one aware `datetime` from
another, which needs no `timedelta` import of its own), the same thing
`db/repository.py`'s own `push_out_due_at`/`next_append_due_at`/
`assign_step_due_ats` already do. It is not the boundary
`workflow.schedule.resolve_schedule` owns: nothing here ever turns a
wall-clock *string* into an absolute instant, so this module carries none
of that boundary's own negative-grep constraints.
"""

from __future__ import annotations

from datetime import datetime
from typing import Sequence

from spire_voice.db.repository import WorkflowRun

_NO_RUNS_LINE = "Scheduled runs: nothing is scheduled right now."


def _format_duration(seconds: float) -> str:
    """A whole-unit English duration, the same fixed rounding rule
    `workflow.steps.compose_lateness_sentence` already applies to a
    step's own lateness: whole seconds under a minute, whole minutes
    under an hour, whole hours under a day, whole days from a day on. An
    operator does not need second-level precision spoken (or read by a
    model) for "about three hours" any more than for a step's own
    lateness, and a fixed rounding rule is what keeps this function's
    output deterministic for a fixed input."""
    seconds = max(0.0, seconds)
    if seconds < 60:
        n = max(1, round(seconds))
        return f"{n} second{'s' if n != 1 else ''}"
    minutes = seconds / 60
    if minutes < 60:
        n = max(1, round(minutes))
        return f"{n} minute{'s' if n != 1 else ''}"
    hours = seconds / 3600
    if hours < 24:
        n = max(1, round(hours))
        return f"{n} hour{'s' if n != 1 else ''}"
    n = max(1, round(hours / 24))
    return f"{n} day{'s' if n != 1 else ''}"


def _relative_due_description(due_at: datetime, now: datetime) -> str:
    """`due_at` in the past reads as "overdue by ..." rather than a
    negative duration -- the one word a model (or an operator reading a
    transcript) needs to tell a run running late from one still ahead."""
    delta = (due_at - now).total_seconds()
    if delta < 0:
        return f"overdue by {_format_duration(-delta)}"
    if delta < 1:
        return "due now"
    return f"due in {_format_duration(delta)}"


def summarise_pending_runs(runs: Sequence[WorkflowRun], now: datetime) -> str:
    """Render every pending or firing run into one block the caller
    injects into the model's per-turn context, the same volatile half
    `_state_message` (`app.py`) already occupies -- never the cacheable
    catalog prefix (D-09).

    Prose the model can match loose speech against, not a table of
    database columns: each run's own stored `summary` (D-09's whole
    mechanism -- the operator's own words, stored once at creation, never
    recomputed here from the steps) carries what the operator would say to
    mean it, and this function's own id, due description, remaining-step
    count, and origin ride alongside it -- the id being the one thing this
    block carries that the operator never speaks aloud (D-09).

    `runs` is expected to already be the pending/firing subset
    (`WorkflowRepository.list_runs(statuses=("pending", "firing"))`,
    D-16's own list) -- this function performs no status filtering of its
    own, only rendering. An empty sequence still renders an explicit
    "nothing is scheduled" line rather than no text at all, the same
    reason `_state_message` renders its own header against zero entity
    lines: silence is not a claim the model can read, and "nothing is
    scheduled" is.
    """
    if not runs:
        return _NO_RUNS_LINE

    lines = ["Scheduled runs:"]
    for run in runs:
        pending_steps = [step for step in run.steps if step.status == "pending"]
        step_count = len(pending_steps)
        if pending_steps:
            next_due = min(step.due_at for step in pending_steps)
            due_desc = _relative_due_description(next_due, now)
        else:
            due_desc = "no steps remaining"
        plural = "" if step_count == 1 else "s"
        lines.append(
            f"- id {run.id}: {run.summary} ({run.origin}) -- {due_desc}, "
            f"{step_count} step{plural} remaining"
        )
    return "\n".join(lines)
