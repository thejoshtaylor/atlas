"""Workflow authoring over HTTP: list what the house is about to do, read
one run, author one, edit one, and cancel one (FLOW-09, FLOW-10).

Every route here -- reads included -- requires `Role.OPERATOR`, matching
`routes/macros.py`'s own stated reasoning: an operator edits these day to
day the way they edit policy, and there is no structurally destructive
operation here that needs the admin gate the policy mode switch carries.

Spoken and webapp-authored runs come back from the same `GET /api/workflows`
list, in the same shape, distinguished only by `origin` (D-08, D-16) -- this
module never builds a second read path for one origin or the other.

The conflict annotation on each `call_service` step is computed by
`routes/conflict.py`'s shared `annotate_conflict`, the exact same module
`routes/macros.py` reads from -- never a second, workflow-specific copy of
the check (see that module's own docstring for the full reasoning).

`workflow.schedule.resolve_schedule` is the only place this module turns a
caller's "Run at" string into an absolute instant -- this module performs
no zone handling, no ISO parsing, and no offset arithmetic of its own
(confirmed by this file's own negative-grep acceptance criterion). The
browser's `datetime-local` input produces a zone-less local wall-clock
string (05-01-PLAN.md PA-03); this route hands that string, the server's
own resolved `server.timezone` (`request.app.state.server_timezone`), and
the current instant to `resolve_schedule`, and passes the absolute UTC
result straight to the repository.

`PUT /api/workflows/{run_id}` surfaces `WorkflowRepository.replace_steps`'s
own row-locked refusal (`WorkflowRunNotAppendableError`, D-11) as its own
distinguishable error rather than re-checking the run's status here and
then writing -- the check-then-act shape this project has now found twice
(WR-03, HI-01) and is not adding a third instance of. `replace_steps`
itself recomputes every step's `due_at` from the run's own existing first
step (never from `now`), so an edit that only changes the step list keeps
the run's original schedule start unchanged; there is no `DELETE` verb and
no delete call anywhere in this module (D-12) -- `POST
/api/workflows/{run_id}/cancel` is a status transition, and cancelling an
already-terminal run is reported distinguishably from cancelling a run that
never existed.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.db.repository import (
    WorkflowRepository,
    WorkflowRun,
    WorkflowRunNotAppendableError,
    WorkflowRunNotFoundError,
    WorkflowStep,
    WorkflowStepSpec,
)
from spire_voice.routes.conflict import ConflictAnnotation, annotate_conflict, known_entity_ids, load_policy_or_none
from spire_voice.workflow.schedule import ScheduleError, resolve_schedule
from spire_voice.workflow.steps import _transition_refusal

router = APIRouter(tags=["workflows"])

# D-16: the webapp's own pending-run list -- `WorkflowRepository.list_runs`'s
# own docstring names this exact pair as what the webapp passes.
_PENDING_RUN_STATUSES = ("pending", "firing")


def _workflow_not_found_error(run_id: int) -> HTTPException:
    return HTTPException(status_code=404, detail=f"no workflow run with id {run_id}")


def _workflow_already_terminal_error(run_id: int, status: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            f"workflow run {run_id} is already {status!r} -- cancelling it again would "
            "change nothing"
        ),
    )


def _workflow_not_appendable_error(run_id: int, status: str) -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=f"workflow run {run_id} can no longer be edited -- it is already {status!r}",
    )


def _zero_steps_error() -> HTTPException:
    return HTTPException(status_code=400, detail="add at least one step before saving")


def _schedule_time_in_past_error() -> HTTPException:
    return HTTPException(status_code=400, detail="pick a time in the future")


def _schedule_error(exc: ScheduleError) -> HTTPException:
    """`exc`'s own message already names, by name, which of
    `resolve_schedule`'s five rules refused the request (ambiguous,
    nonexistent, no configured zone, ...) -- carried through unchanged
    rather than reworded, the same convention `_duplicate_phrase_error`
    (`routes/macros.py`) already established for a lower-level function's
    own refusal text."""
    return HTTPException(status_code=400, detail=str(exc))


def _invalid_wait_duration_error(index: int) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"step at position {index}: a wait step's duration_s must be a non-negative number",
    )


def _missing_call_service_field_error(index: int, field: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"step at position {index}: a call_service step's arguments must carry a non-empty {field}",
    )


def _transition_refusal_error(index: int, reason: str) -> HTTPException:
    return HTTPException(status_code=400, detail=f"step at position {index}: {reason}")


def _blank_speak_text_error(index: int) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"step at position {index}: a speak step's words must not be blank",
    )


class WorkflowStepInput(BaseModel):
    kind: Literal["wait", "call_service", "speak"]
    arguments: dict = Field(default_factory=dict)


class CreateWorkflowRequest(BaseModel):
    summary: str = Field(
        min_length=1,
        description="What the operator would say to mean this run -- used later to cancel it by voice.",
    )
    run_at: str = Field(description="A datetime-local string (PA-03), or an ISO-8601 string with an explicit UTC offset.")
    steps: list[WorkflowStepInput] = Field(default_factory=list)


class ReplaceWorkflowStepsRequest(BaseModel):
    steps: list[WorkflowStepInput] = Field(default_factory=list)


class WorkflowStepResponse(BaseModel):
    id: int
    position: int
    kind: str
    arguments: dict
    due_at: datetime
    status: str
    attempts: int
    result_detail: dict | None
    fired_at: datetime | None
    # Advisory only -- see routes/conflict.py's own module docstring.
    # `unknown` is distinct from `ok` on purpose: a failed check must
    # never render as "no conflict."
    conflict: ConflictAnnotation


class WorkflowRunResponse(BaseModel):
    id: int
    origin: str
    status: str
    summary: str
    created_at: datetime
    updated_at: datetime
    created_by_user_id: int | None
    step_count: int
    # A still-pending step whose due_at has passed -- D-04's "and says so,"
    # made visible on the row that would otherwise look identical to an
    # on-time pending run.
    late: bool
    steps: list[WorkflowStepResponse]


def _validate_steps(steps: Sequence[WorkflowStepInput]) -> list[WorkflowStepSpec]:
    """Every refusal 05-UI-SPEC.md's Copywriting Contract names for a
    malformed step is reproduced here, each with its own named error --
    Phase 4's own review scored a Medium for a route that did not
    reproduce its parser's non-empty refusals; this function exists so
    that finding is not repeated here."""
    if not steps:
        raise _zero_steps_error()
    specs: list[WorkflowStepSpec] = []
    for index, step in enumerate(steps):
        arguments = step.arguments if isinstance(step.arguments, dict) else {}
        if step.kind == "wait":
            duration = arguments.get("duration_s")
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or duration < 0
            ):
                raise _invalid_wait_duration_error(index)
        elif step.kind == "call_service":
            domain = arguments.get("domain")
            service = arguments.get("service")
            if not isinstance(domain, str) or not domain:
                raise _missing_call_service_field_error(index, "domain")
            if not isinstance(service, str) or not service:
                raise _missing_call_service_field_error(index, "service")
            # The same light-only `transition` restriction
            # `workflow/steps.py::_execute_call_service` enforces a
            # second time at fire time (FLOW-03's second layer) -- caught
            # here too, at authoring time, so the operator learns
            # immediately rather than waiting for the step to fire and
            # be refused.
            refusal = _transition_refusal(arguments)
            if refusal is not None:
                raise _transition_refusal_error(index, refusal)
        elif step.kind == "speak":
            text = arguments.get("text")
            if not isinstance(text, str) or not text:
                raise _blank_speak_text_error(index)
        specs.append(WorkflowStepSpec(kind=step.kind, arguments=arguments))
    return specs


def _is_late(run: WorkflowRun, now: datetime) -> bool:
    return any(step.status == "pending" and step.due_at <= now for step in run.steps)


def _to_workflow_step_response(
    step: WorkflowStep, policy, known_entity_ids_snapshot: frozenset[str] | None
) -> WorkflowStepResponse:
    return WorkflowStepResponse(
        id=step.id,
        position=step.position,
        kind=step.kind,
        arguments=step.arguments,
        due_at=step.due_at,
        status=step.status,
        attempts=step.attempts,
        result_detail=step.result_detail,
        fired_at=step.fired_at,
        conflict=annotate_conflict(step.arguments, policy, known_entity_ids_snapshot),
    )


def _to_workflow_run_response(
    run: WorkflowRun,
    policy,
    known_entity_ids_snapshot: frozenset[str] | None,
    *,
    now: datetime,
) -> WorkflowRunResponse:
    return WorkflowRunResponse(
        id=run.id,
        origin=run.origin,
        status=run.status,
        summary=run.summary,
        created_at=run.created_at,
        updated_at=run.updated_at,
        created_by_user_id=run.created_by_user_id,
        step_count=len(run.steps),
        late=_is_late(run, now),
        steps=[_to_workflow_step_response(step, policy, known_entity_ids_snapshot) for step in run.steps],
    )


@router.get("/api/workflows")
async def list_workflows(
    request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> list[WorkflowRunResponse]:
    workflow_repo: WorkflowRepository = request.app.state.workflow_repo
    runs = await workflow_repo.list_runs(statuses=_PENDING_RUN_STATUSES)
    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    now = datetime.now(timezone.utc)
    return [_to_workflow_run_response(run, policy, known, now=now) for run in runs]


@router.get("/api/workflows/{run_id}")
async def get_workflow(
    run_id: int, request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> WorkflowRunResponse:
    workflow_repo: WorkflowRepository = request.app.state.workflow_repo
    run = await workflow_repo.get_run(run_id)
    if run is None:
        raise _workflow_not_found_error(run_id)
    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    now = datetime.now(timezone.utc)
    return _to_workflow_run_response(run, policy, known, now=now)


@router.post("/api/workflows", status_code=201)
async def create_workflow(
    payload: CreateWorkflowRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> WorkflowRunResponse:
    specs = _validate_steps(payload.steps)

    now = datetime.now(timezone.utc)
    zone = getattr(request.app.state, "server_timezone", None)
    try:
        due_at = resolve_schedule(delay_seconds=None, at=payload.run_at, now=now, zone=zone)
    except ScheduleError as exc:
        raise _schedule_error(exc) from exc
    if due_at <= now:
        raise _schedule_time_in_past_error()

    workflow_repo: WorkflowRepository = request.app.state.workflow_repo
    run = await workflow_repo.create_run(
        origin="webapp",
        summary=payload.summary,
        steps=specs,
        base_time=due_at,
        created_by_user_id=user.id,
    )
    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    return _to_workflow_run_response(run, policy, known, now=now)


@router.put("/api/workflows/{run_id}")
async def replace_workflow_steps(
    run_id: int,
    payload: ReplaceWorkflowStepsRequest,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> WorkflowRunResponse:
    specs = _validate_steps(payload.steps)
    workflow_repo: WorkflowRepository = request.app.state.workflow_repo
    now = datetime.now(timezone.utc)
    try:
        run = await workflow_repo.replace_steps(run_id, specs, now=now)
    except WorkflowRunNotFoundError as exc:
        raise _workflow_not_found_error(run_id) from exc
    except WorkflowRunNotAppendableError as exc:
        raise _workflow_not_appendable_error(run_id, exc.status) from exc
    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    return _to_workflow_run_response(run, policy, known, now=now)


@router.post("/api/workflows/{run_id}/cancel")
async def cancel_workflow(
    run_id: int, request: Request, user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> WorkflowRunResponse:
    workflow_repo: WorkflowRepository = request.app.state.workflow_repo
    existing = await workflow_repo.get_run(run_id)
    if existing is None:
        raise _workflow_not_found_error(run_id)

    now = datetime.now(timezone.utc)
    cancelled = await workflow_repo.cancel_run(run_id, now=now, cancelled_by_user_id=user.id)
    if not cancelled:
        # `cancel_run` itself does not distinguish "no such run" from
        # "already terminal" in its own return value (both are `False`) --
        # the `existing is None` check above already ruled out "no such
        # run," so a `False` reaching here can only mean "already terminal."
        raise _workflow_already_terminal_error(run_id, existing.status)

    updated = await workflow_repo.get_run(run_id)
    assert updated is not None  # cancel_run just returned True for this id
    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    return _to_workflow_run_response(updated, policy, known, now=now)
