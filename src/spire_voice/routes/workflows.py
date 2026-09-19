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
the check (see that module's own docstring for the full reasoning). Plan
06-05 (D-12) reuses that same call for a `call_service` step's own bare
tool name (always `ha_call_service`, `workflow/steps.py::
_HA_CALL_SERVICE_TOOL`) becoming ambiguous -- a `wait`/`speak` step passes
no `tool_name` at all, since neither kind ever calls a plugin's tool.

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

PA-01: a `speak` step's words ARE precached, the same `precache_all`
function `app.py`'s own startup precache and `routes/macros.py`'s own save
path both use -- after a successful `POST`/`PUT` commits, every `speak`
step's words not already in `app.state.filler_cache` are synthesized
before the route returns. A synthesis failure is a degraded success, never
a failed save (the run's rows are already committed by the time this
runs); every response that carries a `speak` step -- reads included --
reports whether its words are in the cache right now, computed live from
cache membership, with no new column: the exact keying rule
`routes/macros.py`'s own `reply_cached` already uses, extended verbatim to
a step (04-UI-SPEC.md's precache-state contract, per 05-UI-SPEC.md's own
unresolved row).
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Literal, Mapping, Sequence

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
from spire_voice.providers.tts_cache import precache_all
from spire_voice.routes.conflict import (
    ConflictAnnotation,
    annotate_conflict,
    known_entity_ids,
    load_policy_or_none,
    tool_owners_for,
)
from spire_voice.workflow.schedule import ScheduleError, resolve_schedule
from spire_voice.workflow.steps import _HA_CALL_SERVICE_TOOL, _transition_refusal

router = APIRouter(tags=["workflows"])

# The exact wording 05-UI-SPEC.md's precache-state contract (extended from
# 04-UI-SPEC.md's "Error state -- reply not synthesized" row) implies for a
# scheduled step -- named at module level so `_finish_workflow_save` below
# and any test asserting on it read the same string.
_SPEAK_SYNTHESIS_DEGRADED_MESSAGE = (
    "One or more spoken steps couldn't be prepared for fast playback. The run will "
    "still fire on schedule, but a spoken step may need to synthesize speech at the "
    "moment it runs."
)

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
        detail=f"step at position {index}: a wait step's duration_s must be a positive number",
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
    # PA-01: whether a `speak` step's own words are currently in the same
    # cache a turn reads from -- computed live from cache membership on
    # every response, reads included, the same `reply_cached` keying rule
    # `routes/macros.py`'s `MacroResponse` already uses. `False` for every
    # non-`speak` step -- there is nothing to report about a step with no
    # words to synthesize.
    speak_cached: bool


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
    # Only meaningful on a create/update response -- `False`/`None` on
    # every list/read/cancel response, since none of those re-synthesize
    # anything (mirrors `MacroResponse.reply_synthesis_degraded`).
    reply_synthesis_degraded: bool = False
    reply_synthesis_message: str | None = None


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
            # IN-01 (code review): `> 0`, matching `WorkflowStepEntry`'s
            # own voice-tool validation (`workflow/tool.py`) and its own
            # description ("a positive number of seconds") -- a zero
            # duration was previously accepted here and refused there,
            # which meant the webapp could author a `wait` step a spoken
            # sentence would be refused for building.
            if (
                not isinstance(duration, (int, float))
                or isinstance(duration, bool)
                or duration <= 0
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


def _get_filler_cache(request: Request) -> dict:
    """The same `app.state.filler_cache` dict a turn reads from
    (`turn/controller.py`) -- created empty here if a throwaway test app
    never set one, matching `routes/macros.py::_finish_save`'s own
    convention exactly."""
    filler_cache = getattr(request.app.state, "filler_cache", None)
    if filler_cache is None:
        filler_cache = {}
        request.app.state.filler_cache = filler_cache
    return filler_cache


def _to_workflow_step_response(
    step: WorkflowStep,
    policy,
    known_entity_ids_snapshot: frozenset[str] | None,
    filler_cache: Mapping[str, bytes],
    tool_owners: "Callable[[str], tuple[str, ...]]",
) -> WorkflowStepResponse:
    text = step.arguments.get("text") if step.kind == "speak" and isinstance(step.arguments, dict) else None
    # Only a `call_service` step ever names a plugin's tool at all (always
    # `_HA_CALL_SERVICE_TOOL`, `workflow/steps.py`'s own fixed constant) --
    # `wait`/`speak` pass `tool_name=None`, which `annotate_conflict`
    # treats exactly like a caller that predates plan 06-05.
    tool_name = _HA_CALL_SERVICE_TOOL if step.kind == "call_service" else None
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
        conflict=annotate_conflict(
            step.arguments,
            policy,
            known_entity_ids_snapshot,
            tool_name=tool_name,
            tool_owners=tool_owners,
        ),
        speak_cached=text is not None and text in filler_cache,
    )


def _to_workflow_run_response(
    run: WorkflowRun,
    policy,
    known_entity_ids_snapshot: frozenset[str] | None,
    filler_cache: Mapping[str, bytes],
    tool_owners: "Callable[[str], tuple[str, ...]]",
    *,
    now: datetime,
    reply_synthesis_degraded: bool = False,
    reply_synthesis_message: str | None = None,
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
        steps=[
            _to_workflow_step_response(step, policy, known_entity_ids_snapshot, filler_cache, tool_owners)
            for step in run.steps
        ],
        reply_synthesis_degraded=reply_synthesis_degraded,
        reply_synthesis_message=reply_synthesis_message,
    )


async def _finish_workflow_save(request: Request, run: WorkflowRun) -> WorkflowRunResponse:
    """After a successful create/`PUT` commit, make sure every `speak`
    step's words are in the same cache a turn reads from before this
    route answers (PA-01) -- through `precache_all`, the exact function
    `app.py`'s own startup precache and `routes/macros.py`'s own save path
    both use. Skipped per-text whenever that text is already cached, the
    same rule that both makes a reorder-only save cheap and makes a retry
    after a prior failure actually retry (a failed synthesis never writes
    its text into the cache).

    A synthesis failure is a degraded success, never a failed save: the
    run's rows are already committed by the time this runs, so the
    response reports a degraded flag and a message rather than raising --
    the same reasoning `routes/macros.py::_finish_save` already carries.
    """
    filler_cache = _get_filler_cache(request)

    speak_texts = [
        step.arguments.get("text", "")
        for step in run.steps
        if step.kind == "speak" and isinstance(step.arguments, dict)
    ]
    to_synthesize = [text for text in speak_texts if text and text not in filler_cache]

    degraded = False
    message: str | None = None
    if to_synthesize:
        config = request.app.state.config
        try:
            new_entries = await precache_all(
                request.app.state.tts,
                Path(config.tts.cache_dir),
                to_synthesize,
                config.tts.voice_id,
                request.app.state.tts.browser_sink(),
            )
            filler_cache.update(new_entries)
        except Exception:  # noqa: BLE001 -- any synthesis failure degrades, never loses the edit
            degraded = True
            message = _SPEAK_SYNTHESIS_DEGRADED_MESSAGE

    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    tool_owners = tool_owners_for(request)
    now = datetime.now(timezone.utc)
    return _to_workflow_run_response(
        run,
        policy,
        known,
        filler_cache,
        tool_owners,
        now=now,
        reply_synthesis_degraded=degraded,
        reply_synthesis_message=message,
    )


@router.get("/api/workflows")
async def list_workflows(
    request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> list[WorkflowRunResponse]:
    workflow_repo: WorkflowRepository = request.app.state.workflow_repo
    runs = await workflow_repo.list_runs(statuses=_PENDING_RUN_STATUSES)
    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    filler_cache = _get_filler_cache(request)
    tool_owners = tool_owners_for(request)
    now = datetime.now(timezone.utc)
    return [
        _to_workflow_run_response(run, policy, known, filler_cache, tool_owners, now=now) for run in runs
    ]


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
    filler_cache = _get_filler_cache(request)
    tool_owners = tool_owners_for(request)
    now = datetime.now(timezone.utc)
    return _to_workflow_run_response(run, policy, known, filler_cache, tool_owners, now=now)


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
    return await _finish_workflow_save(request, run)


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
    return await _finish_workflow_save(request, run)


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
        #
        # WR-02 (code review): the status named in the 409 body is read
        # again here, not taken from `existing` above -- `existing` was
        # read before `cancel_run` ran, unlocked and uncoordinated with
        # the poller, so the run's real status can have moved on again
        # (e.g. from `firing` to `completed`) in the window between that
        # read and this one. `cancel_run` itself is unaffected by this
        # fix -- its own atomic `UPDATE ... WHERE status = 'pending'` was
        # already correct; only this error message's own snapshot was
        # stale.
        current = await workflow_repo.get_run(run_id)
        status = current.status if current is not None else existing.status
        raise _workflow_already_terminal_error(run_id, status)

    updated = await workflow_repo.get_run(run_id)
    assert updated is not None  # cancel_run just returned True for this id
    policy = await load_policy_or_none(request)
    known = await known_entity_ids(request)
    filler_cache = _get_filler_cache(request)
    tool_owners = tool_owners_for(request)
    return _to_workflow_run_response(updated, policy, known, filler_cache, tool_owners, now=now)
