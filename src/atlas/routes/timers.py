"""Operator routes to list, create, read, update and delete timers and alarms.

Every route needs Role.OPERATOR, the same as `routes/workflows.py`. The
routes call `atlas.timers.service`, so the limits and the time math are the
ones the voice tools use. A refusal is 400, a missing id is 404 and the entry
limit is 409.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.db.timer_repository import Timer, TimerRepository
from atlas.timers import service
from atlas.timers.core import (
    AlarmSpec,
    TimerChanges,
    TimerError,
    TimerLimitError,
    TimerNotFoundError,
    TimerSpec,
    mask_to_days,
    remaining_seconds,
)

router = APIRouter(tags=["timers"])


class TimerCreate(TimerSpec):
    kind: Literal["timer"]


class AlarmCreate(AlarmSpec):
    kind: Literal["alarm"]


class TimerResponse(BaseModel):
    id: int
    kind: str
    label: str
    enabled: bool
    paused: bool
    duration_seconds: "int | None"
    remaining_seconds: "int | None"
    time: "str | None"
    days: list[str]
    next_fire_at: "datetime | None"
    created_at: datetime


def _repo(request: Request) -> TimerRepository:
    repo = getattr(request.app.state, "timer_repo", None)
    if repo is None:
        raise HTTPException(status_code=503, detail="timers are not available")
    return repo


def _to_response(timer: Timer, now: datetime) -> TimerResponse:
    return TimerResponse(
        id=timer.id,
        kind=timer.kind,
        label=timer.label,
        enabled=timer.enabled,
        paused=timer.kind == "timer" and timer.due_at is None,
        duration_seconds=timer.duration_s,
        remaining_seconds=remaining_seconds(timer, now),
        time=timer.time_of_day,
        days=mask_to_days(timer.repeat_days) if timer.kind == "alarm" else [],
        next_fire_at=timer.due_at,
        created_at=timer.created_at,
    )


def _http_error(exc: TimerError) -> HTTPException:
    # TimerNotFoundError and TimerLimitError are TimerError subclasses, so
    # they are checked first.
    if isinstance(exc, TimerNotFoundError):
        return HTTPException(status_code=404, detail=str(exc))
    if isinstance(exc, TimerLimitError):
        return HTTPException(status_code=409, detail=str(exc))
    return HTTPException(status_code=400, detail=str(exc))


@router.get("/api/timers")
async def list_timers(
    request: Request, _operator: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> list[TimerResponse]:
    now = datetime.now(timezone.utc)
    return [_to_response(timer, now) for timer in await _repo(request).list_timers()]


@router.get("/api/timers/{timer_id}")
async def get_timer(
    timer_id: int, request: Request, _operator: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> TimerResponse:
    timer = await _repo(request).get_timer(timer_id)
    if timer is None:
        raise HTTPException(status_code=404, detail=f"there is no timer or alarm with id {timer_id}")
    return _to_response(timer, datetime.now(timezone.utc))


@router.post("/api/timers", status_code=201)
async def create_timer(
    payload: Annotated[TimerCreate | AlarmCreate, Field(discriminator="kind")],
    request: Request,
    _operator: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> TimerResponse:
    repo = _repo(request)
    now = datetime.now(timezone.utc)
    try:
        if isinstance(payload, TimerCreate):
            timer = await service.create_timer(
                repo, TimerSpec(duration_seconds=payload.duration_seconds, label=payload.label), now=now
            )
        else:
            timer = await service.create_alarm(
                repo,
                AlarmSpec(time=payload.time, days=payload.days, label=payload.label),
                now=now,
                zone=getattr(request.app.state, "server_timezone", None),
            )
    except TimerError as exc:
        raise _http_error(exc) from exc
    return _to_response(timer, now)


@router.patch("/api/timers/{timer_id}")
async def update_timer(
    timer_id: int,
    changes: TimerChanges,
    request: Request,
    _operator: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> TimerResponse:
    now = datetime.now(timezone.utc)
    try:
        timer = await service.update_timer(
            _repo(request),
            timer_id,
            changes,
            now=now,
            zone=getattr(request.app.state, "server_timezone", None),
        )
    except TimerError as exc:
        raise _http_error(exc) from exc
    return _to_response(timer, now)


@router.delete("/api/timers/{timer_id}", status_code=204)
async def delete_timer(
    timer_id: int, request: Request, _operator: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> None:
    try:
        await service.delete_timer(_repo(request), timer_id)
    except TimerError as exc:
        raise _http_error(exc) from exc
