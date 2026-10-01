"""Admin routes for paired Macs (Phase 14, D-01, D-03, D-15, D-28).

Mirrors `routes/edge_devices.py`: the plaintext Mac token is returned once,
in the create response, with `Cache-Control: no-store`. No other route ever
carries it or its hash. Every route sits behind `require_role(Role.ADMIN)`.
Online state comes from the live `DesktopHub`, never from the database.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request, Response
from pydantic import BaseModel, field_validator

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.auth.desktop_tokens import hash_desktop_token, issue_desktop_token
from atlas.db.desktop_repository import (
    DesktopDefaultConflict,
    DesktopDevice,
    DesktopDeviceChanges,
    DesktopDeviceNameTaken,
    DesktopDeviceRepository,
)
from atlas.desktop.hub import DesktopHub, DesktopNotConnected
from atlas.desktop.protocol import CLOSE_REVOKED

router = APIRouter(tags=["desktop-devices"])

# The same closed character set the edge device name uses.
_NAME_RE = re.compile(r"^[A-Za-z0-9 _.-]{1,64}$")


class DesktopDeviceCreateRequest(BaseModel):
    name: str

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str) -> str:
        trimmed = value.strip()
        if not _NAME_RE.match(trimmed):
            raise ValueError(
                "name must be 1-64 characters of letters, digits, space, '-', '_' and '.' "
                "(after trimming)"
            )
        return trimmed


class DesktopDeviceUpdateRequest(BaseModel):
    """A partial update. A key that is absent leaves its value alone, and
    `edge_device_id: null` clears the room mapping. The route tells the two
    apart with `model_fields_set`."""

    name: str | None = None
    edge_device_id: int | None = None
    is_default: bool | None = None

    @field_validator("name")
    @classmethod
    def _validate_name(cls, value: str | None) -> str:
        if value is None:
            raise ValueError("name cannot be null")
        trimmed = value.strip()
        if not _NAME_RE.match(trimmed):
            raise ValueError(
                "name must be 1-64 characters of letters, digits, space, '-', '_' and '.' "
                "(after trimming)"
            )
        return trimmed

    @field_validator("is_default")
    @classmethod
    def _validate_is_default(cls, value: bool | None) -> bool:
        if value is None:
            raise ValueError("is_default cannot be null")
        return value


class DesktopDeviceResponse(BaseModel):
    """No `token` field and no hash field, in any listing response."""

    id: int
    name: str
    created_at: datetime
    revoked: bool
    last_seen_at: "datetime | None"
    connected: bool
    edge_device_id: "int | None"
    is_default: bool


class DesktopDeviceCreatedResponse(BaseModel):
    """The one response that ever carries the plaintext Mac token."""

    id: int
    name: str
    token: str
    created_at: datetime


class DesktopTestResponse(BaseModel):
    answered: bool
    rtt_ms: "int | None"


def _to_response(device: DesktopDevice, *, connected: bool) -> DesktopDeviceResponse:
    return DesktopDeviceResponse(
        id=device.id,
        name=device.name,
        created_at=device.created_at,
        revoked=device.revoked_at is not None,
        last_seen_at=device.last_seen_at,
        connected=connected,
        edge_device_id=device.edge_device_id,
        is_default=device.is_default,
    )


@router.get("/api/desktop-devices")
async def list_desktop_devices(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[DesktopDeviceResponse]:
    repo: DesktopDeviceRepository = request.app.state.desktop_device_repo
    hub: DesktopHub = request.app.state.desktop_hub
    online = hub.snapshot()
    devices = await repo.list_devices()
    return [_to_response(device, connected=device.id in online) for device in devices]


@router.post("/api/desktop-devices", status_code=201)
async def create_desktop_device(
    payload: DesktopDeviceCreateRequest,
    request: Request,
    response: Response,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> DesktopDeviceCreatedResponse:
    """Issues a token, stores only its hash, and returns the plaintext token
    once. The response is never cached (D-03)."""
    response.headers["Cache-Control"] = "no-store"
    repo: DesktopDeviceRepository = request.app.state.desktop_device_repo
    token = issue_desktop_token()
    try:
        device = await repo.create_device(
            name=payload.name,
            token_hash=hash_desktop_token(token),
            created_by_user_id=admin.id,
            created_at=datetime.now(timezone.utc),
        )
    except DesktopDeviceNameTaken:
        raise HTTPException(
            status_code=409, detail="another active Mac already uses this name"
        ) from None
    return DesktopDeviceCreatedResponse(
        id=device.id, name=device.name, token=token, created_at=device.created_at
    )


@router.post("/api/desktop-devices/{device_id}/test")
async def run_desktop_device_test(
    device_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> DesktopTestResponse:
    """A silent round trip: the server pings the live Mac and waits for its
    pong. The Mac shows nothing (D-15). The name is not `test_*`, so pytest
    never collects it from a test module that imports it."""
    repo: DesktopDeviceRepository = request.app.state.desktop_device_repo
    hub: DesktopHub = request.app.state.desktop_hub
    if await repo.get_device(device_id) is None:
        raise HTTPException(status_code=404, detail="no such Mac")
    try:
        rtt_ms = await hub.ping(device_id)
    except DesktopNotConnected:
        raise HTTPException(status_code=409, detail="this Mac is not connected") from None
    if rtt_ms is None:
        return DesktopTestResponse(answered=False, rtt_ms=None)
    return DesktopTestResponse(answered=True, rtt_ms=int(round(rtt_ms)))


@router.patch("/api/desktop-devices/{device_id}")
async def update_desktop_device(
    device_id: int,
    payload: DesktopDeviceUpdateRequest,
    request: Request,
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> DesktopDeviceResponse:
    """Rename a Mac, map it to an edge device (its room), or make it the
    default (D-06, D-14, D-16, D-17). Setting a default clears the previous
    one inside the repository's single transaction."""
    sent = payload.model_fields_set
    if not sent:
        raise HTTPException(status_code=422, detail="nothing to change")
    repo: DesktopDeviceRepository = request.app.state.desktop_device_repo
    hub: DesktopHub = request.app.state.desktop_hub
    device = await repo.get_device(device_id)
    if device is None:
        raise HTTPException(status_code=404, detail="no such Mac")
    if device.revoked_at is not None:
        raise HTTPException(status_code=409, detail="this Mac is revoked")

    if "edge_device_id" in sent and payload.edge_device_id is not None:
        # Checked only when it is set. A mapping to an edge device that was
        # revoked later stays readable.
        edge_repo = getattr(request.app.state, "edge_device_repo", None)
        edge_devices = await edge_repo.list_devices() if edge_repo is not None else []
        if not any(
            edge.id == payload.edge_device_id and edge.revoked_at is None for edge in edge_devices
        ):
            raise HTTPException(status_code=422, detail="no such active edge device")

    changes = DesktopDeviceChanges(
        name=payload.name if "name" in sent else None,
        set_edge_device="edge_device_id" in sent,
        edge_device_id=payload.edge_device_id if "edge_device_id" in sent else None,
        is_default=payload.is_default if "is_default" in sent else None,
    )
    try:
        updated = await repo.update_device(device_id, changes, at=datetime.now(timezone.utc))
    except DesktopDeviceNameTaken:
        raise HTTPException(
            status_code=409, detail="another active Mac already uses this name"
        ) from None
    except DesktopDefaultConflict:
        raise HTTPException(
            status_code=409,
            detail="another Mac became the default at the same time. Try again.",
        ) from None
    if updated is None:
        raise HTTPException(status_code=409, detail="this Mac is revoked")
    return _to_response(updated, connected=hub.is_connected(device_id))


@router.delete("/api/desktop-devices/{device_id}", status_code=204)
async def revoke_desktop_device(
    device_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> Response:
    """Revoke a Mac (D-01, D-21). The row stays, with `revoked_at` set, so the
    list still shows it. A revoke of an already revoked Mac is a 204 no-op.
    The live socket is always closed with 4001, even when this call revoked
    nothing: the close is idempotent, so a socket that slipped through a
    race is closed too."""
    repo: DesktopDeviceRepository = request.app.state.desktop_device_repo
    hub: DesktopHub = request.app.state.desktop_hub
    if await repo.get_device(device_id) is None:
        raise HTTPException(status_code=404, detail="no such Mac")
    await repo.revoke_device(device_id, revoked_at=datetime.now(timezone.utc))
    await hub.disconnect_device(device_id, code=CLOSE_REVOKED, reason="revoked")
    return Response(status_code=204)
