"""Admin routes to create, list, and revoke edge devices, with revoke
taking effect on a live connection at once (D-03, Phase 10, plan 10-04).

Mirrors `routes/accounts.py`'s invite create/list/revoke triplet: a
plaintext device token is returned exactly once, in the create response,
and no other route ever sees it or its hash again. Every route here sits
behind `require_role(Role.ADMIN)`, the same as every invite route -- D-03
names an admin as the one who pairs and revokes a Pi.
"""

from __future__ import annotations

import re
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.auth.edge_tokens import hash_edge_token, issue_edge_token
from atlas.db.edge_repository import EdgeDevice, EdgeDeviceRepository

router = APIRouter(tags=["edge-devices"])

# 1-64 characters, a closed set: letters, digits, space, `-`, `_`, `.` --
# the same "closed character set at the route boundary" discipline
# `<threat_model>` (10-04-PLAN.md, T-10-13) names for this exact field.
_NAME_RE = re.compile(r"^[A-Za-z0-9 _.-]{1,64}$")


class EdgeDeviceCreateRequest(BaseModel):
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


class EdgeDeviceResponse(BaseModel):
    """No `token` field and no hash field -- every device-listing
    response, the same prohibition `InviteResponse` already carries
    (T-10-02)."""

    id: int
    name: str
    created_at: datetime
    revoked: bool
    last_connected_at: "datetime | None"
    connected: bool


class EdgeDeviceCreatedResponse(BaseModel):
    """The one response that ever carries the plaintext device token --
    never again from any route, per this module's own prohibition
    (T-10-02)."""

    id: int
    name: str
    token: str
    created_at: datetime


def _connected_device_id(request: Request) -> "int | None":
    edge_source = getattr(request.app.state, "edge_source", None)
    if edge_source is None:
        return None
    return edge_source.connected_device_id


def _to_response(device: EdgeDevice, *, connected_device_id: "int | None") -> EdgeDeviceResponse:
    return EdgeDeviceResponse(
        id=device.id,
        name=device.name,
        created_at=device.created_at,
        revoked=device.revoked_at is not None,
        last_connected_at=device.last_connected_at,
        connected=connected_device_id == device.id,
    )


def _no_such_device_error() -> HTTPException:
    return HTTPException(status_code=404, detail="no such edge device")


@router.get("/api/edge-devices")
async def list_edge_devices(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[EdgeDeviceResponse]:
    repo: EdgeDeviceRepository = request.app.state.edge_device_repo
    devices = await repo.list_devices()
    connected_device_id = _connected_device_id(request)
    return [_to_response(device, connected_device_id=connected_device_id) for device in devices]


@router.post("/api/edge-devices", status_code=201)
async def create_edge_device(
    payload: EdgeDeviceCreateRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> EdgeDeviceCreatedResponse:
    """Issues a token, stores only its hash, and returns the plaintext
    token exactly once, in this response (D-03) -- never returned again
    from any route."""
    repo: EdgeDeviceRepository = request.app.state.edge_device_repo
    token = issue_edge_token()
    now = datetime.now(timezone.utc)
    device = await repo.create_device(
        name=payload.name,
        token_hash=hash_edge_token(token),
        created_by_user_id=admin.id,
        created_at=now,
    )
    return EdgeDeviceCreatedResponse(
        id=device.id, name=device.name, token=token, created_at=device.created_at
    )


@router.delete("/api/edge-devices/{device_id}", status_code=204)
async def revoke_edge_device(
    device_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> None:
    """Sets `revoked_at`, never deletes the row (D-03). An unknown id is
    404; an already-revoked id is a no-op 204 -- `revoke_device`'s own
    rowcount-decides-truth return only tells the two apart from "the row
    exists at all," which this route checks via `list_devices` first.

    When the device being revoked is the one currently connected, this
    also closes that live connection at once (`EdgeAudioSource.
    disconnect_device`, close 1008 "revoked") -- a revoked Pi's socket
    does not merely fail its *next* reconnect (T-10-12).
    """
    repo: EdgeDeviceRepository = request.app.state.edge_device_repo
    devices = await repo.list_devices()
    if not any(device.id == device_id for device in devices):
        raise _no_such_device_error()

    newly_revoked = await repo.revoke_device(device_id, revoked_at=datetime.now(timezone.utc))

    edge_source = getattr(request.app.state, "edge_source", None)
    if newly_revoked and edge_source is not None and edge_source.connected_device_id == device_id:
        await edge_source.disconnect_device(device_id, code=1008, reason="revoked")
