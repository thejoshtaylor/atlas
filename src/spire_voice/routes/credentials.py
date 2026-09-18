"""Provider credentials: write-only from the browser, encrypted at rest
(PROV-04, D-07).

Every route here is behind `require_role(Role.ADMIN)`. The list route
returns one entry per slot in the closed set (`CredentialSlot`) -- the
slot, a label, whether a value is set, when it was last updated, and
where the effective value currently comes from -- and never a
ciphertext, a plaintext, or any prefix or suffix of either. The write
route encrypts, upserts, records who did it, and returns the same
listing shape, so the browser learns the write landed without ever
receiving the value back.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.config import Config
from spire_voice.crypto.credentials import (
    SLOT_LABELS,
    CredentialSlot,
    encrypt_credential,
    resolve_credential_source,
)
from spire_voice.db.repository import CredentialRepository

router = APIRouter(tags=["credentials"])


def _unknown_slot_error(slot: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"unknown credential slot {slot!r} -- valid slots are {sorted(s.value for s in CredentialSlot)!r}",
    )


class CredentialListEntry(BaseModel):
    """No ciphertext, no plaintext, no prefix or suffix of either --
    this plan's own prohibition, asserted by a test enumerating every
    registered route."""

    slot: str
    label: str
    is_set: bool
    updated_at: datetime | None
    source: str  # "database" | "environment" | "unset"


class CredentialWriteRequest(BaseModel):
    value: str


async def _to_entry(slot: CredentialSlot, repo: CredentialRepository, config: Config) -> CredentialListEntry:
    is_set, source, updated_at = await resolve_credential_source(slot, repo, config)
    return CredentialListEntry(
        slot=slot.value, label=SLOT_LABELS[slot], is_set=is_set, updated_at=updated_at, source=source
    )


@router.get("/api/credentials")
async def list_credentials(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[CredentialListEntry]:
    repo: CredentialRepository = request.app.state.credential_repo
    config: Config = request.app.state.config
    return [await _to_entry(slot, repo, config) for slot in CredentialSlot]


@router.put("/api/credentials/{slot}")
async def write_credential(
    slot: str,
    payload: CredentialWriteRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> CredentialListEntry:
    try:
        credential_slot = CredentialSlot(slot)
    except ValueError:
        raise _unknown_slot_error(slot) from None

    repo: CredentialRepository = request.app.state.credential_repo
    config: Config = request.app.state.config
    ciphertext, key_version = encrypt_credential(payload.value, config.security)
    await repo.upsert_credential(
        credential_slot.value,
        ciphertext=ciphertext,
        key_version=key_version,
        updated_by_user_id=admin.id,
    )
    return await _to_entry(credential_slot, repo, config)
