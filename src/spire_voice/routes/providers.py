"""Provider choice over HTTP: read every served slot's stored and running
provider, and save a new choice per slot (D-01 .. D-04, PROV-01).

Every route here requires `Role.ADMIN`, the same guard band as `/settings`
and `/plugins` (07-UI-SPEC.md). Built on `routes/plugins.py`'s own shape:
one named-error helper per distinct refusal, and a response that reads
live facts fresh at response-build time rather than a stored copy of them
(`_to_plugin_response`'s own convention, applied here to `selected`/
`active` instead of `state`/`reason`).

Plan 07-02: all three slots are served now, in the fixed order
07-UI-SPEC.md pins -- speech to text, text to speech, language model --
so the admin's eye lands on the same slot in the same place after every
restart. `_SERVED_SLOTS` is still the one tuple every route below reads;
extending it to a fourth slot is a tuple edit, never a second copy of the
routes.
"""

from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.crypto.credentials import CredentialSlot, resolve_credential_source
from spire_voice.db.repository import ProviderSelectionRepository
from spire_voice.providers import registry
from spire_voice.providers.boot import SLOT_LABELS, SLOT_ORDER, ProviderSlotStatus

router = APIRouter(tags=["providers"])

# The fixed order 07-UI-SPEC.md pins for the /providers screen: speech to
# text, text to speech, language model. `GET`/`PUT /api/providers` report
# and accept exactly the slots named here, in this order.
#
# CR-03 (code review): the order and the labels both live on
# `providers/boot.py` now, because a refused turn names the same slots in
# the same words this screen shows. Two copies of "Speech to text" would
# be two things to keep in step forever.
_SERVED_SLOTS: "tuple[str, ...]" = SLOT_ORDER

_SLOT_LABELS: "dict[str, str]" = SLOT_LABELS

_CREDENTIAL_SLOT_FOR: "dict[str, CredentialSlot]" = {
    "stt": CredentialSlot.STT,
    "tts": CredentialSlot.TTS,
    "brain": CredentialSlot.BRAIN,
}


# --- Named refusals (`routes/plugins.py`'s own house convention) ----------


def _unknown_slot_error(slot: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"{slot!r} is not a provider slot this server serves -- known slots: {list(_SERVED_SLOTS)!r}",
    )


def _unknown_provider_error(slot: str, name: str) -> HTTPException:
    known = registry.known_names(slot)
    return HTTPException(
        status_code=400,
        detail=f"provider {name!r} is not registered for slot {slot!r} -- known providers: {known!r}",
    )


# --- Request/response models ----------------------------------------------


class ProviderOptionResponse(BaseModel):
    name: str
    label: str
    requires_credential: bool
    # Resolved through `resolve_credential_source` -- the same
    # database-then-environment-then-unset precedence Phase 3 already
    # established, never a second resolver (Task 2).
    credential_set: bool
    wrapped: bool
    licence_note: "str | None" = None
    # `ProviderEntry.needs_server_url`/`.measured_note` (07-04-PLAN.md),
    # carried through unchanged -- data on the entry, never a hardcoded
    # provider name, so the screen reveals a "Server URL" field or a
    # local-set latency caption from a flag rather than a name check
    # (plan 07-06 wires these two fields through for the first time).
    needs_server_url: bool = False
    measured_note: "str | None" = None


class ProviderSlotResponse(BaseModel):
    slot: str
    label: str
    # A fresh repository read, never a stored copy (D-02's own contract).
    selected: str
    # `None` for a degraded slot -- the process built no client at all.
    active: "str | None"
    state: str
    reason: "str | None" = None
    wrapped: bool
    # D-08's honest synthesis-to-first-chunk timing (plan 07-02): a live
    # read off the slot's active client at response-build time, never
    # stored on this response's own source of truth and never persisted.
    # `None` for an unwrapped slot, for a degraded slot with no client at
    # all, and for a wrapped slot that has not synthesized anything since
    # the last restart -- three honest reasons for the same absent value.
    measured_ms: "float | None" = None
    options: list[ProviderOptionResponse]
    # The selection row's own `options` JSON column, renamed on the wire
    # to avoid colliding with this response's own `options` (the list of
    # selectable providers) -- the home for the language-model slot's
    # local server URL (plan 07-04), always `{}` this plan.
    settings: dict
    # WR-08 (code review): whether the stored selection (the provider
    # name OR its settings) has moved since this process built the slot.
    #
    # `selected != active` was the only "needs restart" signal, and it
    # cannot fire for a degraded slot, whose `active` is `None` -- yet a
    # degraded slot is exactly the one an admin has just changed. An
    # admin who filled in the missing server URL and saved saw
    # `needsRestart` false and the OLD failure reason still rendered
    # beside the field they had just filled in: their fix looked like it
    # had not taken. The server has the data to answer this properly;
    # the screen was guessing from a comparison that could not.
    selection_changed_since_boot: bool = False


class ProvidersResponse(BaseModel):
    slots: list[ProviderSlotResponse]


class SetProviderSlotInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    provider_name: str
    settings: dict = Field(default_factory=dict)


class SetProvidersRequest(BaseModel):
    """Every slot in one body (07-UI-SPEC.md's probe addendum): one
    request, one outcome, no per-slot save."""

    model_config = ConfigDict(extra="forbid")

    slots: "dict[str, SetProviderSlotInput]"


class ProvidersWriteResponse(ProvidersResponse):
    # Stated fact, never a browser guess (D-02, matching
    # `routes/plugins.py`'s own `PluginResponse.applies_live` comment) --
    # the one write in this codebase whose honest answer is `False`.
    applies_live: bool = False


# --- Helpers ---------------------------------------------------------------


async def _option_response(request: Request, slot: str, entry: "registry.ProviderEntry") -> ProviderOptionResponse:
    credential_repo = request.app.state.credential_repo
    config = request.app.state.config
    credential_slot = _CREDENTIAL_SLOT_FOR[slot]
    is_set, _source, _updated_at = await resolve_credential_source(credential_slot, credential_repo, config)
    return ProviderOptionResponse(
        name=entry.name,
        label=entry.label,
        requires_credential=entry.requires_credential,
        credential_set=is_set,
        wrapped=entry.batch,
        licence_note=entry.licence_note,
        needs_server_url=entry.needs_server_url,
        measured_note=entry.measured_note,
    )


def _measured_ms_for(request: Request, slot: str) -> "float | None":
    """D-08's honest timing, read live off the slot's active client --
    never a stored copy. `getattr(request.app.state, slot, None)` is the
    exact attribute `lifespan` assigns each slot's client under
    (`app.state.stt`/`app.state.tts`/`app.state.brain`); a client with no
    `last_synthesis_ms` of its own (every unwrapped provider) or no
    client at all (a degraded slot) both read as `None` here, the same
    honest absence as a wrapped slot that has synthesized nothing yet."""
    client = getattr(request.app.state, slot, None)
    if client is None:
        return None
    return getattr(client, "last_synthesis_ms", None)


async def _slot_response(request: Request, slot: str, selection, status: "ProviderSlotStatus | None") -> ProviderSlotResponse:
    options = [await _option_response(request, slot, entry) for entry in registry.known_entries(slot)]
    selected = selection.provider_name if selection is not None else ""
    settings = selection.options if selection is not None else {}

    if status is not None:
        active = status.active
        state = status.state
        reason = status.reason
        wrapped = status.wrapped
        # Both halves of the stored selection, against what the boot
        # actually used. A slot with no row at all at boot and no row now
        # has not changed; a row that appeared since has.
        changed_since_boot = selected != status.selected or settings != status.booted_settings
    else:
        # Transitional read, the same "no manager entry recorded yet"
        # fallback `_to_plugin_response`'s own `resolved_state` uses --
        # never expected once `lifespan` has run for this slot.
        active = None
        state = "starting"
        reason = None
        wrapped = False
        changed_since_boot = False

    return ProviderSlotResponse(
        slot=slot,
        label=_SLOT_LABELS[slot],
        selected=selected,
        active=active,
        state=state,
        reason=reason,
        wrapped=wrapped,
        measured_ms=_measured_ms_for(request, slot),
        options=options,
        settings=settings,
        selection_changed_since_boot=changed_since_boot,
    )


async def _providers_response(request: Request) -> ProvidersResponse:
    provider_selection_repo: ProviderSelectionRepository = request.app.state.provider_selection_repo
    provider_slots: "dict[str, ProviderSlotStatus]" = getattr(request.app.state, "provider_slots", {})
    selections_by_slot = {selection.slot: selection for selection in await provider_selection_repo.list_selections()}

    slots = [
        await _slot_response(request, slot, selections_by_slot.get(slot), provider_slots.get(slot))
        for slot in _SERVED_SLOTS
    ]
    return ProvidersResponse(slots=slots)


# --- Routes -----------------------------------------------------------


@router.get("/api/providers")
async def list_providers(
    request: Request, _user: CurrentUser = Depends(require_role(Role.ADMIN))
) -> ProvidersResponse:
    return await _providers_response(request)


@router.put("/api/providers")
async def set_providers(
    payload: SetProvidersRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> ProvidersWriteResponse:
    provider_selection_repo: ProviderSelectionRepository = request.app.state.provider_selection_repo

    # Validate every submitted slot before writing anything at all -- a
    # partial write across a multi-slot save is the state 07-UI-SPEC.md's
    # probe addendum explicitly refuses to render (Task 3).
    for slot, submitted in payload.slots.items():
        if slot not in _SERVED_SLOTS:
            raise _unknown_slot_error(slot)
        if submitted.provider_name not in registry.known_names(slot):
            raise _unknown_provider_error(slot, submitted.provider_name)

    now = datetime.now(timezone.utc)
    for slot, submitted in payload.slots.items():
        await provider_selection_repo.set_selection(
            slot,
            submitted.provider_name,
            submitted.settings,
            updated_by_user_id=user.id,
            updated_at=now,
        )

    response = await _providers_response(request)
    return ProvidersWriteResponse(slots=response.slots)
