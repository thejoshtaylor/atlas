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

One slot is not a `credentials` row: `CredentialSlot.HOME_ASSISTANT` is
stored as the Home Assistant plugin's own `HA_TOKEN` configuration value,
because that is the one place anything reads it from (WR-06, code review
-- see the comment on `_HOME_ASSISTANT_PLUGIN_SLUG` below for what the
alternative actually cost an operator). Everything else about the slot is
unchanged: same route, same response shape, same write-only property,
same encryption and key version.
"""

from __future__ import annotations

from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.config import Config
from atlas.crypto.credentials import (
    SLOT_LABELS,
    CredentialSlot,
    encrypt_credential,
    resolve_credential_source,
)
from atlas.db.repository import CredentialRepository, PluginConfigValue, PluginRepository

router = APIRouter(tags=["credentials"])

# Stated fact, not a browser guess (Task 3, D-03): every provider this
# application builds from a credential slot is constructed once, in
# `app.py`'s `lifespan`, and nothing rebuilds it on a later database
# write -- unlike the safety policy, where the same write respawns the
# enforcing child before the route returns. A credential entered here
# needs a restart to take effect.
_APPLIES_LIVE = False

# WR-06 (code review): the Home Assistant token is no longer a row in the
# `credentials` table at all -- plan 06-01 (D-01) moved it into
# `plugin_config_values`, where `PluginManager` reads it to spawn the child
# and `routes/wizard.py` reads it for the hub check. The slot itself stayed
# behind: `GET /api/credentials` still listed it, `PUT /api/credentials/
# ha_token` still encrypted and stored a value, and nothing anywhere read
# that value again -- so an operator rotating the token on the credentials
# screen (or entering it in the setup wizard's own hub step, which posts to
# exactly this route) saved it into a table no code consults, and was told
# a restart would apply it. No restart would.
#
# Rather than leave two places a Home Assistant token can be written and
# one place it is read, this slot's *storage* is the plugin row: the two
# operator surfaces keep working unchanged, and the value lands where it
# is actually read. `applies_live` stays `False` for it, which is honest
# -- the running child keeps the token it was spawned with until it is
# restarted (the `/plugins` editor is the surface that restarts it, D-15).
_HOME_ASSISTANT_PLUGIN_SLUG = "ha"
_HOME_ASSISTANT_TOKEN_KEY = "HA_TOKEN"


def _home_assistant_not_installed_error() -> HTTPException:
    return HTTPException(
        status_code=409,
        detail=(
            "there is no Home Assistant plugin to store a token for -- install it "
            "from /plugins first, and its token is part of its configuration there"
        ),
    )


async def _ha_token_entry(repo: PluginRepository) -> CredentialListEntry:
    """The Home Assistant slot's listing entry, read from the plugin row
    that actually holds the token (WR-06). `updated_at` is `None`: a
    plugin configuration value carries no per-value timestamp of its own,
    so this reports "set" without inventing a time it was set at."""
    stored = await _ha_token_value(repo)
    is_set = stored is not None and stored.ciphertext is not None
    return CredentialListEntry(
        slot=CredentialSlot.HOME_ASSISTANT.value,
        label=SLOT_LABELS[CredentialSlot.HOME_ASSISTANT],
        is_set=is_set,
        updated_at=None,
        source="database" if is_set else "unset",
        applies_live=_APPLIES_LIVE,
    )


async def _ha_plugin_id(repo: PluginRepository) -> "int | None":
    for plugin in await repo.list_plugins():
        if plugin.slug == _HOME_ASSISTANT_PLUGIN_SLUG:
            return plugin.id
    return None


async def _ha_token_value(repo: PluginRepository) -> "PluginConfigValue | None":
    plugin_id = await _ha_plugin_id(repo)
    if plugin_id is None:
        return None
    for value in await repo.get_config_values(plugin_id):
        if value.key == _HOME_ASSISTANT_TOKEN_KEY:
            return value
    return None


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
    applies_live: bool


class CredentialWriteRequest(BaseModel):
    value: str


async def _to_entry(slot: CredentialSlot, repo: CredentialRepository, config: Config) -> CredentialListEntry:
    is_set, source, updated_at = await resolve_credential_source(slot, repo, config)
    return CredentialListEntry(
        slot=slot.value,
        label=SLOT_LABELS[slot],
        is_set=is_set,
        updated_at=updated_at,
        source=source,
        applies_live=_APPLIES_LIVE,
    )


@router.get("/api/credentials")
async def list_credentials(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[CredentialListEntry]:
    repo: CredentialRepository = request.app.state.credential_repo
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    config: Config = request.app.state.config
    return [
        await (
            _ha_token_entry(plugin_repo)
            if slot is CredentialSlot.HOME_ASSISTANT
            else _to_entry(slot, repo, config)
        )
        for slot in CredentialSlot
    ]


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
    if credential_slot is CredentialSlot.HOME_ASSISTANT:
        # WR-06: written where it is read -- the Home Assistant plugin's
        # own `HA_TOKEN` configuration value, encrypted exactly as a
        # credential row would be (the same `encrypt_credential`, the same
        # key version), never into the `credentials` table nothing reads.
        plugin_repo: PluginRepository = request.app.state.plugin_repo
        plugin_id = await _ha_plugin_id(plugin_repo)
        if plugin_id is None:
            raise _home_assistant_not_installed_error()
        ciphertext, key_version = encrypt_credential(payload.value, config.security)
        await plugin_repo.set_config_values(
            plugin_id,
            [
                PluginConfigValue(
                    key=_HOME_ASSISTANT_TOKEN_KEY,
                    secret=True,
                    value=None,
                    ciphertext=ciphertext,
                    key_version=key_version,
                )
            ],
        )
        return await _ha_token_entry(plugin_repo)

    ciphertext, key_version = encrypt_credential(payload.value, config.security)
    await repo.upsert_credential(
        credential_slot.value,
        ciphertext=ciphertext,
        key_version=key_version,
        updated_by_user_id=admin.id,
    )
    return await _to_entry(credential_slot, repo, config)
