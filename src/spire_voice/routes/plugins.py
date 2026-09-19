"""Plugin authoring over HTTP: list, read, install, enable/disable,
configure, and delete a plugin (D-01 .. D-16, PLUG-03, PLUG-04, PLUG-08).

Every route here requires `Role.ADMIN` (D-14): plugin configuration is an
admin capability, per Phase 3's own decision, and an admin-entered command
is an arbitrary-execution surface by design (T-06-25) -- the admin role is
what bounds it, and nothing else.

Built on `routes/macros.py`'s own shape, with one deliberate change beyond
the role: every refusal here has its own named factory, matching this
codebase's house convention (`routes/macros.py`'s own module docstring
names it "the house convention for a refusal"). A secret configuration
value never appears in any response model this module returns (D-03,
T-06-27) -- `_to_config_value_response` below is the one place that
guarantee is enforced, and it is enforced by never reading `ciphertext`
into a response field, not by masking a value after the fact.

No route in this module ever accepts or writes `enforces_policy` (T-06-28):
`create_plugin` always inserts `False`, and no write method on
`PluginRepository` even has a parameter that could set it -- an admin
cannot hand the house denylist to a plugin they installed, structurally,
not by a runtime check that could be bypassed by a request shape this
module never anticipated. Every request model below carries
`model_config = ConfigDict(extra="forbid")` for the identical reason: a
request body naming a field this module does not accept is a 422, not a
silently-ignored write.

Plan 06-06, Task 3 (D-15) adds the live half beside this file's own reads
and writes: every install, enable, disable, and configuration save calls
into `PluginManager` and awaits it before the route returns, per
`routes/policy.py`'s own "a write that cannot take live effect is a
failed write" rule -- see `_reconcile_live_state` below.
"""

from __future__ import annotations

import re
import shlex
from typing import Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field

from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.crypto.credentials import encrypt_credential
from spire_voice.db.repository import Plugin, PluginAlreadyExistsError, PluginConfigValue, PluginRepository
from spire_voice.plugins.catalog import DEFAULT_CATALOG_PATH, find_entry, load_catalog
from spire_voice.plugins.host import module_for_stdio_args, validate_remote_url

router = APIRouter(tags=["plugins"])


# --- Named refusals (this module's own house convention, `routes/macros.py`'s
# --- own module docstring) -----------------------------------------------


def _unknown_plugin_error(plugin_id: int) -> HTTPException:
    return HTTPException(status_code=404, detail=f"no plugin with id {plugin_id}")


def _unknown_catalog_entry_error(name: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"no catalog entry named {name!r} -- see GET /api/plugins/catalog for the curated list",
    )


def _undeclared_config_key_error(keys: Sequence[str], entry_name: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=(
            f"catalog entry {entry_name!r} does not declare configuration key(s) {list(keys)!r} -- "
            "a catalog install only accepts the keys the entry itself declares"
        ),
    )


def _builtin_delete_refused_error(display_name: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"{display_name!r} is a built-in plugin and cannot be deleted -- only disabled and reconfigured",
    )


def _ambiguous_install_source_error() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=(
            "a plugin is installed from exactly one source -- a catalog entry, or a hand-entered "
            "command, or a hand-entered URL, never a combination or none of them"
        ),
    )


def _missing_display_name_error() -> HTTPException:
    return HTTPException(status_code=400, detail="a hand-added plugin is missing its display name")


def _invalid_command_error(reason: str) -> HTTPException:
    return HTTPException(status_code=400, detail=f"invalid plugin command: {reason}")


def _invalid_url_error(reason: str) -> HTTPException:
    return HTTPException(status_code=400, detail=f"invalid plugin url: {reason}")


def _invalid_timeout_error(timeout_ms: int) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"plugin timeout_ms must be a positive number of milliseconds, got {timeout_ms!r}",
    )


def _slug_conflict_error(reason: str) -> HTTPException:
    return HTTPException(status_code=409, detail=reason)


def _reconcile_failed_error() -> HTTPException:
    """The plugin row is already committed by the time this is raised
    (`routes/policy.py`'s own `_respawn_failed_error` docstring states the
    identical reasoning) -- the admin is told to retry, not told nothing."""
    return HTTPException(
        status_code=502,
        detail=(
            "the change was saved, but the running assistant could not be updated to match -- "
            "retry this action; until a reconcile succeeds, the running process may still be "
            "in its previous state"
        ),
    )


# --- Request/response models -----------------------------------------------


class CatalogConfigKeyResponse(BaseModel):
    key: str
    label: str
    secret: bool


class CatalogEntryResponse(BaseModel):
    name: str
    description: str
    transport: str
    config_keys: list[CatalogConfigKeyResponse]


class ConfigValueInput(BaseModel):
    """One submitted configuration value -- `secret` decides whether this
    module encrypts it before it is ever written (D-03). A blank `value`
    on an already-set secret key means "leave it alone", never "clear
    it" -- see `_prepare_config_value` below."""

    model_config = ConfigDict(extra="forbid")

    value: str = ""
    secret: bool = False


class InstallPluginRequest(BaseModel):
    """Exactly one of `catalog_entry` or (`transport` + `command`/`url`)
    must be given (D-02, D-13) -- `_resolve_install_source` below is the
    one place that exclusivity is enforced. No field here can ever set
    `enforces_policy`, `builtin`, or `enabled` -- this model has no such
    field, and `extra="forbid"` refuses a request body that tries to add
    one (T-06-28)."""

    model_config = ConfigDict(extra="forbid")

    catalog_entry: "str | None" = None
    display_name: "str | None" = None
    transport: "str | None" = None  # "command" | "url" -- required when catalog_entry is None
    command: "str | None" = None
    url: "str | None" = None
    timeout_ms: int = 5000
    config_values: dict[str, ConfigValueInput] = Field(default_factory=dict)


class SetEnabledRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool


class SaveConfigRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    values: dict[str, ConfigValueInput] = Field(default_factory=dict)


class PluginToolResponse(BaseModel):
    name: str
    description: str
    # Every other plugin's display name that currently publishes this same
    # bare tool name (D-09, D-10, T-06-27's sibling concern for tools
    # rather than secrets) -- `[]` for an uncontested tool.
    collides_with: list[str] = Field(default_factory=list)


class PluginConfigValueResponse(BaseModel):
    key: str
    secret: bool
    # Populated only when `secret` is False -- the server returns plain
    # values as given, since there is nothing to mask (06-UI-SPEC.md's own
    # copywriting contract). Always `None` for a secret key -- this is the
    # one field this module guarantees never carries a decrypted value
    # (D-03, T-06-27).
    value: "str | None" = None
    is_set: bool


class PluginResponse(BaseModel):
    id: int
    slug: str
    display_name: str
    transport: str
    args: list[str]
    url: "str | None"
    enabled: bool
    builtin: bool
    timeout_ms: int
    # The manager's own closed state set (`PluginState.value`), or
    # "starting" for a plugin this manager has not yet recorded at all --
    # the transitional read right after a Task 2-only write, before Task 3's
    # live reconcile has run.
    state: str
    reason: "str | None" = None
    tools: list[PluginToolResponse] = Field(default_factory=list)
    config_values: list[PluginConfigValueResponse] = Field(default_factory=list)
    # Stated fact, not a browser guess (D-15, matching `routes/policy.py`'s
    # own `PolicyResponse.applies_live`): every write this module makes
    # reaches the running assistant before the route returns.
    applies_live: bool = True


# --- Helpers -----------------------------------------------------------


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def _slugify(display_name: str) -> str:
    """A URL/identifier-safe base slug from `display_name` -- never the
    identifier itself, since two plugins can share a display name; see
    `_unique_slug` below for how a collision is resolved."""
    slug = _SLUG_RE.sub("-", display_name.lower()).strip("-")
    return slug or "plugin"


def _unique_slug(display_name: str, existing_slugs: "set[str]") -> str:
    base = _slugify(display_name)
    if base not in existing_slugs:
        return base
    suffix = 2
    while f"{base}-{suffix}" in existing_slugs:
        suffix += 1
    return f"{base}-{suffix}"


def _args_from_command(slug: str, command: str) -> "tuple[str, ...]":
    """Parse a hand-entered command into the exact `["-m", "<module>"]`
    shape `plugins/host.py::module_for_stdio_args` already requires --
    reused directly, never a second, looser parser, so an admin-typed
    command that would fail at spawn time is refused here instead
    (Pitfall 2: never a second spawn path, and never a second place that
    decides what shape one takes). The interpreter is never taken from the
    request (D-02): a command naming one (`python3 -m foo`, an absolute
    path) is refused by the exact same check that refuses any other
    malformed shape -- this function does not special-case that wording."""
    try:
        tokens = tuple(shlex.split(command))
    except ValueError as exc:
        raise _invalid_command_error(str(exc)) from None
    try:
        module_for_stdio_args(slug, tokens)
    except RuntimeError as exc:
        raise _invalid_command_error(str(exc)) from exc
    return tokens


def _validate_url_or_raise(slug: str, url: str) -> None:
    try:
        validate_remote_url(slug, url)
    except RuntimeError as exc:
        raise _invalid_url_error(str(exc)) from exc


def _make_prepare_config_value(security):
    """One submitted key/value/secret triple, turned into the
    already-encrypted-or-plain `PluginConfigValue` the repository writes --
    bound to this request's own `SecurityConfig` (only reachable from
    `request.app.state.config.security`), never module state, so two
    concurrent requests never share a closure. Returns `None` to mean
    "write nothing for this key" (a blank secret value, D-16's own "leave
    a secret key already set that is submitted blank alone") -- the
    caller is responsible for actually skipping a `None` result."""
    def _prepare(key: str, submitted: ConfigValueInput) -> "PluginConfigValue | None":
        if submitted.secret:
            if not submitted.value:
                return None
            ciphertext, key_version = encrypt_credential(submitted.value, security)
            return PluginConfigValue(
                key=key, secret=True, value=None, ciphertext=ciphertext, key_version=key_version
            )
        return PluginConfigValue(key=key, secret=False, value=submitted.value, ciphertext=None, key_version=None)

    return _prepare


def _catalog_install_config_values(
    entry, submitted: "dict[str, ConfigValueInput]", security
) -> "list[PluginConfigValue]":
    """Every one of `entry`'s own declared keys becomes a row (D-16: a
    curated install is not a blank grid) -- a secret key with no submitted
    value becomes a genuinely unset row (`ciphertext=None`), never an
    encrypted empty string, so a later read can tell "declared but never
    set" apart from "set to an empty value" without ever decrypting
    anything (D-03's single-decryption-point rule)."""
    declared = {ck.key for ck in entry.config_keys}
    extra = set(submitted) - declared
    if extra:
        raise _undeclared_config_key_error(sorted(extra), entry.name)
    prepare = _make_prepare_config_value(security)
    values: "list[PluginConfigValue]" = []
    for config_key in entry.config_keys:
        given = submitted.get(config_key.key, ConfigValueInput(value="", secret=config_key.secret))
        prepared = prepare(config_key.key, ConfigValueInput(value=given.value, secret=config_key.secret))
        if prepared is not None:
            values.append(prepared)
        else:
            # A declared key with nothing submitted is still a real row --
            # a plain one as an empty string (the migration's own
            # precedent), a secret one as a genuinely unset placeholder
            # (`ciphertext=None`) so the editor can still show it as "Not
            # set" without this module ever decrypting anything to find
            # out (D-03's single-decryption-point rule, D-16's "a curated
            # install is not a blank grid").
            values.append(
                PluginConfigValue(
                    key=config_key.key, secret=config_key.secret,
                    value="" if not config_key.secret else None,
                    ciphertext=None, key_version=None,
                )
            )
    return values


def _custom_install_config_values(submitted: "dict[str, ConfigValueInput]", security) -> "list[PluginConfigValue]":
    prepare = _make_prepare_config_value(security)
    values: "list[PluginConfigValue]" = []
    for key, given in submitted.items():
        prepared = prepare(key, given)
        if prepared is not None:
            values.append(prepared)
    return values


async def _to_plugin_response(request: Request, plugin: Plugin) -> PluginResponse:
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    plugin_manager = request.app.state.plugin_manager

    config_values = await plugin_repo.get_config_values(plugin.id)
    state = plugin_manager.state_for(plugin.slug)
    reason = plugin_manager.reason_for(plugin.slug)
    host = plugin_manager.tool_host_for(plugin.slug)

    all_plugins = await plugin_repo.list_plugins()
    display_name_by_slug = {p.slug: p.display_name for p in all_plugins}

    tools: "list[PluginToolResponse]" = []
    if host is not None:
        for tool in host.tools:
            owner_slugs = plugin_manager.owners_of_bare_name(tool.name)
            other_owners = [
                display_name_by_slug.get(slug, slug) for slug in owner_slugs if slug != plugin.slug
            ]
            tools.append(
                PluginToolResponse(name=tool.name, description=tool.description or "", collides_with=other_owners)
            )

    config_entries = [
        PluginConfigValueResponse(
            key=value.key,
            secret=value.secret,
            value=None if value.secret else (value.value or ""),
            is_set=(value.ciphertext is not None) if value.secret else bool(value.value),
        )
        for value in config_values
    ]

    resolved_state = state.value if state is not None else ("disabled" if not plugin.enabled else "starting")

    return PluginResponse(
        id=plugin.id,
        slug=plugin.slug,
        display_name=plugin.display_name,
        transport=plugin.transport,
        args=list(plugin.args),
        url=plugin.url,
        enabled=plugin.enabled,
        builtin=plugin.builtin,
        timeout_ms=plugin.timeout_ms,
        state=resolved_state,
        reason=reason,
        tools=tools,
        config_values=config_entries,
    )


async def _reconcile_live_state(request: Request, plugin: Plugin) -> None:
    """Reach the running assistant before the caller's own route returns
    (D-15, Task 3) -- installing or enabling starts `plugin`'s own
    lifecycle fresh (`PluginManager.start_one`), reading whatever
    configuration was just written; a genuine reconcile failure (never a
    startable-but-misconfigured new plugin, which `start_one` itself
    reports as a created/updated row carrying its own degraded reason,
    not an exception) is reported as a failed write, matching
    `routes/policy.py::_respawn_with_current_policy`'s own rule."""
    plugin_manager = request.app.state.plugin_manager
    try:
        await plugin_manager.start_one(plugin)
    except Exception as exc:  # noqa: BLE001 -- any reconcile failure is reported the same way
        raise _reconcile_failed_error() from exc


async def _reconcile_stop(request: Request, plugin: Plugin) -> None:
    """The disable half of `_reconcile_live_state` -- stops `plugin`'s own
    lifecycle and withdraws its tools before the caller's own route
    returns (D-15, PLUG-05)."""
    plugin_manager = request.app.state.plugin_manager
    try:
        await plugin_manager.stop_one(plugin)
    except Exception as exc:  # noqa: BLE001 -- any reconcile failure is reported the same way
        raise _reconcile_failed_error() from exc


# --- Routes -----------------------------------------------------------


@router.get("/api/plugins/catalog")
async def list_catalog(
    _user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> list[CatalogEntryResponse]:
    entries = load_catalog(DEFAULT_CATALOG_PATH)
    return [
        CatalogEntryResponse(
            name=entry.name,
            description=entry.description,
            transport=entry.transport,
            config_keys=[
                CatalogConfigKeyResponse(key=ck.key, label=ck.label, secret=ck.secret)
                for ck in entry.config_keys
            ],
        )
        for entry in entries
    ]


@router.get("/api/plugins")
async def list_plugins(
    request: Request, _user: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[PluginResponse]:
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    plugins = await plugin_repo.list_plugins()
    return [await _to_plugin_response(request, plugin) for plugin in plugins]


@router.get("/api/plugins/{plugin_id}")
async def get_plugin(
    plugin_id: int, request: Request, _user: CurrentUser = Depends(require_role(Role.ADMIN))
) -> PluginResponse:
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    plugin = await plugin_repo.get_plugin(plugin_id)
    if plugin is None:
        raise _unknown_plugin_error(plugin_id)
    return await _to_plugin_response(request, plugin)


@router.post("/api/plugins", status_code=201)
async def install_plugin(
    payload: InstallPluginRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> PluginResponse:
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    security = request.app.state.config.security

    if payload.timeout_ms <= 0:
        raise _invalid_timeout_error(payload.timeout_ms)

    existing_slugs = {p.slug for p in await plugin_repo.list_plugins()}

    if payload.catalog_entry is not None:
        if payload.command is not None or payload.url is not None or payload.transport is not None:
            raise _ambiguous_install_source_error()
        entries = load_catalog(DEFAULT_CATALOG_PATH)
        entry = find_entry(entries, payload.catalog_entry)
        if entry is None:
            raise _unknown_catalog_entry_error(payload.catalog_entry)
        display_name = payload.display_name or entry.name
        transport = entry.transport
        args: "tuple[str, ...]" = entry.args
        url = entry.url
        if transport == "remote" and url is not None:
            _validate_url_or_raise(_slugify(display_name), url)
        config_values = _catalog_install_config_values(entry, payload.config_values, security)
    else:
        if not payload.display_name:
            raise _missing_display_name_error()
        display_name = payload.display_name
        slug_for_validation = _unique_slug(display_name, existing_slugs)
        if payload.transport == "command":
            if payload.url is not None or not payload.command:
                raise _ambiguous_install_source_error()
            transport = "stdio"
            args = _args_from_command(slug_for_validation, payload.command)
            url = None
        elif payload.transport == "url":
            if payload.command is not None or not payload.url:
                raise _ambiguous_install_source_error()
            _validate_url_or_raise(slug_for_validation, payload.url)
            transport = "remote"
            args = ()
            url = payload.url
        else:
            raise _ambiguous_install_source_error()
        config_values = _custom_install_config_values(payload.config_values, security)

    slug = _unique_slug(display_name, existing_slugs)
    try:
        plugin = await plugin_repo.create_plugin(
            slug=slug,
            display_name=display_name,
            transport=transport,
            args=args,
            url=url,
            timeout_ms=payload.timeout_ms,
            config_values=config_values,
            created_by_user_id=user.id,
        )
    except PluginAlreadyExistsError as exc:
        raise _slug_conflict_error(str(exc)) from exc

    await _reconcile_live_state(request, plugin)
    return await _to_plugin_response(request, plugin)


@router.put("/api/plugins/{plugin_id}/enabled")
async def set_plugin_enabled(
    plugin_id: int,
    payload: SetEnabledRequest,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> PluginResponse:
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    existing = await plugin_repo.get_plugin(plugin_id)
    if existing is None:
        raise _unknown_plugin_error(plugin_id)

    updated = await plugin_repo.set_enabled(plugin_id, payload.enabled)
    if payload.enabled:
        await _reconcile_live_state(request, updated)
    else:
        await _reconcile_stop(request, updated)
    return await _to_plugin_response(request, updated)


@router.put("/api/plugins/{plugin_id}/config")
async def save_plugin_config(
    plugin_id: int,
    payload: SaveConfigRequest,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> PluginResponse:
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    plugin = await plugin_repo.get_plugin(plugin_id)
    if plugin is None:
        raise _unknown_plugin_error(plugin_id)

    security = request.app.state.config.security
    to_write = _custom_install_config_values(payload.values, security)
    await plugin_repo.set_config_values(plugin_id, to_write)

    # Only an enabled plugin has anything running to reconcile -- a
    # disabled row's configuration is saved for whenever it is next
    # enabled, with nothing live to restart in the meantime.
    if plugin.enabled:
        await _reconcile_live_state(request, plugin)
    return await _to_plugin_response(request, plugin)


@router.delete("/api/plugins/{plugin_id}", status_code=204)
async def delete_plugin(
    plugin_id: int,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> None:
    plugin_repo: PluginRepository = request.app.state.plugin_repo
    plugin = await plugin_repo.get_plugin(plugin_id)
    if plugin is None:
        raise _unknown_plugin_error(plugin_id)
    if plugin.builtin:
        raise _builtin_delete_refused_error(plugin.display_name)

    await _reconcile_stop(request, plugin)
    await plugin_repo.delete_plugin(plugin_id)
