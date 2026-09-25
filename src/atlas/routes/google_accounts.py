"""Linking, editing, and unlinking Google accounts over HTTP (GOOG-01, the
write half of GOOG-02) -- `GET/PUT /api/google/client`,
`POST /api/google/oauth/start`, `GET /api/google/oauth/callback`,
`GET /api/google/accounts`, `GET/PATCH/DELETE /api/google/accounts/{id}`,
`PUT /api/google/accounts/{id}/calendars/{calendar_id}`, and
`POST /api/google/accounts/{id}/calendars/refresh`.

Every route requires `Role.ADMIN` (CONTEXT.md discretion, matching Phase 6
D-14), and every refusal has its own named factory, `routes/plugins.py`'s
own house convention. No response model in this module ever carries a
client secret, a refresh token, or an access token (T-09-11) -- none of
them has a field for any of the three.

Every write route ends by calling `reconcile_google_plugin` (`atlas.
google.plugin`) so the running `atlas_mcp.google` child either reflects
the write or is stopped, never left running with wider access than the
operator just chose (D-05, T-09-14, `_reconcile_failed_error` below).
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import re
import secrets
import urllib.parse
from datetime import datetime, timedelta, timezone

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import RedirectResponse
from pydantic import BaseModel, ConfigDict, Field

from atlas_mcp.google_account_api import build_consent_url, list_calendars, revoke_token
from atlas_mcp.google_api import GoogleApiError

from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.crypto.credentials import InvalidToken, decrypt_credential, encrypt_credential
from atlas.db.google_repository import CALENDAR_ACCESS, GoogleAccount, GoogleAccountRepository, GoogleAccountStyle
from atlas.google.linking import LinkError, complete_link
from atlas.google.plugin import find_google_plugin, reconcile_google_plugin
from atlas.google.style import learn_style

logger = logging.getLogger("atlas.routes.google_accounts")

router = APIRouter(tags=["google"])

# D-02: a state row lives ten minutes -- long enough for a real consent
# flow, short enough that an abandoned one is not a lingering CSRF token.
_STATE_TTL = timedelta(minutes=10)

_LABEL_RE = re.compile(r"^[a-z0-9][a-z0-9 -]{0,23}$")

_ACCESS_RANK: dict[str, int] = {"off": 0, "read_only": 1, "read_write": 2}


# --- Named refusals (routes/plugins.py's own house convention) -----------


def _client_not_configured_error() -> HTTPException:
    return HTTPException(
        status_code=409, detail="no google oauth client is configured -- set one first"
    )


def _https_required_error() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=(
            "linking must start from an https page -- the request's Origin header "
            "is missing or is not https"
        ),
    )


def _invalid_label_error() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=(
            "a label must start with a lowercase letter or digit, use only lowercase "
            "letters, digits, spaces, and hyphens, and be at most 24 characters"
        ),
    )


def _label_taken_error() -> HTTPException:
    return HTTPException(status_code=409, detail="another account already uses this label")


def _invalid_client_error() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=(
            "client_id and client_secret must both be non-empty, contain no whitespace, "
            "and be at most 256 characters"
        ),
    )


def _unknown_account_error(account_id: int) -> HTTPException:
    return HTTPException(status_code=404, detail=f"no google account with id {account_id}")


def _unknown_calendar_error(calendar_id: int) -> HTTPException:
    return HTTPException(
        status_code=404, detail=f"no calendar with id {calendar_id} on this account"
    )


def _invalid_calendar_access_error(access: str) -> HTTPException:
    return HTTPException(
        status_code=400, detail=f"calendar access must be one of {CALENDAR_ACCESS!r}, got {access!r}"
    )


def _calendar_access_forbidden_error() -> HTTPException:
    return HTTPException(
        status_code=400,
        detail="this calendar is read-only at google -- it cannot be set to read_write",
    )


def _reconcile_failed_error(*, stopped: bool) -> HTTPException:
    """`_reconcile_failed_error`'s own sibling in `routes/plugins.py` --
    the row is already committed by the time this is raised, so the
    detail says so plainly and states whether the child was stopped
    (D-05, T-09-14) or left running at its previous, still-safe access."""
    if stopped:
        detail = (
            "the setting was saved, but the running google tools could not be updated to "
            "match and were stopped -- retry this action to bring them back"
        )
    else:
        detail = (
            "the setting was saved, but the running google tools could not be updated to "
            "match -- they were left running at their previous access; retry this action"
        )
    return HTTPException(status_code=503, detail=detail)


def _google_unreachable_error(detail: "str | None") -> HTTPException:
    return HTTPException(
        status_code=502, detail=f"google could not be reached: {detail or 'unknown error'}"
    )


# Plan 09-09 (D-19): the admin's own profile edit, at most this many
# characters -- generous for a plain-language style description, small
# enough to bound both the stored row and every drafting round's own
# prompt.
_PROFILE_MAX_LEN = 4000


def _style_learning_in_progress_error() -> HTTPException:
    return HTTPException(
        status_code=409, detail="this account's style is already being learned -- try again shortly"
    )


def _profile_too_long_error() -> HTTPException:
    return HTTPException(
        status_code=400, detail=f"a style profile must be at most {_PROFILE_MAX_LEN} characters"
    )


# --- Request/response models ---------------------------------------------


class GoogleClientResponse(BaseModel):
    configured: bool
    client_id: "str | None" = None
    updated_at: "datetime | None" = None
    redirect_path: str = "/api/google/oauth/callback"


class GoogleClientRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    client_id: str
    client_secret: str


class LinkStartRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: str
    relink_account_id: "int | None" = None


class LinkStartResponse(BaseModel):
    authorization_url: str


class GoogleCalendarResponse(BaseModel):
    id: int
    calendar_id: str
    name: str
    is_primary: bool
    can_write: bool
    access: str


class GoogleAccountResponse(BaseModel):
    id: int
    label: str
    email: str
    is_default: bool
    status: str
    status_detail: "str | None" = None
    refresh_token_expires_at: "datetime | None" = None
    linked_at: datetime
    calendars: list[GoogleCalendarResponse] = Field(default_factory=list)
    # The manager's own `state_for` value for the google plugin row, or
    # `None` when no such row has ever been created yet.
    plugin_state: "str | None" = None


class AccountUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    label: "str | None" = None
    is_default: "bool | None" = None


class CalendarAccessRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    access: str


class StyleResponse(BaseModel):
    status: str
    status_detail: "str | None" = None
    profile: str
    samples: list[str] = Field(default_factory=list)
    signature_text: "str | None" = None
    messages_scanned: int
    learned_at: "datetime | None" = None


class StyleUpdateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    profile: str


# --- Helpers ---------------------------------------------------------


def _hash_state(raw_state: str) -> str:
    return hashlib.sha256(raw_state.encode("utf-8")).hexdigest()


def _normalize_label(raw: str) -> "str | None":
    collapsed = re.sub(r"\s+", " ", raw.strip().lower())
    if not _LABEL_RE.match(collapsed):
        return None
    return collapsed


def _to_calendar_response(calendar) -> GoogleCalendarResponse:
    return GoogleCalendarResponse(
        id=calendar.id,
        calendar_id=calendar.google_calendar_id,
        name=calendar.name,
        is_primary=calendar.is_primary,
        can_write=calendar.can_write,
        access=calendar.access,
    )


def _to_account_response(account: GoogleAccount, plugin_state: "str | None") -> GoogleAccountResponse:
    return GoogleAccountResponse(
        id=account.id,
        label=account.label,
        email=account.email,
        is_default=account.is_default,
        status=account.status,
        status_detail=account.status_detail,
        refresh_token_expires_at=account.refresh_token_expires_at,
        linked_at=account.linked_at,
        calendars=[_to_calendar_response(c) for c in account.calendars],
        plugin_state=plugin_state,
    )


async def _google_plugin_state(request: Request) -> "str | None":
    plugin_repo = request.app.state.plugin_repo
    plugin_manager = request.app.state.plugin_manager
    plugin = await find_google_plugin(plugin_repo)
    if plugin is None:
        return None
    state = plugin_manager.state_for(plugin.slug)
    return state.value if state is not None else None


async def _reconcile_or_refuse(request: Request, admin: CurrentUser, *, narrowing: bool) -> None:
    plugin_repo = request.app.state.plugin_repo
    plugin_manager = request.app.state.plugin_manager
    try:
        await reconcile_google_plugin(
            plugin_repo, plugin_manager, created_by_user_id=admin.id, narrowing=narrowing
        )
    except Exception as exc:  # noqa: BLE001 -- every reconcile failure is reported the same way
        raise _reconcile_failed_error(stopped=narrowing) from exc


def _to_style_response(style: GoogleAccountStyle) -> StyleResponse:
    return StyleResponse(
        status=style.status,
        status_detail=style.status_detail,
        profile=style.profile,
        samples=list(style.samples),
        signature_text=style.signature_text,
        messages_scanned=style.messages_scanned,
        learned_at=style.learned_at,
    )


async def _schedule_learn_style(request: Request, account: GoogleAccount) -> None:
    """Mark `account`'s own style row `"learning"` (synchronously -- a
    `GET` right after this call already reads it) and run `learn_style`
    as a detached background task, held in
    `request.app.state.background_turns` until its own done-callback
    discards it (a task with no strong reference can be garbage collected
    before it ever runs). The one scheduling path both
    `POST .../style/relearn` and a new account's own OAuth callback use --
    there is no periodic job anywhere (D-20)."""
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    token_service = request.app.state.google_token_service
    http_client = request.app.state.google_http_client
    brain = getattr(request.app.state, "brain", None)
    now = datetime.now(timezone.utc)
    await repo.set_style_status(account.id, "learning", None, now)

    async def _run() -> None:
        await learn_style(
            account=account,
            repo=repo,
            token_service=token_service,
            http_client=http_client,
            brain=brain,
            now=datetime.now(timezone.utc),
        )

    # `getattr(..., None)` (not a direct attribute read): every real boot
    # sets `app.state.background_turns` in `app.py`'s own `lifespan`, but
    # a test app built without it (this module's own pre-09-09 test
    # fixtures) must not crash a route that otherwise has nothing to do
    # with those tests' own focus -- the same tolerant-default convention
    # `atlas.google.turn_context.build_handoff_context` already uses for
    # `email_list_memory`.
    background_turns = getattr(request.app.state, "background_turns", None)
    if background_turns is None:
        background_turns = set()
        request.app.state.background_turns = background_turns

    task = asyncio.create_task(_run())
    background_turns.add(task)
    task.add_done_callback(background_turns.discard)


# --- Routes -----------------------------------------------------------


@router.get("/api/google/client")
async def get_google_client(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> GoogleClientResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    client = await repo.get_oauth_client()
    if client is None:
        return GoogleClientResponse(configured=False)
    return GoogleClientResponse(configured=True, client_id=client.client_id, updated_at=client.updated_at)


@router.put("/api/google/client")
async def set_google_client(
    payload: GoogleClientRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> GoogleClientResponse:
    client_id = payload.client_id
    client_secret = payload.client_secret
    if (
        not client_id
        or not client_secret
        or len(client_id) > 256
        or len(client_secret) > 256
        or any(c.isspace() for c in client_id)
        or any(c.isspace() for c in client_secret)
    ):
        raise _invalid_client_error()

    security = request.app.state.config.security
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    ciphertext, key_version = encrypt_credential(client_secret, security)
    client = await repo.set_oauth_client(
        client_id=client_id,
        client_secret_ciphertext=ciphertext,
        key_version=key_version,
        updated_by_user_id=user.id,
        updated_at=datetime.now(timezone.utc),
    )
    return GoogleClientResponse(configured=True, client_id=client.client_id, updated_at=client.updated_at)


@router.post("/api/google/oauth/start")
async def start_link(
    payload: LinkStartRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> LinkStartResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    oauth_client = await repo.get_oauth_client()
    if oauth_client is None:
        raise _client_not_configured_error()

    origin = request.headers.get("origin")
    if not origin or urllib.parse.urlsplit(origin).scheme != "https":
        raise _https_required_error()

    label = _normalize_label(payload.label)
    if label is None:
        raise _invalid_label_error()

    accounts = await repo.list_accounts()
    if any(a.label == label and a.id != payload.relink_account_id for a in accounts):
        raise _label_taken_error()

    now = datetime.now(timezone.utc)
    await repo.purge_expired_oauth_states(now)

    raw_state = secrets.token_urlsafe(32)
    redirect_uri = f"{origin}/api/google/oauth/callback"
    await repo.create_oauth_state(
        state_hash=_hash_state(raw_state),
        label=label,
        redirect_uri=redirect_uri,
        created_by_user_id=admin.id,
        created_at=now,
        expires_at=now + _STATE_TTL,
    )
    authorization_url = build_consent_url(oauth_client.client_id, redirect_uri, raw_state)
    return LinkStartResponse(authorization_url=authorization_url)


@router.get("/api/google/oauth/callback")
async def oauth_callback(
    request: Request,
    code: "str | None" = None,
    state: "str | None" = None,
    error: "str | None" = None,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
):
    if error is not None:
        return RedirectResponse("/google?link_error=denied", status_code=303)

    repo: GoogleAccountRepository = request.app.state.google_account_repo
    now = datetime.now(timezone.utc)

    if not state:
        return RedirectResponse("/google?link_error=state_invalid", status_code=303)
    consumed = await repo.consume_oauth_state(_hash_state(state), now)
    if consumed is None or consumed.created_by_user_id != admin.id:
        return RedirectResponse("/google?link_error=state_invalid", status_code=303)

    if not code:
        return RedirectResponse("/google?link_error=exchange_failed", status_code=303)

    security = request.app.state.config.security
    http_client = request.app.state.google_http_client
    token_service = request.app.state.google_token_service

    try:
        outcome = await complete_link(
            repo=repo,
            security=security,
            http_client=http_client,
            token_service=token_service,
            state=consumed,
            code=code,
            now=now,
        )
    except LinkError as exc:
        return RedirectResponse(f"/google?link_error={exc.code}", status_code=303)

    await _reconcile_or_refuse(request, admin, narrowing=False)

    # Plan 09-09 (D-19, D-20): a genuinely new account learns its style
    # once, in the background, right here -- a re-link of an already-
    # linked address schedules nothing (its style, if any, stands as it
    # already was).
    if outcome.is_new:
        await _schedule_learn_style(request, outcome.account)

    return RedirectResponse(f"/google/accounts/{outcome.account.id}?linked=1", status_code=303)


@router.get("/api/google/accounts")
async def list_accounts(
    request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> list[GoogleAccountResponse]:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    accounts = await repo.list_accounts()
    plugin_state = await _google_plugin_state(request)
    return [_to_account_response(a, plugin_state) for a in accounts]


@router.get("/api/google/accounts/{account_id}")
async def get_account(
    account_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> GoogleAccountResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    account = await repo.get_account(account_id)
    if account is None:
        raise _unknown_account_error(account_id)
    plugin_state = await _google_plugin_state(request)
    return _to_account_response(account, plugin_state)


@router.patch("/api/google/accounts/{account_id}")
async def update_account(
    account_id: int,
    payload: AccountUpdateRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> GoogleAccountResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    existing = await repo.get_account(account_id)
    if existing is None:
        raise _unknown_account_error(account_id)

    label: "str | None" = None
    if payload.label is not None:
        label = _normalize_label(payload.label)
        if label is None:
            raise _invalid_label_error()
        accounts = await repo.list_accounts()
        if any(a.label == label and a.id != account_id for a in accounts):
            raise _label_taken_error()

    now = datetime.now(timezone.utc)
    updated = await repo.update_account(account_id, label=label, is_default=payload.is_default, at=now)
    assert updated is not None  # `existing` above already proved the row exists

    # Clearing a default narrows nothing calendar-wise, but a write whose
    # new state grants less than before still gets the narrowing posture
    # (09-03-PLAN.md's own action text) -- a failed reconcile here stops
    # the child rather than leaving it running against a stale env.
    narrowing = payload.is_default is False
    await _reconcile_or_refuse(request, admin, narrowing=narrowing)

    plugin_state = await _google_plugin_state(request)
    return _to_account_response(updated, plugin_state)


@router.put("/api/google/accounts/{account_id}/calendars/{calendar_id}")
async def set_calendar_access(
    account_id: int,
    calendar_id: int,
    payload: CalendarAccessRequest,
    request: Request,
    admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> GoogleAccountResponse:
    if payload.access not in CALENDAR_ACCESS:
        raise _invalid_calendar_access_error(payload.access)

    repo: GoogleAccountRepository = request.app.state.google_account_repo
    account = await repo.get_account(account_id)
    if account is None:
        raise _unknown_account_error(account_id)
    calendar = await repo.get_calendar(calendar_id)
    if calendar is None or calendar.account_id != account_id:
        raise _unknown_calendar_error(calendar_id)
    if payload.access == "read_write" and not calendar.can_write:
        raise _calendar_access_forbidden_error()

    narrowing = _ACCESS_RANK[payload.access] < _ACCESS_RANK[calendar.access]
    now = datetime.now(timezone.utc)
    await repo.set_calendar_access(calendar_id, payload.access, updated_at=now)

    await _reconcile_or_refuse(request, admin, narrowing=narrowing)

    updated_account = await repo.get_account(account_id)
    assert updated_account is not None
    plugin_state = await _google_plugin_state(request)
    return _to_account_response(updated_account, plugin_state)


@router.post("/api/google/accounts/{account_id}/calendars/refresh")
async def refresh_calendars(
    account_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> GoogleAccountResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    account = await repo.get_account(account_id)
    if account is None:
        raise _unknown_account_error(account_id)

    token_service = request.app.state.google_token_service
    http_client = request.app.state.google_http_client
    access_token = await token_service.access_token_for(account)
    if access_token.token is None:
        raise _google_unreachable_error(access_token.unreachable_reason)

    try:
        calendars = await list_calendars(http_client, access_token.token)
    except GoogleApiError as exc:
        raise _google_unreachable_error(str(exc)) from exc

    now = datetime.now(timezone.utc)
    await repo.add_calendars(
        account_id,
        [(c["id"], c["name"], c["primary"], c["can_write"]) for c in calendars],
        discovered_at=now,
    )
    names = {c["id"]: c["name"] for c in calendars}
    await repo.update_calendar_names(account_id, names, now)

    updated_account = await repo.get_account(account_id)
    assert updated_account is not None
    plugin_state = await _google_plugin_state(request)
    return _to_account_response(updated_account, plugin_state)


@router.delete("/api/google/accounts/{account_id}", status_code=204)
async def delete_account(
    account_id: int, request: Request, admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> None:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    account = await repo.get_account(account_id)
    if account is None:
        raise _unknown_account_error(account_id)

    security = request.app.state.config.security
    http_client = request.app.state.google_http_client
    # B1-WR-03: `revoke_token` is already best effort (it swallows
    # `httpx.HTTPError` so a revoke failure never blocks the unlink), but
    # the decrypt step that produces its argument was not -- a corrupted
    # ciphertext or a rotated/incompatible secret key raised `InvalidToken`
    # here, before `repo.delete_account` ever ran, leaving the row (and
    # the broken credential) in place with no way to remove it through
    # this endpoint. Make the decrypt-and-revoke pair itself best effort
    # with respect to the delete, the same way the revoke call already is.
    try:
        refresh_token = decrypt_credential(account.refresh_token_ciphertext, account.key_version, security)
        await revoke_token(http_client, refresh_token)  # best effort -- never blocks the unlink
    except InvalidToken:
        logger.warning(
            "google account %r: refresh token could not be decrypted for revoke; unlinking anyway",
            account.label,
        )

    await repo.delete_account(account_id)
    token_service = request.app.state.google_token_service
    token_service.forget(account_id)

    await _reconcile_or_refuse(request, admin, narrowing=True)


# --- Plan 09-09: style learning and drafting ------------------------------


@router.get("/api/google/accounts/{account_id}/style")
async def get_account_style(
    account_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> StyleResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    account = await repo.get_account(account_id)
    if account is None:
        raise _unknown_account_error(account_id)
    style = await repo.get_style(account_id)
    return _to_style_response(style)


@router.put("/api/google/accounts/{account_id}/style")
async def update_account_style(
    account_id: int,
    payload: StyleUpdateRequest,
    request: Request,
    _admin: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> StyleResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    account = await repo.get_account(account_id)
    if account is None:
        raise _unknown_account_error(account_id)
    if len(payload.profile) > _PROFILE_MAX_LEN:
        raise _profile_too_long_error()

    await repo.update_style_profile(account_id, payload.profile, datetime.now(timezone.utc))
    style = await repo.get_style(account_id)
    return _to_style_response(style)


@router.post("/api/google/accounts/{account_id}/style/relearn", status_code=202)
async def relearn_account_style(
    account_id: int, request: Request, _admin: CurrentUser = Depends(require_role(Role.ADMIN))
) -> StyleResponse:
    repo: GoogleAccountRepository = request.app.state.google_account_repo
    account = await repo.get_account(account_id)
    if account is None:
        raise _unknown_account_error(account_id)

    current = await repo.get_style(account_id)
    if current.status == "learning":
        raise _style_learning_in_progress_error()

    await _schedule_learn_style(request, account)
    style = await repo.get_style(account_id)
    return _to_style_response(style)
