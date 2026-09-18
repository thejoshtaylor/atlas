"""Macro authoring over HTTP: list, read, create, update and delete a
macro, with a read-only safety-conflict annotation on every action and a
reply re-synthesized before a save returns (MACRO-03, D-09, D-10, D-12).

Every route -- reads included -- requires `Role.OPERATOR`: an operator
edits macros day to day the way they edit policy, and there is no
structurally destructive operation here that needs the admin gate the
policy mode switch carries (`routes/policy.py`'s own module docstring
states that reasoning for the sibling case this mirrors).

The conflict annotation on each action is advisory, never a second
enforcement point. It calls the exact same `allow_call`/`allow_read`
(`mcp/spire_mcp/safety.py`) and the exact same kind of live-catalog read
`routes/policy.py` already uses for its own "Not found in Home Assistant"
badge -- read-only, without ever performing the service call a macro
action names. The real boundary still refuses at fire time
(`turn/macros.py::fire_macro`, unchanged by this module); a badge here
that could drift from that boundary would be worse than no badge at all,
so this module never writes a second, editor-specific copy of the check.
`unknown` is a distinct, first-class annotation value: a policy or catalog
read that failed is reported as "could not check," never silently folded
into "no conflict" -- an operator shipping a macro on the strength of a
badge that was actually a failed check is exactly the outcome this
distinction exists to prevent.

Every refusal the file-parsed `spire_voice.config.MacroConfig`/
`MacroActionConfig` would make on a malformed macro, this module makes
too, with the same reason and the same shared collision check
(`spire_voice.config._check_macros_do_not_collide`) -- reused directly
against the repository's own `Macro` objects (duck-type compatible, see
`spire_voice.db.repository.Macro`'s own docstring), never restated.

A create or update re-synthesizes the macro's reply through the exact
`precache_all` function `app.py`'s own startup precache uses -- same
cache directory, same voice id, same sink -- and merges the result into
the same `app.state.filler_cache` dict a turn reads from
(`turn/controller.py`), awaited before this route returns. A synthesis
failure is reported as a degraded success, never a lost edit: the macro
row is already committed by the time synthesis is attempted, so losing
the operator's edit to a text-to-speech hiccup would be the worse
failure (D-12, UI-SPEC). Synthesis is skipped whenever the reply text is
already present in the cache -- this both spares a reorder-only save the
cost of a pointless re-synthesis and is what makes a retry after a prior
failure actually retry: a failed synthesis never wrote its text into the
cache, so the next save for the same text tries again rather than being
skipped as unchanged.

What the next turn reads is `_current_macros` (`app.py`), a live,
per-request repository read chosen in plan 04-05 specifically so no
write route here has to remember to invalidate a cache of its own --
this module needs no separate refresh step for that reason, and does not
add one that would do nothing.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal, Mapping, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from spire_mcp.safety import Denied, Policy, allow_call, allow_read
from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.config import ConfigError, _check_macros_do_not_collide
from spire_voice.db.repository import Macro, MacroAction, MacroRepository
from spire_voice.turn.macros import normalize

router = APIRouter(tags=["macros"])

ConflictAnnotation = Literal["ok", "denied", "not_found", "unknown"]


def _macro_not_found_error(macro_id: int) -> HTTPException:
    return HTTPException(status_code=404, detail=f"no macro with id {macro_id}")


def _zero_actions_error(phrase: str) -> HTTPException:
    """The exact reason `spire_voice.config.MacroConfig.from_config`
    already refuses a zero-action macro (D-12's own precache guarantee):
    an unconditionally-precached reply for a macro that did nothing would
    be a lie."""
    return HTTPException(
        status_code=400,
        detail=(
            f"macro {phrase!r} has no actions -- a zero-action macro's precached "
            "reply would be an unconditional lie"
        ),
    )


def _missing_tool_name_error(phrase: str, index: int) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"macro {phrase!r}'s action at position {index} is missing its 'tool' name",
    )


def _duplicate_phrase_error(message: str) -> HTTPException:
    """`message` is `_check_macros_do_not_collide`'s own `ConfigError`
    text, carried through unchanged -- it already names both colliding
    macros by their written phrases, the exact shape the file parser's own
    refusal uses."""
    return HTTPException(status_code=400, detail=message)


class MacroActionResponse(BaseModel):
    id: int
    position: int
    tool: str
    arguments: dict
    # Advisory only -- see the module docstring. `unknown` is distinct
    # from `ok` on purpose (Task 1's own instruction): a failed check must
    # never render as "no conflict."
    conflict: ConflictAnnotation


class MacroResponse(BaseModel):
    id: int
    phrase: str
    aliases: list[str]
    reply: str
    actions: list[MacroActionResponse]
    created_at: datetime
    updated_at: datetime
    created_by_user_id: int | None
    # Whether `reply` is currently in the same cache a turn reads from --
    # true for any macro whose reply has ever been precached (at startup
    # or by a prior save) and still matches. UI-SPEC's "Reply cached ·
    # ready" badge reads this directly.
    reply_cached: bool
    # Only meaningful on a create/update response (Task 3) -- `False`/
    # `None` on every list/read response, since a read never re-synthesizes
    # anything.
    reply_synthesis_degraded: bool = False
    reply_synthesis_message: str | None = None


class MacroActionInput(BaseModel):
    tool: str = ""
    arguments: dict = Field(default_factory=dict)


class CreateMacroRequest(BaseModel):
    phrase: str
    aliases: list[str] = Field(default_factory=list)
    reply: str
    actions: list[MacroActionInput] = Field(default_factory=list)


class UpdateMacroRequest(BaseModel):
    phrase: str
    aliases: list[str] = Field(default_factory=list)
    reply: str
    actions: list[MacroActionInput] = Field(default_factory=list)


@dataclass(frozen=True)
class _CandidateMacro:
    """The would-be macro a create or update is about to write, shaped
    exactly like `spire_voice.db.repository.Macro`'s own duck-typed
    contract (`.phrase`, `.normalized_keys`) so it can sit in the same
    list `_check_macros_do_not_collide` walks alongside the real,
    already-stored `Macro` rows -- with no change to that function."""

    phrase: str
    aliases: tuple[str, ...]

    @property
    def normalized_keys(self) -> frozenset[str]:
        return frozenset(normalize(k) for k in (self.phrase, *self.aliases))


def _validate_actions(phrase: str, actions: Sequence[MacroActionInput]) -> None:
    if not actions:
        raise _zero_actions_error(phrase)
    for index, action in enumerate(actions):
        if not action.tool:
            raise _missing_tool_name_error(phrase, index)


async def _check_no_collision(
    macro_repo: MacroRepository,
    phrase: str,
    aliases: Sequence[str],
    *,
    exclude_macro_id: int | None = None,
) -> None:
    """Self-collision is legal, cross-macro collision is not -- the same
    distinction `Macro.normalized_keys`/`_check_macros_do_not_collide`
    already draw for the file path. An update excludes its own existing
    row before adding the candidate back in, so a save that keeps a
    macro's own phrase is never refused by a check comparing it against
    itself."""
    existing = await macro_repo.list_macros()
    if exclude_macro_id is not None:
        existing = [m for m in existing if m.id != exclude_macro_id]
    candidate = _CandidateMacro(phrase=phrase, aliases=tuple(aliases))
    try:
        _check_macros_do_not_collide([*existing, candidate])
    except ConfigError as exc:
        raise _duplicate_phrase_error(str(exc)) from exc


def _tool_result_json(result: object) -> object:
    """Best-effort extraction of a tool result's JSON payload -- the exact
    logic `routes/policy.py::_tool_result_json` already applies,
    duplicated locally for the same reason that module states: `app.py`
    imports `spire_voice.routes` (this module, transitively), so an
    import the other way would cycle."""
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if text:
            try:
                import json

                return json.loads(text)
            except Exception:  # noqa: BLE001 -- malformed text is "no payload," not a crash
                return None
    return None


async def _known_entity_ids(request: Request) -> frozenset[str] | None:
    """A best-effort snapshot of entity ids Home Assistant currently
    reports -- the same kind of read `routes/policy.py::_known_entity_ids`
    already performs for its own "Not found in Home Assistant" badge,
    duplicated locally for the same import-cycle reason
    `_tool_result_json` above states. `None` means "could not check,"
    never "empty": `_annotate_action_conflict` below treats `None` as
    "annotate every entity-targeting action unknown," per this module's
    own rule that a failed check must never render as no conflict."""
    tool_host = getattr(request.app.state, "tool_host", None)
    if tool_host is None:
        return None
    try:
        result = await tool_host.call_tool("ha_list_entities", {})
    except Exception:  # noqa: BLE001 -- any failure here means "unknown," not a broken load
        return None
    payload = _tool_result_json(result)
    if not isinstance(payload, list):
        return None
    return frozenset(
        entity["entity_id"] for entity in payload if isinstance(entity, dict) and "entity_id" in entity
    )


async def _load_policy_or_none(request: Request) -> Policy | None:
    """A best-effort load of the running policy, used only for the
    read-only conflict annotation below. `None` means "could not check,"
    never "no policy" -- a transient failure here degrades every entity-
    targeting action's annotation to `unknown` rather than silently
    reporting `ok`."""
    policy_repo = getattr(request.app.state, "policy_repo", None)
    if policy_repo is None:
        return None
    try:
        return await policy_repo.load_policy()
    except Exception:  # noqa: BLE001 -- any failure here means "unknown," matching _known_entity_ids
        return None


def _action_target(action: MacroAction) -> tuple[str, str, str] | None:
    """`(domain, service, entity_id)` if `action.arguments` carries the
    shape `ha_call_service` needs to target one entity, else `None` --
    an action with no single-entity target (a non-HA tool, or one that
    targets an area/device/label this read-only check does not expand)
    has nothing for the conflict check to run against."""
    args = action.arguments if isinstance(action.arguments, dict) else {}
    domain = args.get("domain")
    service = args.get("service")
    entity_id = args.get("entity_id")
    if domain and service and entity_id:
        return domain, service, entity_id
    return None


def _annotate_action_conflict(
    action: MacroAction, policy: Policy | None, known_entity_ids: frozenset[str] | None
) -> ConflictAnnotation:
    """Read-only: calls `allow_read`/`allow_call` exactly as the fire path
    does, but never performs the service call the action names. See the
    module docstring for why this must stay the one check, not a second,
    editor-specific one."""
    target = _action_target(action)
    if target is None:
        return "ok"
    domain, service, entity_id = target
    if policy is None or known_entity_ids is None:
        return "unknown"
    try:
        entity_id = allow_read(entity_id)
    except Denied:
        # A malformed id cannot resolve against a live catalog either --
        # treat it the same as absent, not as a third shape of failure.
        return "not_found"
    if entity_id not in known_entity_ids:
        return "not_found"
    try:
        allow_call(policy, domain, service, entity_id)
    except Denied:
        return "denied"
    return "ok"


def _to_macro_response(
    macro: Macro,
    policy: Policy | None,
    known_entity_ids: frozenset[str] | None,
    filler_cache: Mapping[str, bytes],
    *,
    reply_synthesis_degraded: bool = False,
    reply_synthesis_message: str | None = None,
) -> MacroResponse:
    return MacroResponse(
        id=macro.id,
        phrase=macro.phrase,
        aliases=list(macro.aliases),
        reply=macro.reply,
        actions=[
            MacroActionResponse(
                id=action.id,
                position=action.position,
                tool=action.tool,
                arguments=action.arguments,
                conflict=_annotate_action_conflict(action, policy, known_entity_ids),
            )
            for action in macro.actions
        ],
        created_at=macro.created_at,
        updated_at=macro.updated_at,
        created_by_user_id=macro.created_by_user_id,
        reply_cached=macro.reply in filler_cache,
        reply_synthesis_degraded=reply_synthesis_degraded,
        reply_synthesis_message=reply_synthesis_message,
    )


@router.get("/api/macros")
async def list_macros(
    request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> list[MacroResponse]:
    macro_repo: MacroRepository = request.app.state.macro_repo
    macros = await macro_repo.list_macros()
    policy = await _load_policy_or_none(request)
    known_entity_ids = await _known_entity_ids(request)
    filler_cache = getattr(request.app.state, "filler_cache", None) or {}
    return [_to_macro_response(macro, policy, known_entity_ids, filler_cache) for macro in macros]


@router.get("/api/macros/{macro_id}")
async def get_macro(
    macro_id: int, request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> MacroResponse:
    macro_repo: MacroRepository = request.app.state.macro_repo
    macro = await macro_repo.get_macro(macro_id)
    if macro is None:
        raise _macro_not_found_error(macro_id)
    policy = await _load_policy_or_none(request)
    known_entity_ids = await _known_entity_ids(request)
    filler_cache = getattr(request.app.state, "filler_cache", None) or {}
    return _to_macro_response(macro, policy, known_entity_ids, filler_cache)


@router.post("/api/macros", status_code=201)
async def create_macro(
    payload: CreateMacroRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> MacroResponse:
    _validate_actions(payload.phrase, payload.actions)
    macro_repo: MacroRepository = request.app.state.macro_repo
    await _check_no_collision(macro_repo, payload.phrase, payload.aliases)

    macro = await macro_repo.create_macro(
        phrase=payload.phrase,
        aliases=payload.aliases,
        reply=payload.reply,
        actions=[(action.tool, action.arguments) for action in payload.actions],
        created_by_user_id=user.id,
    )
    policy = await _load_policy_or_none(request)
    known_entity_ids = await _known_entity_ids(request)
    filler_cache = getattr(request.app.state, "filler_cache", None) or {}
    return _to_macro_response(macro, policy, known_entity_ids, filler_cache)


@router.put("/api/macros/{macro_id}")
async def update_macro(
    macro_id: int,
    payload: UpdateMacroRequest,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> MacroResponse:
    macro_repo: MacroRepository = request.app.state.macro_repo
    existing = await macro_repo.get_macro(macro_id)
    if existing is None:
        raise _macro_not_found_error(macro_id)

    _validate_actions(payload.phrase, payload.actions)
    await _check_no_collision(macro_repo, payload.phrase, payload.aliases, exclude_macro_id=macro_id)

    # Ordering is sent, not inferred: the whole action list is replaced in
    # the order the request carried, which is what makes a reorder-only
    # update persist exactly as sent (Task 2's own instruction).
    macro = await macro_repo.update_macro(
        macro_id,
        phrase=payload.phrase,
        aliases=payload.aliases,
        reply=payload.reply,
        actions=[(action.tool, action.arguments) for action in payload.actions],
    )
    policy = await _load_policy_or_none(request)
    known_entity_ids = await _known_entity_ids(request)
    filler_cache = getattr(request.app.state, "filler_cache", None) or {}
    return _to_macro_response(macro, policy, known_entity_ids, filler_cache)


@router.delete("/api/macros/{macro_id}", status_code=204)
async def delete_macro(
    macro_id: int, request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> None:
    macro_repo: MacroRepository = request.app.state.macro_repo
    # A no-op when `macro_id` is already gone -- matches
    # `PolicyRepository.remove_rule`'s own convention (`routes/policy.py`).
    await macro_repo.delete_macro(macro_id)
