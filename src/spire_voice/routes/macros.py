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
distinction exists to prevent. Plan 06-05 reuses this exact value for an
action whose own `tool` is no longer owned by exactly one plugin
(D-12) -- `annotate_conflict` itself decides this, from `action.tool` and
the running `PluginManager`'s own naming answer, never a second check in
this module.

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
from pathlib import Path
from typing import Callable, Mapping, Sequence

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from spire_mcp.safety import Policy
from spire_voice import config as _config_module
from spire_voice.auth.dependencies import CurrentUser, Role, require_role
from spire_voice.db.repository import Macro, MacroRepository
from spire_voice.providers.tts_cache import precache_all
from spire_voice.routes.conflict import (
    ConflictAnnotation,
    annotate_conflict,
    known_entity_ids as _known_entity_ids,
    load_policy_or_none as _load_policy_or_none,
    tool_owners_for as _tool_owners_for,
)
from spire_voice.turn.macros import normalize

# The exact wording UI-SPEC's "Error state -- reply not synthesized" row
# specifies -- named at module level so `_finish_save` below and any test
# asserting on it read the same string.
_REPLY_SYNTHESIS_DEGRADED_MESSAGE = (
    "This reply couldn't be prepared for fast playback. The macro will still "
    "work, but won't answer instantly until this succeeds."
)

router = APIRouter(tags=["macros"])


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


def _blank_phrase_error() -> HTTPException:
    """MED-01 fix (phase 4 code review): the exact wording
    `MacroConfig.from_config` already raises for a missing `phrase`
    (`ConfigError("a macro is missing its 'phrase'")`) -- reproduced here
    because `normalize("")` yields `""`, and `turn/macros.py::match()`
    explicitly refuses to match any transcript against that key, so a
    macro saved with a blank phrase can never fire by voice again, with
    no error at save time to say so."""
    return HTTPException(status_code=400, detail="a macro is missing its 'phrase'")


def _blank_reply_error(phrase: str) -> HTTPException:
    """MED-01 fix (phase 4 code review): the exact wording
    `MacroConfig.from_config` already raises for a missing `reply`
    (`ConfigError(f"macro {phrase!r} is missing its 'reply'")`) --
    reproduced here because a blank reply is handed to `precache_all` on
    save, attempting to synthesize zero-length speech through whichever
    TTS provider is configured."""
    return HTTPException(status_code=400, detail=f"macro {phrase!r} is missing its 'reply'")


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


def _validate_actions(phrase: str, reply: str, actions: Sequence[MacroActionInput]) -> None:
    """MED-01 fix (phase 4 code review): this module's own docstring
    states "every refusal the file-parsed `MacroConfig`/`MacroActionConfig`
    would make on a malformed macro, this module makes too" -- until this
    fix, that was true only of the action-list checks below, not of
    `MacroConfig.from_config`'s blank-phrase/blank-reply checks. Same
    order the file parser checks in (phrase, then reply, then actions),
    same wording, so an operator sees the identical message regardless of
    which path caught the same mistake."""
    if not phrase:
        raise _blank_phrase_error()
    if not reply:
        raise _blank_reply_error(phrase)
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
        # Referenced off the module object, not a name bound at this
        # module's own import time (`from spire_voice.config import
        # ConfigError`): `tests/test_config.py`'s own
        # `test_config_and_turn_macros_import_in_either_order` reloads
        # `spire_voice.config` in place, which replaces `ConfigError` with
        # a fresh class object in that module's namespace -- a name bound
        # here before the reload would then no longer `except` what a
        # post-reload `_check_macros_do_not_collide` raises. Attribute
        # access on `_config_module` always resolves against whatever is
        # currently in `sys.modules['spire_voice.config']`, so this stays
        # correct across a reload the same way `spire_voice.config`'s own
        # module docstring already documents that test proving.
        _config_module._check_macros_do_not_collide([*existing, candidate])
    except _config_module.ConfigError as exc:
        raise _duplicate_phrase_error(str(exc)) from exc


def _to_macro_response(
    macro: Macro,
    policy: Policy | None,
    known_entity_ids: frozenset[str] | None,
    filler_cache: Mapping[str, bytes],
    tool_owners: "Callable[[str], tuple[str, ...]]",
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
                conflict=annotate_conflict(
                    action.arguments,
                    policy,
                    known_entity_ids,
                    tool_name=action.tool,
                    tool_owners=tool_owners,
                ),
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
    tool_owners = _tool_owners_for(request)
    return [
        _to_macro_response(macro, policy, known_entity_ids, filler_cache, tool_owners) for macro in macros
    ]


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
    tool_owners = _tool_owners_for(request)
    return _to_macro_response(macro, policy, known_entity_ids, filler_cache, tool_owners)


async def _finish_save(request: Request, macro: Macro) -> MacroResponse:
    """After a successful create/update commit, make sure `macro.reply` is
    in the same cache a turn reads from before this route answers (D-12,
    T-04-33) -- through `precache_all`, the exact function `app.py`'s own
    startup precache uses, with the same cache directory, voice id, and
    sink. One synthesis function, not two, per `tts_cache.py`'s own
    docstring.

    Skipped when `macro.reply` is already in the cache -- this both spares
    a reorder-only update the cost of a pointless re-synthesis, and is
    what makes a retry after a prior failure actually retry: a failed
    synthesis never writes its text into the cache, so the next save for
    the same text tries again rather than being skipped as unchanged.

    A synthesis failure is not a failed save (D-12, UI-SPEC): the macro
    row is already committed by the time this runs, so the response
    reports a degraded state and a message rather than raising -- losing
    an operator's edit to a text-to-speech hiccup would be the worse
    failure.

    Nothing here refreshes what the next turn reads: `_current_macros`
    (`app.py`) is a live, per-turn repository read by construction (plan
    04-05), so the database write `create_macro`/`update_macro` already
    performed is the only thing a future turn needs to see this macro --
    a second refresh step here would do nothing.
    """
    filler_cache = getattr(request.app.state, "filler_cache", None)
    if filler_cache is None:
        filler_cache = {}
        request.app.state.filler_cache = filler_cache

    degraded = False
    message: str | None = None
    if macro.reply not in filler_cache:
        config = request.app.state.config
        try:
            new_entries = await precache_all(
                request.app.state.tts,
                Path(config.tts.cache_dir),
                [macro.reply],
                config.tts.voice_id,
                request.app.state.tts.browser_sink(),
            )
            filler_cache.update(new_entries)
        except Exception:  # noqa: BLE001 -- any synthesis failure degrades, never loses the edit
            degraded = True
            message = _REPLY_SYNTHESIS_DEGRADED_MESSAGE

    policy = await _load_policy_or_none(request)
    known_entity_ids = await _known_entity_ids(request)
    tool_owners = _tool_owners_for(request)
    return _to_macro_response(
        macro,
        policy,
        known_entity_ids,
        filler_cache,
        tool_owners,
        reply_synthesis_degraded=degraded,
        reply_synthesis_message=message,
    )


@router.post("/api/macros", status_code=201)
async def create_macro(
    payload: CreateMacroRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> MacroResponse:
    _validate_actions(payload.phrase, payload.reply, payload.actions)
    macro_repo: MacroRepository = request.app.state.macro_repo
    await _check_no_collision(macro_repo, payload.phrase, payload.aliases)

    # HI-01 fix (phase 4 code review): `_check_no_collision` above is a
    # check-then-act pre-check -- it decides the common-case, non-racing
    # 400 and names the colliding phrases in it -- but it is not, by
    # itself, what makes two concurrent creates with colliding phrases
    # impossible. `macro_repo.create_macro` now reruns the identical
    # check under a database-level advisory lock, inside the same
    # transaction as the insert; a losing concurrent request raises the
    # same `ConfigError` here, which this route converts to the same
    # refusal shape `_check_no_collision` already produces, rather than
    # letting it surface as an unhandled 500.
    try:
        macro = await macro_repo.create_macro(
            phrase=payload.phrase,
            aliases=payload.aliases,
            reply=payload.reply,
            actions=[(action.tool, action.arguments) for action in payload.actions],
            created_by_user_id=user.id,
        )
    except _config_module.ConfigError as exc:
        raise _duplicate_phrase_error(str(exc)) from exc
    return await _finish_save(request, macro)


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

    _validate_actions(payload.phrase, payload.reply, payload.actions)
    await _check_no_collision(macro_repo, payload.phrase, payload.aliases, exclude_macro_id=macro_id)

    # Ordering is sent, not inferred: the whole action list is replaced in
    # the order the request carried, which is what makes a reorder-only
    # update persist exactly as sent (Task 2's own instruction).
    #
    # HI-01 fix (phase 4 code review): same reasoning as create_macro
    # above -- `macro_repo.update_macro` reruns the collision check under
    # the advisory lock, inside the same transaction as the write, and a
    # losing concurrent request's `ConfigError` is converted to the same
    # refusal shape here.
    try:
        macro = await macro_repo.update_macro(
            macro_id,
            phrase=payload.phrase,
            aliases=payload.aliases,
            reply=payload.reply,
            actions=[(action.tool, action.arguments) for action in payload.actions],
        )
    except _config_module.ConfigError as exc:
        raise _duplicate_phrase_error(str(exc)) from exc
    return await _finish_save(request, macro)


@router.delete("/api/macros/{macro_id}", status_code=204)
async def delete_macro(
    macro_id: int, request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> None:
    macro_repo: MacroRepository = request.app.state.macro_repo
    # A no-op when `macro_id` is already gone -- matches
    # `PolicyRepository.remove_rule`'s own convention (`routes/policy.py`).
    await macro_repo.delete_macro(macro_id)
