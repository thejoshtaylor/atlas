"""Policy editing: add and remove denylist/allowlist rules, and switch
between the two modes -- every successful write respawns the enforcing
MCP child before the route returns, and a mode switch writes a named
audit row (SAFE-06, SAFE-07, D-13).

Reads and rule edits require `Role.OPERATOR`: an operator edits the
policy day to day. The mode switch requires `Role.ADMIN`: changing the
stance the whole house is governed by is exactly the action the audit row
below exists to hold someone accountable for, and that accountability
means only an admin can trigger it.

A write that cannot respawn the enforcing child is a failed write (this
plan's own prohibition): by the time `_respawn_with_current_policy` is
attempted, the database row from an add/remove/mode-switch is already
committed, so a respawn failure here is reported with a named error
telling the operator the write landed but the running process may still
be enforcing the previous policy until a retry succeeds -- never silently
returning success for a policy the child never received.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import get_args

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel

from atlas_mcp.safety import Denied, Mode, allow_read
from atlas.auth.dependencies import CurrentUser, Role, require_role
from atlas.db.repository import PolicyRepository, PolicyRule
from atlas.policy_snapshot import safety_block_from_policy

router = APIRouter(tags=["policy"])

_VALID_MODES = get_args(Mode)
_RULE_KINDS = ("deny_entity", "deny_pattern", "allow_entity", "allow_pattern")

# Mirrors `atlas_mcp.safety._ENTITY_RE`'s `domain.name` shape, but tolerant
# of fnmatch's own wildcard metacharacters (`*`, `?`, `[`, `]`) -- a pattern
# rule needs to admit exactly the characters `_ENTITY_RE` forbids, or every
# useful pattern (`switch.example_*`) would be rejected by the same check
# that validates a literal entity id.
_PATTERN_RE = re.compile(r"^[a-z_*?\[\]]+\.[a-z0-9_*?\[\]]+$")


def _unknown_mode_error(mode: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"unknown policy mode {mode!r} -- valid modes are {sorted(_VALID_MODES)!r}",
    )


def _unknown_kind_error(kind: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"unknown rule kind {kind!r} -- valid kinds are {sorted(_RULE_KINDS)!r}",
    )


def _invalid_value_error(kind: str, value: str) -> HTTPException:
    return HTTPException(
        status_code=400,
        detail=f"{value!r} is not a valid value for a {kind} rule",
    )


def _respawn_failed_error() -> HTTPException:
    """The policy row is already committed by the time this is raised
    (see the module docstring) -- the operator is told to retry, not told
    nothing."""
    return HTTPException(
        status_code=502,
        detail=(
            "the policy change was saved, but the process that enforces it could "
            "not be restarted with the new policy -- retry this action; until a "
            "respawn succeeds, the running process may still enforce the previous "
            "policy"
        ),
    )


class PolicyRuleResponse(BaseModel):
    id: int
    kind: str
    value: str
    note: str | None
    created_at: datetime
    # 03-08's "Not found in Home Assistant" badge (T-03-54): true unless
    # this is an entity-kind rule *and* a live entity catalog was
    # actually retrieved *and* the value is absent from it. A
    # pattern-kind rule (no single entity to resolve) and a catalog that
    # could not be fetched (HA unreachable, no tool host yet) both leave
    # this `True` -- a policy row must never look "vanished" because of
    # a transient failure to check it, only because the entity itself is
    # genuinely gone.
    resolved: bool = True


class PolicyResponse(BaseModel):
    mode: str
    rules: list[PolicyRuleResponse]
    # Stated fact, not a browser guess (Task 3, D-03): a policy write
    # always respawns the enforcing child before this route returns, so
    # the policy is live the instant a write succeeds -- unlike a
    # provider credential, which `GET /api/credentials` reports as
    # needing a restart (the providers it feeds are constructed once, in
    # `lifespan`, and nothing rebuilds them on a later write).
    applies_live: bool = True


class AddRuleRequest(BaseModel):
    kind: str
    value: str
    note: str | None = None


class SetModeRequest(BaseModel):
    mode: str


_ENTITY_KINDS = ("deny_entity", "allow_entity")


def _to_rule_response(rule: PolicyRule, known_entity_ids: frozenset[str] | None = None) -> PolicyRuleResponse:
    resolved = True
    if known_entity_ids is not None and rule.kind in _ENTITY_KINDS:
        resolved = rule.value in known_entity_ids
    return PolicyRuleResponse(
        id=rule.id,
        kind=rule.kind,
        value=rule.value,
        note=rule.note,
        created_at=rule.created_at,
        resolved=resolved,
    )


def _tool_result_json(result: object) -> object:
    """Best-effort extraction of a tool result's JSON payload -- the exact
    logic `app.py::_tool_result_json` already applies, duplicated locally
    rather than imported: `app.py` imports `atlas.routes` (this
    module, transitively), so an import the other way would cycle."""
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
    reports, used only to flag a policy rule whose entity id no longer
    resolves (03-08's "Not found in Home Assistant" badge) -- never used
    to decide what is enforced. `None` (HA unreachable, no tool host)
    means "unknown," not "empty": `_to_rule_response` treats `None` as
    "leave every rule marked resolved," since a transient failure to
    check must never make a real rule look like it vanished."""
    tool_host = getattr(request.app.state, "tool_host", None)
    if tool_host is None:
        return None
    try:
        result = await tool_host.call_tool("ha_list_entities", {})
    except Exception:  # noqa: BLE001 -- any failure here means "unknown," not a broken policy load
        return None
    payload = _tool_result_json(result)
    if not isinstance(payload, list):
        return None
    return frozenset(
        entity["entity_id"] for entity in payload if isinstance(entity, dict) and "entity_id" in entity
    )


def _validate_rule_value(kind: str, value: str) -> str:
    """Validate `value` against the same shape `safety.py` enforces for
    that rule kind, returning the normalized value to store -- a rule
    that looks saved and matches nothing is the same failure shape as a
    denylist that looks configured and enforces nothing, which is what
    this whole subsystem exists to prevent.

    An entity-kind value is checked with `safety.py`'s own `allow_read` --
    the exact shape check the boundary itself applies to every entity id
    it ever sees, not a second, locally-invented one. A pattern-kind value
    is checked against `_PATTERN_RE` above, since `allow_read`'s regex
    forbids the wildcard characters a pattern must contain.
    """
    if kind.endswith("_entity"):
        try:
            return allow_read(value)
        except Denied:
            raise _invalid_value_error(kind, value) from None
    normalized = (value or "").strip().lower()
    if not _PATTERN_RE.match(normalized):
        raise _invalid_value_error(kind, value)
    return normalized


async def _respawn_with_current_policy(request: Request) -> None:
    """Rebuild the policy from the repository and respawn the enforcing
    child with it -- called after every successful write, and awaited
    before the route returns, which is the only thing that makes the
    webapp's Live badge honest (this plan's own key link).

    The respawn is requested through `PluginManager`, never by calling
    `McpToolHost.respawn()` on `app.state.tool_host` directly. Since Phase
    6 every plugin's host is owned by one persistent lifecycle task, and
    the installed `mcp` stdio transport binds its `anyio` cancel scope to
    the task that entered it -- a respawn driven from this request's own
    task raises `RuntimeError: Attempted to exit cancel scope in a
    different task than it was entered in`, which would fail every policy
    write in the webapp. `request_policy_respawn` hands the work to the
    task that owns the child; see its docstring.
    """
    policy_repo: PolicyRepository = request.app.state.policy_repo
    policy = await policy_repo.load_policy()
    block = safety_block_from_policy(policy)
    try:
        await request.app.state.plugin_manager.request_policy_respawn(block)
    except Exception as exc:  # noqa: BLE001 -- any respawn failure is reported the same way
        raise _respawn_failed_error() from exc
    request.app.state.safety_block = block


@router.get("/api/policy")
async def get_policy(
    request: Request, _user: CurrentUser = Depends(require_role(Role.OPERATOR))
) -> PolicyResponse:
    policy_repo: PolicyRepository = request.app.state.policy_repo
    policy = await policy_repo.load_policy()
    rules = await policy_repo.list_rules()
    known_entity_ids = await _known_entity_ids(request)
    return PolicyResponse(
        mode=policy.mode, rules=[_to_rule_response(r, known_entity_ids) for r in rules]
    )


@router.post("/api/policy/rules", status_code=201)
async def add_policy_rule(
    payload: AddRuleRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> PolicyRuleResponse:
    if payload.kind not in _RULE_KINDS:
        raise _unknown_kind_error(payload.kind)
    value = _validate_rule_value(payload.kind, payload.value)

    policy_repo: PolicyRepository = request.app.state.policy_repo
    rule = await policy_repo.add_rule(
        kind=payload.kind, value=value, note=payload.note, created_by_user_id=user.id
    )
    await _respawn_with_current_policy(request)
    return _to_rule_response(rule)


@router.delete("/api/policy/rules/{rule_id}", status_code=204)
async def remove_policy_rule(
    rule_id: int,
    request: Request,
    _user: CurrentUser = Depends(require_role(Role.OPERATOR)),
) -> None:
    policy_repo: PolicyRepository = request.app.state.policy_repo
    await policy_repo.remove_rule(rule_id)
    await _respawn_with_current_policy(request)


@router.put("/api/policy/mode")
async def set_policy_mode(
    payload: SetModeRequest,
    request: Request,
    user: CurrentUser = Depends(require_role(Role.ADMIN)),
) -> PolicyResponse:
    if payload.mode not in _VALID_MODES:
        raise _unknown_mode_error(payload.mode)

    policy_repo: PolicyRepository = request.app.state.policy_repo
    old_policy = await policy_repo.load_policy()
    await policy_repo.set_mode(payload.mode, updated_by_user_id=user.id)
    await policy_repo.record_audit(
        "policy.mode_changed",
        {"old_mode": old_policy.mode, "new_mode": payload.mode},
        user.id,
    )
    await _respawn_with_current_policy(request)

    policy = await policy_repo.load_policy()
    rules = await policy_repo.list_rules()
    return PolicyResponse(mode=policy.mode, rules=[_to_rule_response(r) for r in rules])
