"""The one, shared, read-only safety-conflict annotator -- extracted out
of `routes/macros.py` (04-06) so `routes/workflows.py` (05-04) reads from
the exact same check rather than a second, editor-specific copy of it.

The annotation this module computes is advisory, never a second
enforcement point. It calls the exact same `allow_call`/`allow_read`
(`mcp/atlas_mcp/safety.py`) the fire path calls -- for a macro action,
`turn/macros.py::fire_macro`; for a workflow step, `workflow/steps.py`'s
own `_execute_call_service` -- plus the same kind of live-catalog read
`routes/policy.py` already uses for its own "Not found in Home Assistant"
badge, read-only, without ever performing the service call a macro action
or a workflow step names. The real boundary still refuses at fire time,
unchanged by this module; a badge here that could drift from that
boundary would be worse than no badge at all, so this module is the one
place either route calls, never a second, per-caller copy of the same
check.

`unknown` is a distinct, first-class annotation value: a policy or catalog
read that failed is reported as "could not check," never silently folded
into "no conflict" -- an operator shipping a macro or a scheduled step on
the strength of a badge that was actually a failed check is exactly the
outcome this distinction exists to prevent.

Plan 06-05 (D-12) reuses this exact value for a second, unrelated reason a
check can be unresolvable: the action's or step's own bare tool name is no
longer owned by exactly one plugin. `annotate_conflict` below checks this
first, before it ever looks at `arguments` -- a tool name two plugins now
publish is unresolvable regardless of what the call would have targeted,
and this module adds no second annotator to say so (both editors already
call this one, and both screens already render its `unknown` value).
`tool_owners_for` reads the answer directly from the running
`PluginManager` (`plugins/naming.py`'s own computed fact, plan 06-04) --
never a second implementation of "how many plugins publish this bare
name" in this route layer.

This module never widens the check beyond what the shipped macro editor
already did: it only annotates an action or a step shaped like a
single-entity `ha_call_service` call (`domain`, `service`, and `entity_id`
together in its own `arguments`). A target on an area, a device, a label,
or a non-Home-Assistant tool has nothing for this read-only check to run
against and is annotated `ok` -- never `unknown` or `denied`. Phase 4's
own review recorded that boundary explicitly; it moves across unchanged,
not widened into a second area/device/label resolver.

`call_target` takes a plain `arguments` mapping rather than a typed
`MacroAction` or `WorkflowStepRow` -- a macro action's JSON arguments and a
workflow step's JSON arguments are both already plain dicts by the time
either route module reaches this function, so neither module needs to
know about the other's own row/value-object type for this shared check to
run.
"""

from __future__ import annotations

import json
from typing import Callable, Literal, Mapping

from fastapi import Request

from atlas_mcp.safety import Denied, Policy, allow_call, allow_read

ConflictAnnotation = Literal["ok", "denied", "not_found", "unknown"]

# What `tool_owners_for` falls back to when a request's app never wired a
# `plugin_manager` (every route test file predating plan 06-05) -- answers
# "no owners" for every bare name, which correctly folds into `ok`, not
# `unknown`: a throwaway test app with no plugin manager at all is a
# different situation from a real deployment's policy or catalog read
# failing, and must not sound like one.
def _no_owners(_bare_name: str) -> "tuple[str, ...]":
    return ()


def tool_owners_for(request: Request) -> "Callable[[str], tuple[str, ...]]":
    """A callable bound to the running `PluginManager`'s own naming
    pre-pass answer for how many plugins currently publish a given bare
    tool name (`plugins/naming.py::NamingResult.owners_of_bare_name`,
    plan 06-04, exposed as `PluginManager.owners_of_bare_name`) -- read
    directly from the one place that fact is computed, per this module's
    own "never a second implementation" rule."""
    plugin_manager = getattr(request.app.state, "plugin_manager", None)
    if plugin_manager is None:
        return _no_owners
    return plugin_manager.owners_of_bare_name


def _tool_result_json(result: object) -> object:
    """Best-effort extraction of a tool result's JSON payload -- the exact
    logic `routes/policy.py::_tool_result_json` already applies,
    duplicated locally for the same reason that module states: `app.py`
    imports `atlas.routes` (this module, transitively), so an
    import the other way would cycle."""
    structured = getattr(result, "structuredContent", None) or getattr(result, "structured_content", None)
    if structured is not None:
        return structured
    content = getattr(result, "content", None) or []
    if content:
        text = getattr(content[0], "text", None)
        if text:
            try:
                return json.loads(text)
            except Exception:  # noqa: BLE001 -- malformed text is "no payload," not a crash
                return None
    return None


async def known_entity_ids(request: Request) -> frozenset[str] | None:
    """A best-effort snapshot of entity ids Home Assistant currently
    reports -- the same kind of read `routes/policy.py::_known_entity_ids`
    already performs for its own "Not found in Home Assistant" badge,
    duplicated locally for the same import-cycle reason `_tool_result_json`
    above states. `None` means "could not check," never "empty":
    `annotate_conflict` below treats `None` as "annotate every
    entity-targeting action or step unknown," per this module's own rule
    that a failed check must never render as no conflict."""
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


async def load_policy_or_none(request: Request) -> Policy | None:
    """A best-effort load of the running policy, used only for the
    read-only conflict annotation below. `None` means "could not check,"
    never "no policy" -- a transient failure here degrades every entity-
    targeting action's or step's annotation to `unknown` rather than
    silently reporting `ok`."""
    policy_repo = getattr(request.app.state, "policy_repo", None)
    if policy_repo is None:
        return None
    try:
        return await policy_repo.load_policy()
    except Exception:  # noqa: BLE001 -- any failure here means "unknown," matching known_entity_ids
        return None


def call_target(arguments: Mapping | None) -> tuple[str, str, str] | None:
    """`(domain, service, entity_id)` if `arguments` carries the shape
    `ha_call_service` needs to target one entity, else `None` -- an action
    or step with no single-entity target (a non-HA tool, or one that
    targets an area/device/label this read-only check does not expand)
    has nothing for the conflict check to run against."""
    args = arguments if isinstance(arguments, dict) else {}
    domain = args.get("domain")
    service = args.get("service")
    entity_id = args.get("entity_id")
    if domain and service and entity_id:
        return domain, service, entity_id
    return None


def annotate_conflict(
    arguments: Mapping | None,
    policy: Policy | None,
    known_entity_ids: frozenset[str] | None,
    *,
    tool_name: str | None = None,
    tool_owners: "Callable[[str], tuple[str, ...]] | None" = None,
) -> ConflictAnnotation:
    """Read-only: calls `allow_read`/`allow_call` exactly as the fire path
    does, but never performs the service call the action or step names.
    See the module docstring for why this must stay the one check, not a
    second, per-caller one.

    `tool_name`/`tool_owners` (plan 06-05, D-12) are both optional and
    both default to `None` -- a caller that predates this plan (or a step
    kind with no tool name of its own to check, like `wait`/`speak`) gets
    byte-identical behavior, the entity-conflict check below unchanged.
    When both are given, the ambiguity check runs first and short-circuits
    the rest: a tool name two or more plugins now publish is unresolvable
    regardless of what `arguments` would have targeted."""
    if tool_name is not None and tool_owners is not None and len(tool_owners(tool_name)) > 1:
        return "unknown"
    target = call_target(arguments)
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
