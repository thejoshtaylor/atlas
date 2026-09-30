"""Home control guard: an identified member without permission cannot change
the home.

An admin clears `can_control_home` for a member on the Speakers page. In
enforce mode, when the speaker gate identifies that member, the turn controller
wraps the turn's tool host in `HomeControlGuardHost`. The guard refuses every
home write with one fixed sentence. Home Assistant gets no call.

This module follows the shape of `entity_claims`:

- A home write is what `entity_claims.is_home_write` says it is. The guard
  imports that rule and does not keep a second copy.
- The refusal is an error-shaped tool result. The turn controller already
  speaks such a result as written, so no new reply channel exists. The text is
  a fixed template built here in code. No model words it.
- Reads, `get_` services, and every tool that is not a home write pass
  through. The guard blocks a home write and nothing else.

Rules that shape this module:

- The identity comes from the turn's speaker gate result. The controller gives
  it to the constructor as `name`. `call_tool` takes `(name, arguments)` and
  reads no identity from the arguments or the transcript (T-p12-02).
- The guard is the outermost wrapper. The claims wrapper sits inside it. The
  controller wraps `unclaimed(tool_host)` for a macro, so `unclaimed()` never
  removes the guard.
- A scheduled workflow is a side door. `schedule_workflow` and
  `append_workflow_steps` can carry a `call_service` step, and the job runs
  later with no speaker. Creation time is the only point where the check can
  run. The guard refuses these two tools when any step is a home write, or
  when the steps cannot be read (fail closed).
- `cancel_workflow_run` stays allowed. Cancelling prevents a change. It is not
  a home write under D-C.
- A refusal logs and records the bare tool name only. It never carries a
  member name (Phase 11 D-15 log rule).

Decision D-F: this module deliberately overrides Phase 11 D-15 for home writes
only. D-15 says a speaker label is untrusted and grants nothing. That stays
true. The label still never grants anything. It can only remove home control,
so the change is strictly tighter than before.

Known ceiling: a similar voice, or a recording of a permitted member, passes
the speaker gate as that member. The guard stops casual use. It does not stop
a determined attacker. In off and record mode, and for a voice the gate did
not identify, no permission check runs (D-D).
"""

from __future__ import annotations

import logging
from types import SimpleNamespace
from typing import Any, Callable

from atlas.turn.entity_claims import bare_tool_name, is_home_write
from atlas.workflow.tool import APPEND_WORKFLOW_STEPS_TOOL_NAME, SCHEDULE_WORKFLOW_TOOL_NAME

logger = logging.getLogger(__name__)

HOME_CONTROL_REFUSED_EVENT = "home_control.refused"


def home_control_refusal_text(name: str | None) -> str:
    """The sentence the refused turn speaks. It is a fixed template and no
    model words it. In a reply group the group adds the member label, so the
    caller passes `None` and the name is not said twice."""
    if name is None:
        return "you can't control the house."
    return f"{name}, you can't control the house."


def _refusal_result(text: str) -> SimpleNamespace:
    return SimpleNamespace(isError=True, content=[SimpleNamespace(text=text)], home_control_refusal=True)


def is_home_control_refusal(result: Any) -> str | None:
    """The refusal text when `result` is a home-control refusal, else `None`."""
    if getattr(result, "home_control_refusal", False) is not True or not getattr(result, "isError", False):
        return None
    content = getattr(result, "content", None) or []
    text = getattr(content[0], "text", None) if content else None
    return text if isinstance(text, str) else None


_WORKFLOW_STEP_KINDS = frozenset({"wait", "call_service", "speak"})
_WORKFLOW_CREATION_TOOL_NAMES = frozenset({SCHEDULE_WORKFLOW_TOOL_NAME, APPEND_WORKFLOW_STEPS_TOOL_NAME})


def workflow_has_home_write(arguments: Any) -> bool:
    """True when the steps of a workflow call contain a home write, or cannot
    be read. It reads `arguments["steps"]`. A top-level `kind` and `arguments`
    pair with no `steps` is the legacy form with one step. The write rule is
    `is_home_write`, the same rule as everywhere else."""
    if not isinstance(arguments, dict):
        return True
    steps = arguments.get("steps")
    if steps is None and "kind" in arguments:
        steps = [{"kind": arguments.get("kind"), "arguments": arguments.get("arguments", {})}]
    if not isinstance(steps, list):
        return True
    for step in steps:
        if not isinstance(step, dict):
            return True
        kind = step.get("kind")
        if kind not in _WORKFLOW_STEP_KINDS:
            return True
        if kind == "call_service" and is_home_write("ha_call_service", step.get("arguments")):
            return True
    return False


class HomeControlGuardHost:
    """Wraps one turn's tool host and refuses home writes. Same shape as
    `ClaimingToolHost`: `.inner`, `.tools`, and `call_tool`."""

    def __init__(
        self,
        inner: Any,
        *,
        name: str | None,
        record_event: Callable[[dict], None] | None = None,
    ) -> None:
        self.inner = inner
        self._name = name
        self._record_event = record_event

    @property
    def tools(self) -> Any:
        return self.inner.tools

    def _refuse(self, bare_name: str) -> SimpleNamespace:
        logger.info("home control refused for tool %s", bare_name)
        if self._record_event is not None:
            self._record_event({"type": HOME_CONTROL_REFUSED_EVENT, "tool": bare_name})
        return _refusal_result(home_control_refusal_text(self._name))

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        bare_name = bare_tool_name(name)
        if is_home_write(bare_name, arguments):
            return self._refuse(bare_name)
        if bare_name in _WORKFLOW_CREATION_TOOL_NAMES and workflow_has_home_write(arguments):
            return self._refuse(bare_name)
        return await self.inner.call_tool(name, arguments)


def restrict_home_writes(
    host: Any,
    *,
    allowed: bool,
    name: str | None,
    record_event: Callable[[dict], None] | None = None,
) -> Any:
    """The host itself when the turn may control the home or there is no host.
    Otherwise the host wrapped in the guard."""
    if allowed or host is None:
        return host
    return HomeControlGuardHost(host, name=name, record_event=record_event)
