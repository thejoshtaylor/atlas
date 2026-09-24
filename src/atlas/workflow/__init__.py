"""Scheduled workflows (FLOW-01 through FLOW-10, SAFE-08): a spoken or
webapp-authored plan of ordered steps (`wait`, `call_service`, `speak`),
each claimed off the database when its own absolute `due_at` comes due by
the in-process poller `scheduler.py` starts (05-CONTEXT.md D-01 .. D-04).

- `schedule.py`: `resolve_schedule`, the one boundary that turns a
  caller's way of saying "when" into an absolute, aware UTC instant.
- `scheduler.py`: `WorkflowScheduler`, the poller, shaped exactly like
  `session/retention.py`'s `RetentionScheduler`.
- `steps.py`: `StepOutcome`, `execute_step` -- kind dispatch, one tool-host
  call, never a second copy of `allow_call` (D-13).
- `tool.py`: `ScheduleWorkflowRequest`, `WorkflowToolHost` -- the
  model-facing `schedule_workflow` tool, joining the same
  `McpToolHostLookup` the Home Assistant and weather children already sit
  in (D-08).
"""

from __future__ import annotations
