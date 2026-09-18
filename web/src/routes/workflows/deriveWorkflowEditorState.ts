// WorkflowEditorRoute's own logic, kept apart from its JSX -- copied
// verbatim in shape from `routes/macros/deriveMacroEditorState.ts`'s
// `QueryLike` interface and discriminated union. Imports nothing from
// React, for the same stated reason.
import { ApiError } from "@/lib/api"
import type { Macro } from "@/lib/macros"
import type { ConflictAnnotation, WorkflowRun, WorkflowStep } from "@/lib/workflows"
import type { DraftStep } from "@/stores/workflowDraftStore"
import { deriveReplyPrecacheState, type ReplyPrecacheState } from "@/routes/macros/deriveMacroEditorState"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

// `run: null` is the "new, unsaved run" ready state -- there is no id to
// fetch yet, so `WorkflowEditorRoute` never runs a query for it at all
// (see `query: null` below) rather than deriving a fake loading state
// for a fetch that would never happen.
export type WorkflowEditorScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; run: WorkflowRun | null }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load this run. It may have already run, been cancelled, or no longer exists."
}

/**
 * `query` is `null` for a new run (`/workflows/new`, no id to fetch) --
 * that route never contacts the server at all, so it is always ready
 * with `run: null` rather than a query that stays pending forever.
 */
export function deriveWorkflowEditorState(query: QueryLike<WorkflowRun> | null): WorkflowEditorScreenState {
  if (query === null) return { kind: "ready", run: null }
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  if (!query.data) return { kind: "loading" }
  return { kind: "ready", run: query.data }
}

// A sixteenth state, not named by 05-UI-SPEC.md's own Copywriting
// Contract table but forced by the backend contract it builds on:
// `CreateWorkflowRequest.summary` (routes/workflows.py) is
// `min_length=1` -- "what the operator would say to mean this run,
// used later to cancel it by voice" (D-09). The UI-SPEC's own Focal
// Point row for this screen never names a Summary input, but a "New
// workflow" flow with no way to supply this field would always fail
// POST /api/workflows with a 422 -- Rule 2 (auto-add missing critical
// functionality): a Summary field is added, guarded inline the same way
// every other required field on this screen is, and recorded in this
// plan's own SUMMARY.md as a discovered sixteenth state. Only meaningful
// for a NEW run -- `PUT /api/workflows/{id}` never accepts `summary`
// (05-04's own key-decision), so an existing run's summary is read-only
// after creation and this rule never applies to it.
export const BLANK_SUMMARY_SAVE_BLOCKED_REASON = "Add a summary before saving."

export function saveBlockedByBlankSummary(summary: string): boolean {
  return summary.trim() === ""
}

/** UI-SPEC's exact "Error state -- zero steps on save" copy. */
export const ZERO_STEPS_SAVE_BLOCKED_REASON = "Add at least one step before saving."

/** Save is blocked, inline, before submit -- rather than a
 * submit-then-fail round trip, the same instinct the macro editor's
 * zero-action rule already follows. */
export function saveBlockedByZeroSteps(steps: DraftStep[]): string | null {
  return steps.length === 0 ? ZERO_STEPS_SAVE_BLOCKED_REASON : null
}

/** UI-SPEC's exact "Error state -- schedule time in the past" copy. */
export const PAST_SCHEDULE_SAVE_BLOCKED_REASON = "Pick a time in the future."

/** A blank `runAt` is not itself a past-schedule refusal -- an empty
 * required field is a different, unaddressed-by-this-function problem
 * (the native `<input required>` handles it). This only fires once the
 * operator has entered a value that already sits at or before `now`. */
export function saveBlockedByPastSchedule(runAt: string, now: Date): string | null {
  if (runAt.trim() === "") return null
  const parsed = new Date(runAt)
  if (Number.isNaN(parsed.getTime())) return null
  return parsed.getTime() <= now.getTime() ? PAST_SCHEDULE_SAVE_BLOCKED_REASON : null
}

/** Every refusal `routes/workflows.py::_validate_steps` reproduces at
 * authoring time, reproduced here too so the operator learns
 * immediately rather than after a doomed submit (T-05-31: this client
 * side copy exists to be kind, never to be the gate -- the route
 * enforces the same rule independently). */
export function stepIsValid(step: DraftStep): boolean {
  if (step.kind === "wait") {
    const amount = Number(step.durationValue)
    return step.durationValue.trim() !== "" && Number.isFinite(amount) && amount >= 0
  }
  if (step.kind === "call_service") {
    if (step.domain.trim() === "" || step.service.trim() === "") return false
    const transition = step.transition.trim()
    if (transition === "") return true
    const amount = Number(transition)
    if (!Number.isFinite(amount) || amount < 0) return false
    // The same light-only restriction `_transition_refusal`
    // (workflow/steps.py) enforces a second time at fire time --
    // caught here too so the operator learns immediately.
    return step.domain.trim() === "light"
  }
  return step.text.trim() !== ""
}

/** D-11's append-only-once-firing rule, made visible rather than a
 * surprise at save time: once a run has left `pending`, adding,
 * removing, and reordering steps is refused server-side
 * (`WorkflowRunNotAppendableError`, 05-04) -- this makes that fact
 * visible on the editor itself. A brand-new (`null`) run is never
 * locked -- there is nothing yet to have started firing. */
export function editorLocked(run: WorkflowRun | null): boolean {
  return run !== null && run.status !== "pending"
}

export type StepConflictDisplay =
  | { kind: "ok" }
  | { kind: "denied"; text: string }
  | { kind: "not_found"; text: string }
  | { kind: "unknown"; text: string }

/**
 * 05-UI-SPEC.md's Copywriting Contract, verbatim, for the workflow
 * editor's own wording (deliberately weaker than the macro editor's --
 * SAFE-08 re-reads the policy at fire time, so "would be refused right
 * now" is the honest claim, never "will be refused"). `unknown` is a
 * distinct member from `ok` on purpose -- a policy or catalog read that
 * failed while the editor was open must never render as "no conflict"
 * (T-05-32).
 */
export function stepConflictDisplay(conflict: ConflictAnnotation): StepConflictDisplay {
  switch (conflict) {
    case "ok":
      return { kind: "ok" }
    case "denied":
      return { kind: "denied", text: "Denied for control · would be refused right now" }
    case "not_found":
      return { kind: "not_found", text: "Not found in Home Assistant" }
    case "unknown":
      return { kind: "unknown", text: "Couldn't check this against the safety policy." }
  }
}

/**
 * PA-01: a `speak` step's words are precached exactly like a macro's
 * reply (05-04's own backend implementation) -- delegating to
 * `deriveReplyPrecacheState` rather than re-deriving its three states.
 * That function only ever reads `.reply_cached` off its first argument,
 * so a step's own `speak_cached` flag is handed through under that same
 * shape; the cast is safe because no other `Macro` field is read.
 */
export function speakStepPrecacheState(step: WorkflowStep | null, synthesisFailedMessage: string | null): ReplyPrecacheState {
  const macroLike = step ? ({ reply_cached: step.speak_cached } as unknown as Macro) : null
  return deriveReplyPrecacheState(macroLike, synthesisFailedMessage)
}
