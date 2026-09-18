// WorkflowsRoute's own logic, kept apart from its JSX -- the same
// pattern `routes/macros/deriveMacrosScreenState.ts` and
// `routes/policy/derivePolicyScreenState.ts` already established.
// Imports nothing from React, for the same stated reason: what belongs
// here is only ever a fact about data, never a fact about a render.
import { ApiError } from "@/lib/api"
import type { WorkflowRun } from "@/lib/workflows"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type WorkflowsScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; runs: WorkflowRun[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load pending runs. Try again."
}

export function deriveWorkflowsScreenState(query: QueryLike<WorkflowRun[]>): WorkflowsScreenState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  const runs = query.data
  if (!runs) return { kind: "loading" }
  return { kind: "ready", runs }
}

/** "1 run" / "{n} runs" -- the same zero-one-many rule every other count
 * line in this product follows (`formatActionCount`, `formatRuleCount`). */
export function formatRunCount(count: number): string {
  return `${count} ${count === 1 ? "run" : "runs"}`
}

/** "1 step" / "{n} steps" under a run's summary -- 05-UI-SPEC.md's
 * zero-one-many row, verbatim. */
export function formatStepCount(count: number): string {
  return `${count} ${count === 1 ? "step" : "steps"}`
}

/** Only a step still `pending` can meaningfully conflict "right now" --
 * a step that has already fired, failed, or was skipped already
 * happened (or didn't) and re-litigating its conflict state at read time
 * would claim something about the future that is no longer true. */
function remainingSteps(run: WorkflowRun): WorkflowRun["steps"] {
  return run.steps.filter((step) => step.status === "pending")
}

/** `denied` and `not_found` both count as a conflict (the must-have's
 * "a denied, not-found or unreachable entity") -- `unknown` does not: an
 * unchecked step is not a known problem, it is an unknown one, and the
 * run-level summary specifically claims "would be refused right now,"
 * which an unknown check cannot yet claim. */
export function conflictedStepCount(run: WorkflowRun): number {
  return remainingSteps(run).filter((step) => step.conflict === "denied" || step.conflict === "not_found").length
}

/** "1 step would be refused right now" / "{n} steps would be refused
 * right now" -- 05-UI-SPEC.md's Copywriting Contract, verbatim. Only
 * ever rendered when `conflictedStepCount(run) > 0`; the caller is
 * responsible for that guard, matching `formatConflictSummary`'s own
 * macro-screen precedent. */
export function formatConflictSummary(count: number): string {
  return `${count} ${count === 1 ? "step" : "steps"} would be refused right now`
}

export type RunConflictDisplay =
  | { kind: "none" }
  | { kind: "denied"; text: string }
  | { kind: "unknown"; text: string }

/**
 * The run-level fire-time conflict flag -- `unknown` takes priority over
 * a known-clean count, because a policy or catalog read that failed
 * while this list was open must never render as "no conflict" (T-04-36's
 * rule, extended to this screen). A run with only `unknown` remaining
 * steps produces `conflictedStepCount(run) === 0` on its own, which
 * would otherwise silently look identical to a genuinely clean run --
 * this function is what prevents that.
 */
export function runConflictDisplay(run: WorkflowRun): RunConflictDisplay {
  const remaining = remainingSteps(run)
  if (remaining.some((step) => step.conflict === "unknown")) {
    return { kind: "unknown", text: "Couldn't check this against the safety policy." }
  }
  const conflicted = conflictedStepCount(run)
  if (conflicted > 0) return { kind: "denied", text: formatConflictSummary(conflicted) }
  return { kind: "none" }
}

export interface RunBadge {
  key: string
  label: string
}

/** The ordered `secondary` badges for a run row: origin (D-16), then
 * run status ("Running now" for a firing run, "Late" for one that came
 * due while the process was down, D-04). Both, one, or neither status
 * badge may apply; origin always does. */
export function runBadges(run: WorkflowRun): RunBadge[] {
  const badges: RunBadge[] = [{ key: "origin", label: run.origin === "spoken" ? "Spoken" : "Webapp" }]
  if (run.status === "firing") badges.push({ key: "firing", label: "Running now" })
  if (run.late) badges.push({ key: "late", label: "Late" })
  return badges
}

function formatElapsed(elapsedMs: number): string {
  const minutes = Math.max(1, Math.round(elapsedMs / 60_000))
  if (minutes < 60) return `${minutes} ${minutes === 1 ? "minute" : "minutes"}`
  const hours = Math.round(minutes / 60)
  if (hours < 24) return `${hours} ${hours === 1 ? "hour" : "hours"}`
  const days = Math.round(hours / 24)
  return `${days} ${days === 1 ? "day" : "days"}`
}

/** "Was due {relative time} ago." (D-04's own reasoning, stated
 * plainly) -- computed from the earliest still-pending step whose
 * `due_at` has already passed, relative to `now`. `null` when the run
 * is not late, or when `late` is true but no step's own `due_at` can
 * still be found in the past relative to `now` (a defensive case; `now`
 * should always be at least as current as the server's own
 * `late`-computing read). */
export function lateCaption(run: WorkflowRun, now: Date): string | null {
  if (!run.late) return null
  const overdueMs = remainingSteps(run)
    .map((step) => new Date(step.due_at).getTime())
    .filter((ms) => ms <= now.getTime())
  if (overdueMs.length === 0) return null
  return `Was due ${formatElapsed(now.getTime() - Math.min(...overdueMs))} ago.`
}

export interface CancelDialogCopy {
  body: string
  confirmLabel: string
}

/**
 * The one copy fork in 05-UI-SPEC.md: a firing run's cancel dialog says
 * any step already in progress will still finish but no further step
 * will run; a not-yet-firing run's says the run will not happen.
 * Silently implying an in-flight step can be un-fired would be the
 * CMD-07 failure this project keeps finding at exactly this kind of
 * seam.
 */
export function cancelDialogCopy(run: WorkflowRun): CancelDialogCopy {
  if (run.status === "firing") {
    return {
      body: `Cancel "${run.summary}"? Any step already in progress will still finish, but no further step will run. This cannot be undone.`,
      confirmLabel: "Cancel run",
    }
  }
  return {
    body: `Cancel "${run.summary}"? This run will not happen, and this cannot be undone.`,
    confirmLabel: "Cancel run",
  }
}
