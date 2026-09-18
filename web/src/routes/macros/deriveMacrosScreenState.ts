// MacrosRoute's own logic, kept apart from its JSX -- this project has
// no rendered-component test infrastructure (see
// `routes/policy/derivePolicyScreenState.ts` for the established pattern
// this file copies). Imports nothing from React, for the same stated
// reason: what belongs here is only ever a fact about data, never a fact
// about a render.
import { ApiError } from "@/lib/api"
import type { ConflictAnnotation, Macro } from "@/lib/macros"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type MacrosScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; macros: Macro[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load macros. Try again."
}

export function deriveMacrosScreenState(query: QueryLike<Macro[]>): MacrosScreenState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  const macros = query.data
  if (!macros) return { kind: "loading" }
  return { kind: "ready", macros }
}

/**
 * "1 action" / "{n} actions" under a macro's phrase -- 03-UI-SPEC.md's
 * zero-one-many row, the same rule `derivePolicyScreenState.ts`'s
 * `formatRuleCount` and `lib/accounts.ts`'s `formatPendingInviteCount`
 * already follow.
 */
export function formatActionCount(count: number): string {
  return `${count} ${count === 1 ? "action" : "actions"}`
}

/** Which of a macro's saved actions currently conflict with the running
 * safety policy -- `denied` and `not_found` both count (04-UI-SPEC.md's
 * must-have: "a denied or unresolved entity"), `unknown` does not: an
 * unchecked action is not a known problem, it is an unknown one, and the
 * list-level summary is specifically "would be refused," which an
 * unknown check cannot yet claim. */
function isConflicted(conflict: ConflictAnnotation): boolean {
  return conflict === "denied" || conflict === "not_found"
}

export function conflictedActionCount(macro: Macro): number {
  return macro.actions.filter((action) => isConflicted(action.conflict)).length
}

/**
 * "1 action denied by policy" / "{n} actions denied by policy" --
 * 04-UI-SPEC.md's Copywriting Contract, verbatim. Only ever rendered
 * when `conflictedActionCount(macro) > 0`; the caller is responsible for
 * that guard, matching `formatRuleCount`'s own precedent of never being
 * called for a zero count.
 */
export function formatConflictSummary(count: number): string {
  return `${count} ${count === 1 ? "action" : "actions"} denied by policy`
}
