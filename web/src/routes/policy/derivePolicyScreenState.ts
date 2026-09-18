// PolicyRoute's own logic, kept apart from its JSX -- this project has
// no rendered-component test infrastructure (see CalibrationRoute's own
// derive module, 03-06, for the established pattern this file copies).
import { ApiError } from "@/lib/api"
import type { Mode, Policy, PolicyRule } from "@/lib/policy"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type PolicyScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; mode: Mode; rules: PolicyRule[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load the safety policy. Editing is disabled until this loads."
}

export function derivePolicyScreenState(query: QueryLike<Policy>): PolicyScreenState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  const policy = query.data
  if (!policy) return { kind: "loading" }
  return { kind: "ready", mode: policy.mode, rules: policy.rules }
}

/**
 * The rules a given mode actually governs -- deny-list mode shows the
 * denylist (`deny_entity`/`deny_pattern`), allow-list mode shows the
 * allowlist (`allow_entity`/`allow_pattern`). A rule of the "other"
 * kind can still exist in storage (switching modes never deletes a
 * rule) but is not what this mode enforces, so it is not shown here.
 */
export function visibleRules(mode: Mode, rules: PolicyRule[]): PolicyRule[] {
  const kinds = mode === "allowlist_only" ? ["allow_entity", "allow_pattern"] : ["deny_entity", "deny_pattern"]
  return rules.filter((rule) => kinds.includes(rule.kind))
}

/**
 * "3 entities denied" / "1 entity denied" -- 03-UI-SPEC.md's
 * zero-one-many row, factored out the same way
 * `lib/accounts.ts::formatPendingInviteCount` is. `PolicyRoute` never
 * calls this for `count === 0`: the empty state renders instead.
 */
export function formatRuleCount(mode: Mode, count: number): string {
  const noun = count === 1 ? "entity" : "entities"
  const verb = mode === "allowlist_only" ? "allowed" : "denied"
  return `${count} ${noun} ${verb}`
}
