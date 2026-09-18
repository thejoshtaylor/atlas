// The safety policy surface: the two modes, the denylist/allowlist rules
// in each, and the mode switch's audit trail (SAFE-06, SAFE-07). Every
// shape here matches `src/spire_voice/routes/policy.py`'s own response
// models field-for-field.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

export type Mode = "allow_all_except_denylist" | "allowlist_only"

export type RuleKind = "deny_entity" | "deny_pattern" | "allow_entity" | "allow_pattern"

/** `PolicyRuleResponse`'s exact shape, including `resolved` -- true
 * unless this is an entity-kind rule the live Home Assistant catalog was
 * actually checked against and found absent from (T-03-54). */
export interface PolicyRule {
  id: number
  kind: RuleKind
  value: string
  note: string | null
  created_at: string
  resolved: boolean
}

/** `PolicyResponse`'s exact shape. `applies_live` is always `true`: a
 * policy write always respawns the enforcing child before the route
 * returns (`routes/policy.py`'s own docstring). */
export interface Policy {
  mode: Mode
  rules: PolicyRule[]
  applies_live: boolean
}

export const POLICY_QUERY_KEY = ["policy"] as const

export function fetchPolicy(): Promise<Policy> {
  return apiFetch<Policy>("/api/policy")
}

export interface AddRuleInput {
  kind: RuleKind
  value: string
  note?: string | null
}

/**
 * No optimistic update: the row this returns is only ever added to the
 * list after the server -- and the real MCP child behind it -- confirms
 * the write (this plan's own key link: every write awaits a real
 * respawn before returning, which is what makes the Live badge true).
 */
export const addRuleMutationOptions: UseMutationOptions<PolicyRule, unknown, AddRuleInput> = {
  mutationFn: (input) => apiFetch<PolicyRule>("/api/policy/rules", { method: "POST", body: input }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: POLICY_QUERY_KEY })
  },
}

export interface RemoveRuleInput {
  ruleId: number
}

export const removeRuleMutationOptions: UseMutationOptions<void, unknown, RemoveRuleInput> = {
  mutationFn: ({ ruleId }) => apiFetch<void>(`/api/policy/rules/${ruleId}`, { method: "DELETE" }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: POLICY_QUERY_KEY })
  },
}

export interface SetModeInput {
  mode: Mode
}

/**
 * No `onMutate` -- deliberately. The mode control's displayed value
 * comes straight from `POLICY_QUERY_KEY`'s cached data, which this
 * mutation only ever touches in `onSuccess`. A failed switch therefore
 * "reverts" by construction: the cache was never touched, so there is
 * nothing to revert (this plan's own acceptance criteria, proven
 * directly in `policy.test.ts` against the query cache).
 */
export const setModeMutationOptions: UseMutationOptions<Policy, unknown, SetModeInput> = {
  mutationFn: (input) => apiFetch<Policy>("/api/policy/mode", { method: "PUT", body: input }),
  onSuccess: (policy) => {
    queryClient.setQueryData(POLICY_QUERY_KEY, policy)
  },
}
