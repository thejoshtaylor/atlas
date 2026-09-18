import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { Policy, PolicyRule } from "@/lib/policy"
import { derivePolicyScreenState, formatRuleCount, visibleRules, type QueryLike } from "./derivePolicyScreenState"

function pending(): QueryLike<Policy> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored(error: unknown): QueryLike<Policy> {
  return { status: "error", data: undefined, error }
}
function success(data: Policy): QueryLike<Policy> {
  return { status: "success", data, error: undefined }
}

function rule(overrides: Partial<PolicyRule>): PolicyRule {
  return {
    id: 1,
    kind: "deny_entity",
    value: "switch.example",
    note: null,
    created_at: "2026-09-17T00:00:00Z",
    resolved: true,
    ...overrides,
  }
}

describe("derivePolicyScreenState", () => {
  test("pending renders loading -- the list must show skeleton rows, never an empty flash", () => {
    expect(derivePolicyScreenState(pending())).toEqual({ kind: "loading" })
  })

  test("a failed load carries the Copywriting Contract's exact error text when the server gives none more specific", () => {
    const state = derivePolicyScreenState(errored(new Error("network down")))
    expect(state).toEqual({
      kind: "error",
      message: "Couldn't load the safety policy. Editing is disabled until this loads.",
    })
  })

  test("a failed load with a named server reason surfaces it unmodified", () => {
    const state = derivePolicyScreenState(errored(new ApiError(500, "database unreachable")))
    expect(state).toEqual({ kind: "error", message: "database unreachable" })
  })

  test("a successful load renders ready with the mode and every rule, resolved flag included", () => {
    const policy: Policy = {
      mode: "allow_all_except_denylist",
      applies_live: true,
      rules: [rule({ id: 1 }), rule({ id: 2, resolved: false })],
    }
    const state = derivePolicyScreenState(success(policy))
    expect(state).toEqual({ kind: "ready", mode: "allow_all_except_denylist", rules: policy.rules })
  })
})

describe("visibleRules -- deny-list mode shows the denylist, allow-list mode shows the allowlist", () => {
  const rules = [
    rule({ id: 1, kind: "deny_entity" }),
    rule({ id: 2, kind: "deny_pattern" }),
    rule({ id: 3, kind: "allow_entity" }),
    rule({ id: 4, kind: "allow_pattern" }),
  ]

  test("deny-list mode", () => {
    expect(visibleRules("allow_all_except_denylist", rules).map((r) => r.id)).toEqual([1, 2])
  })

  test("allow-list mode", () => {
    expect(visibleRules("allowlist_only", rules).map((r) => r.id)).toEqual([3, 4])
  })
})

describe("formatRuleCount -- zero-one-many", () => {
  test("1 reads as singular", () => {
    expect(formatRuleCount("allow_all_except_denylist", 1)).toBe("1 entity denied")
  })
  test("3 reads as plural, deny-list mode", () => {
    expect(formatRuleCount("allow_all_except_denylist", 3)).toBe("3 entities denied")
  })
  test("3 reads as plural, allow-list mode", () => {
    expect(formatRuleCount("allowlist_only", 3)).toBe("3 entities allowed")
  })
})
