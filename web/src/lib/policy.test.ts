import { afterEach, describe, expect, test } from "bun:test"
import { addRuleMutationOptions, fetchPolicy, POLICY_QUERY_KEY, removeRuleMutationOptions, setModeMutationOptions } from "./policy"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

const SAMPLE_POLICY = {
  mode: "allow_all_except_denylist" as const,
  applies_live: true,
  rules: [
    {
      id: 1,
      kind: "deny_entity" as const,
      value: "switch.example",
      note: null,
      created_at: "2026-09-17T00:00:00Z",
      resolved: true,
    },
  ],
}

describe("policy.ts -- calls the real routes/policy.py paths", () => {
  test("fetchPolicy calls GET /api/policy", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, SAMPLE_POLICY)
    }) as typeof fetch
    await fetchPolicy()
    expect(calledUrl).toBe("/api/policy")
  })

  test("addRuleMutationOptions POSTs to /api/policy/rules", async () => {
    let calledUrl: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledBody = init?.body as string
      return jsonResponse(201, SAMPLE_POLICY.rules[0])
    }) as typeof fetch
    await addRuleMutationOptions.mutationFn!({ kind: "deny_entity", value: "switch.example" }, {} as never)
    expect(calledUrl).toBe("/api/policy/rules")
    expect(JSON.parse(calledBody!)).toEqual({ kind: "deny_entity", value: "switch.example" })
  })

  test("removeRuleMutationOptions DELETEs /api/policy/rules/{id}", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await removeRuleMutationOptions.mutationFn!({ ruleId: 9 }, {} as never)
    expect(calledUrl).toBe("/api/policy/rules/9")
  })

  test("setModeMutationOptions PUTs to /api/policy/mode", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return jsonResponse(200, { ...SAMPLE_POLICY, mode: "allowlist_only" })
    }) as typeof fetch
    await setModeMutationOptions.mutationFn!({ mode: "allowlist_only" }, {} as never)
    expect(calledUrl).toBe("/api/policy/mode")
    expect(calledMethod).toBe("PUT")
  })
})

describe("setModeMutationOptions -- a failed switch leaves the cache untouched (the 'revert')", () => {
  test("no onMutate exists to optimistically flip the displayed mode before the server confirms", () => {
    expect(setModeMutationOptions.onMutate).toBeUndefined()
  })

  test("a rejected mode switch never calls onSuccess, so the cached policy is exactly what it was before", async () => {
    queryClient.setQueryData(POLICY_QUERY_KEY, SAMPLE_POLICY)
    global.fetch = (async () => jsonResponse(400, { detail: "unknown policy mode" })) as typeof fetch

    await expect(setModeMutationOptions.mutationFn!({ mode: "allowlist_only" }, {} as never)).rejects.toBeDefined()
    expect(queryClient.getQueryData(POLICY_QUERY_KEY)).toEqual(SAMPLE_POLICY)
  })

  test("a successful switch writes the server's own returned policy into the cache", async () => {
    queryClient.setQueryData(POLICY_QUERY_KEY, SAMPLE_POLICY)
    const updated = { ...SAMPLE_POLICY, mode: "allowlist_only" as const }
    global.fetch = (async () => jsonResponse(200, updated)) as typeof fetch

    const result = await setModeMutationOptions.mutationFn!({ mode: "allowlist_only" }, {} as never)
    setModeMutationOptions.onSuccess?.(result, { mode: "allowlist_only" }, undefined, {} as never)
    expect(queryClient.getQueryData(POLICY_QUERY_KEY)).toEqual(updated)
  })
})
