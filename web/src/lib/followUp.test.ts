import { afterEach, describe, expect, test } from "bun:test"
import { queryClient } from "./queryClient"
import { fetchFollowUpWindow, saveFollowUpWindowMutationOptions } from "./followUp"

// `web/` has no rendered-DOM test infrastructure for a pure fetch-layer
// module -- this exercises `fetchFollowUpWindow`/
// `saveFollowUpWindowMutationOptions`'s own request shape with
// `global.fetch` mocked, the same style `TimezoneField.test.ts`'s own
// `setTimezoneMutationOptions` block already establishes.

const originalFetch = global.fetch
afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("fetchFollowUpWindow -- GET /api/settings/follow-up-window", () => {
  test("fetches the route and returns its body untouched", async () => {
    let calledPath: string | undefined
    global.fetch = (async (url: string) => {
      calledPath = url
      return new Response(
        JSON.stringify({ window_s: 6, default_s: 6, resolved_from: "config" }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    }) as typeof fetch

    const status = await fetchFollowUpWindow()

    expect(calledPath).toBe("/api/settings/follow-up-window")
    expect(status).toEqual({ window_s: 6, default_s: 6, resolved_from: "config" })
  })
})

describe("saveFollowUpWindowMutationOptions -- PUT /api/settings/follow-up-window", () => {
  test("mutationFn sends window_s as the request body", async () => {
    let calledPath: string | undefined
    let calledMethod: string | undefined
    let sentBody: unknown
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledPath = url
      calledMethod = init?.method
      sentBody = init?.body ? JSON.parse(init.body as string) : undefined
      return new Response(
        JSON.stringify({ window_s: 8, default_s: 6, resolved_from: "database" }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    }) as typeof fetch

    const result = await saveFollowUpWindowMutationOptions.mutationFn!({ window_s: 8 }, {} as never)

    expect(calledPath).toBe("/api/settings/follow-up-window")
    expect(calledMethod).toBe("PUT")
    expect(sentBody).toEqual({ window_s: 8 })
    expect(result).toEqual({ window_s: 8, default_s: 6, resolved_from: "database" })
  })
})
