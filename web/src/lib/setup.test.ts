import { afterEach, describe, expect, test } from "bun:test"
import { fetchSetupStatus, firstUnfinishedStep } from "./setup"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("firstUnfinishedStep -- the pure routing decision WizardRoute reads", () => {
  test("returns the first step whose complete is false, in the server's own order", () => {
    const steps = [
      { name: "admin_account", complete: true },
      { name: "hub", complete: false },
      { name: "provider_set", complete: false },
    ]
    expect(firstUnfinishedStep(steps)).toBe("hub")
  })

  test("returns null once every step is complete", () => {
    const steps = [
      { name: "admin_account", complete: true },
      { name: "hub", complete: true },
    ]
    expect(firstUnfinishedStep(steps)).toBeNull()
  })

  test("returns the very first step's own name when nothing is complete yet", () => {
    const steps = [{ name: "admin_account", complete: false }]
    expect(firstUnfinishedStep(steps)).toBe("admin_account")
  })
})

describe("fetchSetupStatus -- the unauthenticated status route", () => {
  test("resolves with the server's own complete/steps shape, unauthenticated", async () => {
    let calledPath: string | undefined
    global.fetch = (async (url: string) => {
      calledPath = url
      return jsonResponse(200, {
        complete: false,
        steps: [
          { name: "admin_account", complete: true },
          { name: "hub", complete: false },
        ],
      })
    }) as typeof fetch

    const status = await fetchSetupStatus()
    expect(calledPath).toBe("/api/setup/status")
    expect(status.complete).toBe(false)
    expect(status.steps).toHaveLength(2)
  })
})
