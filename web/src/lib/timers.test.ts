import { afterEach, describe, expect, test } from "bun:test"
import {
  createTimerMutationOptions,
  deleteTimerMutationOptions,
  fetchTimers,
  timersQueryOptions,
  updateTimerMutationOptions,
} from "./timers"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("timers.ts -- calls the real routes/timers.py paths", () => {
  test("fetchTimers calls GET /api/timers", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [])
    }) as typeof fetch
    await fetchTimers()
    expect(calledUrl).toBe("/api/timers")
    expect(timersQueryOptions.refetchInterval).toBe(5000)
  })

  test("create POSTs the body to /api/timers", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(201, { id: 1 })
    }) as typeof fetch
    await createTimerMutationOptions.mutationFn!({ kind: "timer", label: "pasta", duration_seconds: 600 }, {} as never)
    expect(calledUrl).toBe("/api/timers")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ kind: "timer", label: "pasta", duration_seconds: 600 })
  })

  test("update PATCHes /api/timers/{id} with only the changes body", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, { id: 4 })
    }) as typeof fetch
    await updateTimerMutationOptions.mutationFn!({ id: 4, changes: { paused: true } }, {} as never)
    expect(calledUrl).toBe("/api/timers/4")
    expect(calledMethod).toBe("PATCH")
    expect(JSON.parse(calledBody!)).toEqual({ paused: true })
  })

  test("delete DELETEs /api/timers/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await deleteTimerMutationOptions.mutationFn!({ id: 7 }, {} as never)
    expect(calledUrl).toBe("/api/timers/7")
    expect(calledMethod).toBe("DELETE")
  })
})
