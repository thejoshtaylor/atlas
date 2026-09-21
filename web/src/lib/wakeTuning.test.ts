import { afterEach, describe, expect, test } from "bun:test"
import { queryClient } from "./queryClient"
import {
  WAKE_EVENTS_QUERY_KEY,
  fetchWakeEvents,
  setWakeThresholdMutationOptions,
  type WakeEventsResponse,
} from "./wakeTuning"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

function sampleResponse(overrides: Partial<WakeEventsResponse> = {}): WakeEventsResponse {
  return {
    events: [],
    engine: "vosk",
    engine_grades: false,
    threshold: 0.5,
    not_scored_session_count: 0,
    capped: false,
    ...overrides,
  }
}

describe("wakeTuning.ts -- calls the real routes/wake.py paths", () => {
  test("fetchWakeEvents calls GET /api/wake-events", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, sampleResponse())
    }) as typeof fetch
    const result = await fetchWakeEvents()
    expect(calledUrl).toBe("/api/wake-events")
    expect(result).toEqual(sampleResponse())
  })

  test("setWakeThresholdMutationOptions PUTs the threshold to /api/wake-threshold", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, { threshold: 0.62 })
    }) as typeof fetch
    const result = await setWakeThresholdMutationOptions.mutationFn!({ threshold: 0.62 }, {} as never)
    expect(calledUrl).toBe("/api/wake-threshold")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({ threshold: 0.62 })
    expect(result).toEqual({ threshold: 0.62 })
  })

  test("setWakeThresholdMutationOptions onSuccess invalidates the events query, never issuing a second fetch itself", async () => {
    let fetchCount = 0
    global.fetch = (async () => {
      fetchCount += 1
      return jsonResponse(200, sampleResponse())
    }) as typeof fetch

    const invalidated: unknown[] = []
    const originalInvalidate = queryClient.invalidateQueries.bind(queryClient)
    queryClient.invalidateQueries = (async (options: unknown) => {
      invalidated.push(options)
      return originalInvalidate(options as never)
    }) as typeof queryClient.invalidateQueries

    setWakeThresholdMutationOptions.onSuccess?.({ threshold: 0.7 }, { threshold: 0.7 }, undefined, {} as never)

    expect(invalidated).toEqual([{ queryKey: WAKE_EVENTS_QUERY_KEY }])
    // onSuccess itself performs no fetch -- invalidation only marks the
    // query stale for whichever component is mounted to re-request it.
    expect(fetchCount).toBe(0)

    queryClient.invalidateQueries = originalInvalidate
  })
})
