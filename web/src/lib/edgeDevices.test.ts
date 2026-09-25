import { afterEach, describe, expect, test } from "bun:test"
import { createEdgeDeviceMutationOptions, fetchEdgeDevices, revokeEdgeDeviceMutationOptions } from "./edgeDevices"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("edgeDevices.ts -- calls the real routes/edge_devices.py paths", () => {
  test("fetchEdgeDevices calls GET /api/edge-devices", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [])
    }) as typeof fetch
    await fetchEdgeDevices()
    expect(calledUrl).toBe("/api/edge-devices")
  })

  test("createEdgeDeviceMutationOptions POSTs {name} to /api/edge-devices and returns the plaintext token", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(201, { id: 1, name: "kitchen", token: "tok-abc", created_at: "2026-09-25T00:00:00Z" })
    }) as typeof fetch
    const created = await createEdgeDeviceMutationOptions.mutationFn!({ name: "kitchen" }, {} as never)
    expect(calledUrl).toBe("/api/edge-devices")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ name: "kitchen" })
    expect(created.token).toBe("tok-abc")
  })

  test("revokeEdgeDeviceMutationOptions DELETEs /api/edge-devices/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await revokeEdgeDeviceMutationOptions.mutationFn!({ deviceId: 7 }, {} as never)
    expect(calledUrl).toBe("/api/edge-devices/7")
    expect(calledMethod).toBe("DELETE")
  })
})
