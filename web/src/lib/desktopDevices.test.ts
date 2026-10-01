import { afterEach, describe, expect, test } from "bun:test"
import {
  buildPairLink,
  createDesktopDeviceMutationOptions,
  desktopDevicesQueryOptions,
  fetchDesktopDevices,
} from "./desktopDevices"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("desktopDevices.ts", () => {
  test("fetchDesktopDevices calls GET /api/desktop-devices", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [])
    }) as typeof fetch
    await fetchDesktopDevices()
    expect(calledUrl).toBe("/api/desktop-devices")
  })

  test("the list query has the desktop-devices key and refetches every 5 seconds", () => {
    expect(desktopDevicesQueryOptions.queryKey).toEqual(["desktop-devices"])
    expect(desktopDevicesQueryOptions.refetchInterval).toBe(5000)
  })

  test("createDesktopDeviceMutationOptions POSTs {name} and returns the token", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(201, { id: 1, name: "desk", token: "t-1", created_at: "2026-10-01T00:00:00Z" })
    }) as typeof fetch
    const created = await createDesktopDeviceMutationOptions.mutationFn!({ name: "desk" }, {} as never)
    expect(calledUrl).toBe("/api/desktop-devices")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ name: "desk" })
    expect(created.token).toBe("t-1")
  })

  test("the create onSuccess only invalidates the list and never writes the token to the cache", async () => {
    const invalidated: unknown[] = []
    const written: unknown[] = []
    const originalInvalidate = queryClient.invalidateQueries.bind(queryClient)
    const originalSet = queryClient.setQueryData.bind(queryClient)
    queryClient.invalidateQueries = ((filters: unknown) => {
      invalidated.push(filters)
      return Promise.resolve()
    }) as typeof queryClient.invalidateQueries
    queryClient.setQueryData = ((...args: unknown[]) => {
      written.push(args)
      return undefined
    }) as typeof queryClient.setQueryData
    try {
      await (createDesktopDeviceMutationOptions.onSuccess as (...a: unknown[]) => unknown)(
        { id: 1, name: "desk", token: "t-1", created_at: "x" },
        { name: "desk" },
        undefined,
        {},
      )
    } finally {
      queryClient.invalidateQueries = originalInvalidate
      queryClient.setQueryData = originalSet
    }
    expect(invalidated).toEqual([{ queryKey: ["desktop-devices"] }])
    expect(written).toEqual([])
  })

  test("buildPairLink encodes the host and the token", () => {
    expect(buildPairLink("svr.test:8443", "t-1")).toBe("atlas://pair?server=svr.test%3A8443&token=t-1")
    expect(buildPairLink("svr.test", "a&b=c")).toBe("atlas://pair?server=svr.test&token=a%26b%3Dc")
  })
})
