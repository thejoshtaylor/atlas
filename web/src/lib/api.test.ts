import { afterEach, describe, expect, test } from "bun:test"
import { ApiError, SetupIncompleteError, UnauthorizedError, apiFetch } from "./api"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("apiFetch -- the fetch seam's three distinguishable outcomes", () => {
  test("a 401 throws UnauthorizedError and clears the cached session", async () => {
    queryClient.setQueryData(["session"], { email: "a@example.com", role: "admin" })
    global.fetch = (async () => jsonResponse(401, { detail: "Not authenticated." })) as typeof fetch

    await expect(apiFetch("/auth/session")).rejects.toBeInstanceOf(UnauthorizedError)
    expect(queryClient.getQueryData(["session"])).toBeNull()
  })

  test("a 503 naming setup incomplete throws SetupIncompleteError", async () => {
    global.fetch = (async () =>
      jsonResponse(503, { detail: "setup incomplete: no admin account exists yet" })) as typeof fetch

    await expect(apiFetch("/auth/session")).rejects.toBeInstanceOf(SetupIncompleteError)
  })

  test("a 503 that does NOT name setup incomplete throws a plain ApiError, not SetupIncompleteError", async () => {
    global.fetch = (async () => jsonResponse(503, { detail: "database unreachable" })) as typeof fetch

    const rejection = apiFetch("/policy")
    await expect(rejection).rejects.toBeInstanceOf(ApiError)
    await expect(rejection).rejects.not.toBeInstanceOf(SetupIncompleteError)
  })

  test("any other non-2xx throws ApiError carrying the server's own detail string", async () => {
    global.fetch = (async () => jsonResponse(422, { detail: "hub URL is not reachable" })) as typeof fetch

    await expect(apiFetch("/wizard/hub")).rejects.toMatchObject({
      name: "ApiError",
      status: 422,
      message: "hub URL is not reachable",
    })
  })

  test("a 2xx response resolves with the parsed JSON body", async () => {
    global.fetch = (async () => jsonResponse(200, { email: "a@example.com", role: "operator" })) as typeof fetch

    await expect(apiFetch("/auth/session")).resolves.toEqual({
      email: "a@example.com",
      role: "operator",
    })
  })
})

describe("apiFetch -- refresh-on-401", () => {
  test("401 then a successful refresh then a successful retry resolves with the retry body, session not cleared", async () => {
    queryClient.setQueryData(["session"], { email: "a@example.com", role: "admin" })
    const calls: string[] = []
    global.fetch = (async (input: RequestInfo | URL) => {
      const path = String(input)
      calls.push(path)
      if (path === "/api/auth/refresh") return jsonResponse(200, { ok: true })
      if (calls.length === 1) return jsonResponse(401, { detail: "Not authenticated." })
      return jsonResponse(200, { data: "after refresh" })
    }) as typeof fetch

    await expect(apiFetch("/api/policy")).resolves.toEqual({ data: "after refresh" })
    expect(calls).toEqual(["/api/policy", "/api/auth/refresh", "/api/policy"])
    expect(queryClient.getQueryData(["session"])).toEqual({ email: "a@example.com", role: "admin" })
  })

  test("401 then a failed refresh throws UnauthorizedError, clears the session, and never retries", async () => {
    queryClient.setQueryData(["session"], { email: "a@example.com", role: "admin" })
    const calls: string[] = []
    global.fetch = (async (input: RequestInfo | URL) => {
      const path = String(input)
      calls.push(path)
      if (path === "/api/auth/refresh") return jsonResponse(401, { detail: "refresh token expired" })
      return jsonResponse(401, { detail: "Not authenticated." })
    }) as typeof fetch

    await expect(apiFetch("/api/policy")).rejects.toBeInstanceOf(UnauthorizedError)
    expect(calls).toEqual(["/api/policy", "/api/auth/refresh"])
    expect(queryClient.getQueryData(["session"])).toBeNull()
  })

  test("401 then a successful refresh then a 401 retry throws UnauthorizedError after exactly 3 fetch calls", async () => {
    let count = 0
    global.fetch = (async (input: RequestInfo | URL) => {
      const path = String(input)
      count += 1
      if (path === "/api/auth/refresh") return jsonResponse(200, { ok: true })
      return jsonResponse(401, { detail: "Not authenticated." })
    }) as typeof fetch

    await expect(apiFetch("/api/policy")).rejects.toBeInstanceOf(UnauthorizedError)
    expect(count).toBe(3)
  })

  test("two concurrent 401s share exactly one refresh call", async () => {
    let refreshCalls = 0
    let resolveRefresh: ((response: Response) => void) | undefined
    const refreshPromise = new Promise<Response>((resolve) => {
      resolveRefresh = resolve
    })

    const attempts: Record<string, number> = {}
    global.fetch = (async (input: RequestInfo | URL) => {
      const path = String(input)
      if (path === "/api/auth/refresh") {
        refreshCalls += 1
        return refreshPromise
      }
      attempts[path] = (attempts[path] ?? 0) + 1
      // First attempt at each policy path 401s (triggering the refresh);
      // the retry, made after the shared refresh resolves, succeeds.
      if (attempts[path] === 1) return jsonResponse(401, { detail: "Not authenticated." })
      return jsonResponse(200, { data: path })
    }) as typeof fetch

    const first = apiFetch("/api/policy-a")
    const second = apiFetch("/api/policy-b")

    // Let both calls reach their initial 401 and start (or join) a refresh
    // before the refresh itself resolves. A macrotask tick flushes every
    // pending microtask first, so both calls are guaranteed to have
    // reached the "awaiting the shared refresh" point.
    await new Promise((resolve) => setTimeout(resolve, 0))
    resolveRefresh?.(jsonResponse(200, { ok: true }))

    await expect(first).resolves.toEqual({ data: "/api/policy-a" })
    await expect(second).resolves.toEqual({ data: "/api/policy-b" })
    expect(refreshCalls).toBe(1)
  })

  test("a 401 from /api/auth/refresh itself does not trigger a refresh", async () => {
    const calls: string[] = []
    global.fetch = (async (input: RequestInfo | URL) => {
      calls.push(String(input))
      return jsonResponse(401, { detail: "refresh token expired" })
    }) as typeof fetch

    await expect(apiFetch("/api/auth/refresh", { method: "POST" })).rejects.toBeInstanceOf(UnauthorizedError)
    expect(calls).toEqual(["/api/auth/refresh"])
  })

  test("a 401 from /api/auth/login does not trigger a refresh", async () => {
    const calls: string[] = []
    global.fetch = (async (input: RequestInfo | URL) => {
      calls.push(String(input))
      return jsonResponse(401, { detail: "bad credentials" })
    }) as typeof fetch

    await expect(apiFetch("/api/auth/login", { method: "POST" })).rejects.toBeInstanceOf(UnauthorizedError)
    expect(calls).toEqual(["/api/auth/login"])
  })
})
