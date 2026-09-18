import { afterEach, describe, expect, test } from "bun:test"
import { fetchSession, loginMutationOptions, logoutMutationOptions, SESSION_QUERY_KEY } from "./session"
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

describe("session -- the real routes, not the placeholder ones api.ts used to point at", () => {
  test("fetchSession calls /api/auth/me, not /auth/session", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, { id: 1, email: "a@example.invalid", display_name: "A", role: "admin" })
    }) as typeof fetch

    await fetchSession()
    expect(calledUrl).toBe("/api/auth/me")
  })

  test("loginMutationOptions posts to /api/auth/login, not /auth/login", async () => {
    let calledUrl: string | undefined
    let calledInit: RequestInit | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledInit = init
      return jsonResponse(200, { id: 1, email: "a@example.invalid", display_name: "A", role: "operator" })
    }) as typeof fetch

    await loginMutationOptions.mutationFn!({ email: "a@example.invalid", password: "hunter2" }, {} as never)
    expect(calledUrl).toBe("/api/auth/login")
    expect(calledInit?.method).toBe("POST")
  })

  test("a successful login writes the session into the query cache under SESSION_QUERY_KEY", async () => {
    global.fetch = (async () =>
      jsonResponse(200, { id: 1, email: "a@example.invalid", display_name: "A", role: "viewer" })) as typeof fetch

    const session = await loginMutationOptions.mutationFn!(
      { email: "a@example.invalid", password: "hunter2" },
      {} as never,
    )
    loginMutationOptions.onSuccess?.(session, { email: "a@example.invalid", password: "hunter2" }, undefined, {} as never)
    expect(queryClient.getQueryData(SESSION_QUERY_KEY)).toEqual(session)
  })

  test("logoutMutationOptions posts to /api/auth/logout and clears the cached session", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return new Response(null, { status: 204 })
    }) as typeof fetch

    queryClient.setQueryData(SESSION_QUERY_KEY, { id: 1, email: "a@example.invalid", display_name: "A", role: "admin" })
    await logoutMutationOptions.mutationFn!(undefined, {} as never)
    logoutMutationOptions.onSuccess?.(undefined, undefined, undefined, {} as never)

    expect(calledUrl).toBe("/api/auth/logout")
    expect(queryClient.getQueryData(SESSION_QUERY_KEY)).toBeNull()
  })
})
