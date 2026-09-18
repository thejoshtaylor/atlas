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
