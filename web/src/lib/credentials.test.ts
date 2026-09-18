import { afterEach, describe, expect, test } from "bun:test"
import { fetchCredentials, formatRelativeDate, saveCredentialMutationOptions } from "./credentials"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("credentials.ts -- calls the real routes/credentials.py paths", () => {
  test("fetchCredentials calls GET /api/credentials and never filters what the server returns", async () => {
    const listing = [
      { slot: "stt", label: "Speech to text", is_set: true, updated_at: "2026-09-01T00:00:00Z", source: "database", applies_live: false },
      // A slot this frontend has never heard of -- the settings surface
      // must render it rather than drop it (this plan's own "content
      // driven by the server's listing" rule).
      { slot: "a_future_slot", label: "A future provider", is_set: false, updated_at: null, source: "unset", applies_live: false },
    ]
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, listing)
    }) as typeof fetch

    const result = await fetchCredentials()
    expect(calledUrl).toBe("/api/credentials")
    expect(result).toEqual(listing)
    expect(result.find((e) => e.slot === "a_future_slot")).toBeDefined()
  })

  test("saveCredentialMutationOptions PUTs to /api/credentials/{slot} and never re-sends the value in its own return type", async () => {
    let calledUrl: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledBody = init?.body as string
      return jsonResponse(200, {
        slot: "stt",
        label: "Speech to text",
        is_set: true,
        updated_at: "2026-09-18T00:00:00Z",
        source: "database",
        applies_live: false,
      })
    }) as typeof fetch

    const result = await saveCredentialMutationOptions.mutationFn!({ slot: "stt", value: "secret-value" }, {} as never)
    expect(calledUrl).toBe("/api/credentials/stt")
    expect(JSON.parse(calledBody!)).toEqual({ value: "secret-value" })
    expect(result).not.toHaveProperty("value")
    expect(JSON.stringify(result)).not.toContain("secret-value")
  })
})

describe("formatRelativeDate", () => {
  test("a timestamp from 2 days ago reads as '2 days ago'", () => {
    const twoDaysAgo = new Date(Date.now() - 2 * 24 * 60 * 60 * 1000).toISOString()
    expect(formatRelativeDate(twoDaysAgo)).toBe("2 days ago")
  })
  test("a timestamp from a few minutes ago reads as 'less than an hour ago'", () => {
    const recently = new Date(Date.now() - 5 * 60 * 1000).toISOString()
    expect(formatRelativeDate(recently)).toBe("less than an hour ago")
  })
})
