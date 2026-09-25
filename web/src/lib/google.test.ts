import { afterEach, describe, expect, test } from "bun:test"
import {
  LINK_ERROR_MESSAGES,
  fetchGoogleAccount,
  fetchGoogleAccounts,
  fetchGoogleClient,
  linkErrorMessage,
  refreshCalendarsMutationOptions,
  saveGoogleClientMutationOptions,
  setCalendarAccessMutationOptions,
  startGoogleLink,
  unlinkGoogleAccountMutationOptions,
  updateGoogleAccountMutationOptions,
  type LinkErrorCode,
} from "./google"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

function sampleAccount(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    label: "work",
    email: "operator@example.com",
    is_default: false,
    status: "ok",
    status_detail: null,
    refresh_token_expires_at: null,
    linked_at: "2026-09-01T00:00:00Z",
    calendars: [],
    plugin_state: null,
    ...overrides,
  }
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("google.ts -- calls the real routes/google_accounts.py paths", () => {
  test("fetchGoogleClient calls GET /api/google/client", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, {
        configured: false,
        client_id: null,
        updated_at: null,
        redirect_path: "/api/google/oauth/callback",
      })
    }) as typeof fetch

    const result = await fetchGoogleClient()
    expect(calledUrl).toBe("/api/google/client")
    expect(result.configured).toBe(false)
  })

  test("saveGoogleClientMutationOptions PUTs to /api/google/client with client_id and client_secret", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, {
        configured: true,
        client_id: "abc",
        updated_at: "2026-09-24T00:00:00Z",
        redirect_path: "/api/google/oauth/callback",
      })
    }) as typeof fetch

    const result = await saveGoogleClientMutationOptions.mutationFn!(
      { client_id: "abc", client_secret: "shh" },
      {} as never,
    )
    expect(calledUrl).toBe("/api/google/client")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({ client_id: "abc", client_secret: "shh" })
    expect(result.client_id).toBe("abc")
  })

  test("startGoogleLink POSTs to /api/google/oauth/start with label and relink_account_id", async () => {
    let calledUrl: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledBody = init?.body as string
      return jsonResponse(200, { authorization_url: "https://accounts.google.com/o/oauth2/v2/auth?x=1" })
    }) as typeof fetch

    const result = await startGoogleLink({ label: "work", relink_account_id: null })
    expect(calledUrl).toBe("/api/google/oauth/start")
    expect(JSON.parse(calledBody!)).toEqual({ label: "work", relink_account_id: null })
    expect(result.authorization_url).toBe("https://accounts.google.com/o/oauth2/v2/auth?x=1")
  })

  test("fetchGoogleAccounts calls GET /api/google/accounts", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [])
    }) as typeof fetch

    await fetchGoogleAccounts()
    expect(calledUrl).toBe("/api/google/accounts")
  })

  test("fetchGoogleAccount calls GET /api/google/accounts/{id}", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, sampleAccount())
    }) as typeof fetch

    await fetchGoogleAccount(7)
    expect(calledUrl).toBe("/api/google/accounts/7")
  })

  test("updateGoogleAccountMutationOptions PATCHes /api/google/accounts/{id}, excluding accountId from the body", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, sampleAccount({ id: 7, label: "home" }))
    }) as typeof fetch

    await updateGoogleAccountMutationOptions.mutationFn!({ accountId: 7, label: "home" }, {} as never)
    expect(calledUrl).toBe("/api/google/accounts/7")
    expect(calledMethod).toBe("PATCH")
    expect(JSON.parse(calledBody!)).toEqual({ label: "home" })
  })

  test("setCalendarAccessMutationOptions PUTs /api/google/accounts/{id}/calendars/{calendar_id} with access", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, sampleAccount())
    }) as typeof fetch

    await setCalendarAccessMutationOptions.mutationFn!({ accountId: 7, calendarId: 3, access: "read_only" }, {} as never)
    expect(calledUrl).toBe("/api/google/accounts/7/calendars/3")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({ access: "read_only" })
  })

  test("refreshCalendarsMutationOptions POSTs /api/google/accounts/{id}/calendars/refresh", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return jsonResponse(200, sampleAccount())
    }) as typeof fetch

    await refreshCalendarsMutationOptions.mutationFn!({ accountId: 7 }, {} as never)
    expect(calledUrl).toBe("/api/google/accounts/7/calendars/refresh")
    expect(calledMethod).toBe("POST")
  })

  test("unlinkGoogleAccountMutationOptions DELETEs /api/google/accounts/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch

    await unlinkGoogleAccountMutationOptions.mutationFn!({ accountId: 7 }, {} as never)
    expect(calledUrl).toBe("/api/google/accounts/7")
    expect(calledMethod).toBe("DELETE")
  })
})

describe("linkErrorMessage", () => {
  test("null shows nothing", () => {
    expect(linkErrorMessage(null)).toBeNull()
  })

  const codes: LinkErrorCode[] = [
    "denied",
    "state_invalid",
    "client_missing",
    "exchange_failed",
    "no_refresh_token",
    "scopes_missing",
    "profile_failed",
    "label_taken",
  ]

  for (const code of codes) {
    test(`${code} has its own sentence`, () => {
      const message = linkErrorMessage(code)
      expect(message).toBe(LINK_ERROR_MESSAGES[code])
      expect(message).not.toBe(code)
    })
  }

  test("scopes_missing tells the operator to leave every box ticked", () => {
    expect(linkErrorMessage("scopes_missing")).toContain("leave every box ticked")
  })

  test("an unknown code shows a generic sentence, never the raw code alone", () => {
    const message = linkErrorMessage("not_a_real_code")
    expect(message).not.toBeNull()
    expect(message).not.toBe("not_a_real_code")
  })
})
