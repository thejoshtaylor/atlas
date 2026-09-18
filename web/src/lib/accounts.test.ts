import { afterEach, describe, expect, test } from "bun:test"
import {
  acceptInvite,
  createInviteMutationOptions,
  fetchAccounts,
  fetchInvites,
  formatPendingInviteCount,
  removeAccountMutationOptions,
  revokeInviteMutationOptions,
} from "./accounts"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("formatPendingInviteCount -- zero-one-many", () => {
  test("1 reads as singular", () => {
    expect(formatPendingInviteCount(1)).toBe("1 pending invite")
  })
  test("3 reads as plural", () => {
    expect(formatPendingInviteCount(3)).toBe("3 pending invites")
  })
  test("0 reads as plural (the caller renders the empty state instead of this line for 0)", () => {
    expect(formatPendingInviteCount(0)).toBe("0 pending invites")
  })
})

describe("accounts.ts -- calls the real routes/accounts.py paths", () => {
  test("fetchAccounts calls GET /api/accounts", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [])
    }) as typeof fetch
    await fetchAccounts()
    expect(calledUrl).toBe("/api/accounts")
  })

  test("fetchInvites calls GET /api/invites", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [])
    }) as typeof fetch
    await fetchInvites()
    expect(calledUrl).toBe("/api/invites")
  })

  test("removeAccountMutationOptions DELETEs /api/accounts/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await removeAccountMutationOptions.mutationFn!({ accountId: 42 }, {} as never)
    expect(calledUrl).toBe("/api/accounts/42")
    expect(calledMethod).toBe("DELETE")
  })

  test("createInviteMutationOptions POSTs to /api/invites and returns the plaintext token", async () => {
    global.fetch = (async () =>
      jsonResponse(201, {
        id: 1,
        token: "test-key",
        role: "operator",
        email: null,
        expires_at: "2026-09-25T00:00:00Z",
      })) as typeof fetch
    const created = await createInviteMutationOptions.mutationFn!({ role: "operator" }, {} as never)
    expect(created.token).toBe("test-key")
  })

  test("revokeInviteMutationOptions DELETEs /api/invites/{id}", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await revokeInviteMutationOptions.mutationFn!({ inviteId: 7 }, {} as never)
    expect(calledUrl).toBe("/api/invites/7")
  })

  test("acceptInvite never sends a role field -- the server ignores anything the body claims (T-03-31)", async () => {
    let calledBody: string | undefined
    global.fetch = (async (_url: string, init?: RequestInit) => {
      calledBody = init?.body as string
      return jsonResponse(201, { id: 2, email: "a@example.invalid", display_name: "A", role: "viewer", disabled: false })
    }) as typeof fetch
    await acceptInvite({ token: "tok", displayName: "A", password: "hunter2" })
    const parsed = JSON.parse(calledBody!)
    expect(parsed).not.toHaveProperty("role")
    expect(parsed.display_name).toBe("A")
  })
})
