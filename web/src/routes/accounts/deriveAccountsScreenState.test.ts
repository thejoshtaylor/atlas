import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import { deriveAccountsScreenState, type QueryLike } from "./deriveAccountsScreenState"

function pending<T>(): QueryLike<T> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored<T>(error: unknown): QueryLike<T> {
  return { status: "error", data: undefined, error }
}
function success<T>(data: T): QueryLike<T> {
  return { status: "success", data, error: undefined }
}

describe("deriveAccountsScreenState", () => {
  test("either query pending renders loading -- never a mid-load empty list", () => {
    expect(deriveAccountsScreenState({ accounts: pending(), invites: success([]) })).toEqual({ kind: "loading" })
    expect(deriveAccountsScreenState({ accounts: success([]), invites: pending() })).toEqual({ kind: "loading" })
  })

  test("a failed accounts load disables the whole screen, even if invites loaded fine", () => {
    const state = deriveAccountsScreenState({
      accounts: errored(new ApiError(500, "database unreachable")),
      invites: success([]),
    })
    expect(state).toEqual({ kind: "error", message: "database unreachable" })
  })

  test("a failed invites load disables the whole screen, even if accounts loaded fine", () => {
    const state = deriveAccountsScreenState({
      accounts: success([]),
      invites: errored(new ApiError(500, "database unreachable")),
    })
    expect(state).toEqual({ kind: "error", message: "database unreachable" })
  })

  test("both settled successfully renders ready with both lists", () => {
    const state = deriveAccountsScreenState({
      accounts: success([{ id: 1, email: "a@example.invalid", display_name: "A", role: "admin", disabled: false }]),
      invites: success([]),
    })
    expect(state.kind).toBe("ready")
  })
})
