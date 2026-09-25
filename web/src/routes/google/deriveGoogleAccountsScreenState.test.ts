import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import { HTTPS_REQUIRED_REASON, deriveGoogleAccountsScreenState, linkAvailability } from "./deriveGoogleAccountsScreenState"

function pending() {
  return { status: "pending" as const, data: undefined, error: undefined }
}
function success<T>(data: T) {
  return { status: "success" as const, data, error: undefined }
}
function errored(error: unknown) {
  return { status: "error" as const, data: undefined, error }
}

const sampleClient = {
  configured: true,
  client_id: "abc",
  updated_at: null,
  redirect_path: "/api/google/oauth/callback",
}

describe("deriveGoogleAccountsScreenState", () => {
  test("either query pending -> loading", () => {
    expect(deriveGoogleAccountsScreenState(pending(), success([]))).toEqual({ kind: "loading" })
    expect(deriveGoogleAccountsScreenState(success(sampleClient), pending())).toEqual({ kind: "loading" })
  })

  test("either query errored -> error with the server's message", () => {
    expect(deriveGoogleAccountsScreenState(errored(new ApiError(500, "boom")), success([]))).toEqual({
      kind: "error",
      message: "boom",
    })
    expect(deriveGoogleAccountsScreenState(success(sampleClient), errored(new ApiError(500, "boom")))).toEqual({
      kind: "error",
      message: "boom",
    })
  })

  test("both ready -> ready with client and accounts", () => {
    expect(deriveGoogleAccountsScreenState(success(sampleClient), success([]))).toEqual({
      kind: "ready",
      client: sampleClient,
      accounts: [],
    })
  })
})

describe("linkAvailability", () => {
  test("https: is available", () => {
    expect(linkAvailability("https:")).toEqual({ available: true })
  })

  test("http: is unavailable, with a named reason", () => {
    expect(linkAvailability("http:")).toEqual({ available: false, reason: HTTPS_REQUIRED_REASON })
  })
})
