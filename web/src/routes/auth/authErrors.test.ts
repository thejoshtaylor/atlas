import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import { classifyAcceptInviteError, classifyLoginError } from "./authErrors"

describe("classifyLoginError -- a wrong email and a wrong password read the same", () => {
  test("a 401 from an unknown-email attempt renders the exact Copywriting Contract text", () => {
    // The server's own `detail` for "no such user" -- deliberately not
    // surfaced verbatim, so this scenario cannot be told apart from the
    // next one by reading the rendered message.
    expect(classifyLoginError(new ApiError(401, "incorrect email or password"))).toBe(
      "Wrong email or password. Try again.",
    )
  })

  test("a 401 from a wrong-password attempt renders the identical text", () => {
    expect(classifyLoginError(new ApiError(401, "incorrect email or password"))).toBe(
      "Wrong email or password. Try again.",
    )
  })

  test("a non-401 ApiError is not rewritten into the login-error copy", () => {
    expect(classifyLoginError(new ApiError(500, "database unreachable"))).toBe("database unreachable")
  })
})

describe("classifyAcceptInviteError", () => {
  test("surfaces the server's own detail for an invalid/expired/already-used invite", () => {
    expect(classifyAcceptInviteError(new ApiError(400, "this invite is invalid, expired, or already used"))).toBe(
      "this invite is invalid, expired, or already used",
    )
  })

  test("a non-ApiError falls back to a generic message", () => {
    expect(classifyAcceptInviteError(new TypeError("network down"))).toBe(
      "This invite is invalid, expired, or already used.",
    )
  })
})
