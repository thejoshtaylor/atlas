import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import {
  READ_ONLY_CALENDAR_NOTE,
  accountStatusDisplay,
  calendarAccessOptions,
  deriveGoogleAccountState,
} from "./deriveGoogleAccountState"

function pending() {
  return { status: "pending" as const, data: undefined, error: undefined }
}
function success<T>(data: T) {
  return { status: "success" as const, data, error: undefined }
}
function errored(error: unknown) {
  return { status: "error" as const, data: undefined, error }
}

const sampleAccount = {
  id: 1,
  label: "work",
  email: "operator@example.com",
  is_default: false,
  status: "ok" as const,
  status_detail: null,
  refresh_token_expires_at: null,
  linked_at: "2026-09-01T00:00:00Z",
  calendars: [],
  plugin_state: null,
}

describe("deriveGoogleAccountState", () => {
  test("pending -> loading", () => {
    expect(deriveGoogleAccountState(pending())).toEqual({ kind: "loading" })
  })

  test("a 404 ApiError -> not_found, distinct from every other error", () => {
    expect(deriveGoogleAccountState(errored(new ApiError(404, "no google account with id 9")))).toEqual({
      kind: "not_found",
    })
  })

  test("a non-404 error -> error, with the server's own message", () => {
    expect(deriveGoogleAccountState(errored(new ApiError(500, "boom")))).toEqual({ kind: "error", message: "boom" })
  })

  test("success -> ready with the account", () => {
    expect(deriveGoogleAccountState(success(sampleAccount))).toEqual({ kind: "ready", account: sampleAccount })
  })
})

describe("calendarAccessOptions", () => {
  test("a writable calendar has three enabled options, in Off / Read only / Read and write order", () => {
    expect(calendarAccessOptions({ can_write: true })).toEqual([
      { value: "off", label: "Off", disabled: false },
      { value: "read_only", label: "Read only", disabled: false },
      { value: "read_write", label: "Read and write", disabled: false },
    ])
  })

  test("a read-only-at-google calendar disables Read and write only", () => {
    const options = calendarAccessOptions({ can_write: false })
    expect(options.find((o) => o.value === "off")!.disabled).toBe(false)
    expect(options.find((o) => o.value === "read_only")!.disabled).toBe(false)
    expect(options.find((o) => o.value === "read_write")!.disabled).toBe(true)
  })
})

describe("accountStatusDisplay", () => {
  test("every GoogleAccountStatus value maps to its own badge", () => {
    expect(accountStatusDisplay("ok")).toEqual({ badgeVariant: "secondary", badgeText: "Linked" })
    expect(accountStatusDisplay("needs_relink")).toEqual({ badgeVariant: "denied", badgeText: "Needs re-link" })
    expect(accountStatusDisplay("unreachable")).toEqual({ badgeVariant: "denied", badgeText: "Unreachable" })
  })
})

test("READ_ONLY_CALENDAR_NOTE is a real sentence, not empty", () => {
  expect(READ_ONLY_CALENDAR_NOTE.length).toBeGreaterThan(0)
})
