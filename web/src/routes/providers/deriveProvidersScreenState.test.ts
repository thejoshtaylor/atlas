import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { ProviderSlot, ProvidersState } from "@/lib/providers"
import {
  degradedBadge,
  deriveProvidersScreenState,
  needsRestart,
  needsRestartBadge,
  wrappedBadge,
  type QueryLike,
} from "./deriveProvidersScreenState"

function pending(): QueryLike<ProvidersState> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored(error: unknown): QueryLike<ProvidersState> {
  return { status: "error", data: undefined, error }
}
function success(data: ProvidersState): QueryLike<ProvidersState> {
  return { status: "success", data, error: undefined }
}

function slot(overrides: Partial<ProviderSlot> = {}): ProviderSlot {
  return {
    slot: "stt",
    label: "Speech to text",
    selected: "xai",
    active: "xai",
    state: "running",
    reason: null,
    wrapped: false,
    measured_ms: null,
    options: [
      {
        name: "xai",
        label: "xAI",
        requires_credential: true,
        credential_set: true,
        wrapped: false,
        licence_note: null,
        needs_server_url: false,
        measured_note: null,
      },
    ],
    settings: {},
    ...overrides,
  }
}

describe("deriveProvidersScreenState", () => {
  test("pending renders loading -- the screen must show skeleton rows, never an empty flash", () => {
    expect(deriveProvidersScreenState(pending())).toEqual({ kind: "loading" })
  })

  test("a failed load carries the Copywriting Contract's exact fallback text when the server gives none more specific", () => {
    expect(deriveProvidersScreenState(errored(new Error("network down")))).toEqual({
      kind: "error",
      message: "Couldn't load providers. Try again.",
    })
  })

  test("a failed load with a named server reason surfaces it unmodified", () => {
    expect(deriveProvidersScreenState(errored(new ApiError(500, "database unreachable")))).toEqual({
      kind: "error",
      message: "database unreachable",
    })
  })

  test("a successful load renders ready with every slot", () => {
    const slots = [slot()]
    expect(deriveProvidersScreenState(success({ slots }))).toEqual({ kind: "ready", slots })
  })
})

describe("needsRestart -- D-02's own divergence fact", () => {
  test("false when selected matches active", () => {
    expect(needsRestart(slot({ selected: "xai", active: "xai" }))).toBe(false)
  })

  test("true when selected differs from a running active", () => {
    expect(needsRestart(slot({ selected: "xai", active: "other", state: "running" }))).toBe(true)
  })

  test("false for a degraded slot -- that is a different fact, not a pending restart", () => {
    expect(needsRestart(slot({ selected: "xai", active: null, state: "degraded" }))).toBe(false)
  })
})

describe("needsRestartBadge -- the secondary badge plus the honest running-provider caption", () => {
  test("null in steady state -- no noise", () => {
    expect(needsRestartBadge(slot({ selected: "xai", active: "xai" }))).toBeNull()
  })

  test("names the running option's own label, not its raw name", () => {
    const withTwoOptions = slot({
      selected: "xai",
      active: "faster_whisper",
      options: [
        {
          name: "xai",
          label: "xAI",
          requires_credential: true,
          credential_set: true,
          wrapped: false,
          licence_note: null,
          needs_server_url: false,
          measured_note: null,
        },
        {
          name: "faster_whisper",
          label: "faster-whisper",
          requires_credential: false,
          credential_set: true,
          wrapped: false,
          licence_note: null,
          needs_server_url: false,
          measured_note: null,
        },
      ],
    })
    expect(needsRestartBadge(withTwoOptions)).toEqual({
      badge: { badgeVariant: "secondary", badgeText: "Needs restart" },
      caption: "Currently running: faster-whisper.",
    })
  })
})

describe("degradedBadge -- the denied badge plus the server's own reason, verbatim", () => {
  test("null when not degraded", () => {
    expect(degradedBadge(slot({ state: "running", reason: null }))).toBeNull()
  })

  test("carries the reason unchanged, no prefix, no paraphrase", () => {
    expect(degradedBadge(slot({ state: "degraded", reason: "Missing an API key for xAI. Add one in Settings." }))).toEqual({
      badge: { badgeVariant: "denied", badgeText: "Degraded — won't start" },
      caption: "Missing an API key for xAI. Add one in Settings.",
    })
  })
})

describe("wrappedBadge -- D-06/D-08's honest synthesis-to-first-chunk figure", () => {
  test("null for an unwrapped option", () => {
    expect(wrappedBadge(false, null)).toBeNull()
  })

  test("wrapped with no measurement yet states the honest absence rather than inventing a number", () => {
    expect(wrappedBadge(true, null)).toEqual({
      badge: { badgeVariant: "outline", badgeText: "Wrapped" },
      caption: "Renders the whole reply, then streams it out. No synthesis measured since the last restart.",
    })
  })

  test("wrapped with a measurement states it plainly", () => {
    expect(wrappedBadge(true, 812.4)).toEqual({
      badge: { badgeVariant: "outline", badgeText: "Wrapped" },
      caption: "Renders the whole reply, then streams it out. Measured: 812 ms synthesis-start to first chunk.",
    })
  })
})
