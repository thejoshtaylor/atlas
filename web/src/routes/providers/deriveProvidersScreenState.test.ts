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
    selection_changed_since_boot: false,
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

  test("false for a degraded slot the admin has not touched -- that is a different fact", () => {
    expect(needsRestart(slot({ selected: "xai", active: null, state: "degraded" }))).toBe(false)
  })

  // --- WR-08 (code review) ---

  test("true for a degraded slot whose stored selection has moved since boot", () => {
    expect(
      needsRestart(
        slot({
          selected: "local",
          active: null,
          state: "degraded",
          selection_changed_since_boot: true,
        }),
      ),
    ).toBe(true)
  })

  test("true when only the slot's settings moved, with the provider name unchanged", () => {
    // The exact walk-through WR-08 describes: the language-model slot is
    // set to `local` with no server URL, boot degrades it, the admin
    // types the URL and saves. The provider name never changes, so a
    // name comparison can say nothing -- and `active` is null, so the
    // old `selected !== active` clause could not fire either.
    expect(
      needsRestart(
        slot({
          slot: "brain",
          selected: "local",
          active: null,
          state: "degraded",
          settings: { server_url: "http://localhost:8000/v1" },
          selection_changed_since_boot: true,
        }),
      ),
    ).toBe(true)
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

  // --- WR-08 (code review) ---

  test("a degraded slot the admin has just fixed gets the badge, with an honest caption", () => {
    expect(
      needsRestartBadge(
        slot({
          selected: "local",
          active: null,
          state: "degraded",
          selection_changed_since_boot: true,
        }),
      ),
    ).toEqual({
      badge: { badgeVariant: "secondary", badgeText: "Needs restart" },
      // Never omitted: a badge with no present-tense fact beside it is
      // the same "states neither" failure this finding is about.
      caption: "Currently running: nothing — this slot did not start.",
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

  // --- WR-08 (code review) ---

  test("the stale reason is withdrawn once the admin has stored a different configuration", () => {
    // The reason describes the configuration this process booted with.
    // Once the stored selection has moved, it describes a configuration
    // that is no longer saved anywhere -- rendering it beside the field
    // the admin just corrected is a stale claim about the present, not
    // the server's reason verbatim.
    expect(
      degradedBadge(
        slot({
          state: "degraded",
          reason: "Missing a server URL for Self-hosted (local). Add one in Settings.",
          selection_changed_since_boot: true,
        }),
      ),
    ).toBeNull()
  })

  test("an untouched degraded slot still shows its reason -- the fix withdraws nothing else", () => {
    expect(
      degradedBadge(
        slot({
          state: "degraded",
          reason: "Missing a server URL for Self-hosted (local). Add one in Settings.",
          selection_changed_since_boot: false,
        }),
      ),
    ).not.toBeNull()
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
