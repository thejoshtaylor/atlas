import { afterEach, describe, expect, test } from "bun:test"
import { queryClient } from "./queryClient"
import {
  PROVIDERS_QUERY_KEY,
  fetchProviders,
  saveProvidersMutationOptions,
  type ProviderSlot,
  type ProvidersState,
} from "./providers"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

function sampleSlot(overrides: Partial<ProviderSlot> = {}): ProviderSlot {
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

describe("providers.ts -- calls the real routes/providers.py paths", () => {
  test("fetchProviders calls GET /api/providers", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, { slots: [sampleSlot()] } satisfies ProvidersState)
    }) as typeof fetch
    await fetchProviders()
    expect(calledUrl).toBe("/api/providers")
  })

  test("saveProvidersMutationOptions PUTs the full three-slot body to /api/providers", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, { slots: [sampleSlot()], applies_live: false })
    }) as typeof fetch

    await saveProvidersMutationOptions.mutationFn!(
      {
        slots: {
          stt: { provider_name: "xai" },
          tts: { provider_name: "xai" },
          brain: { provider_name: "local", settings: { server_url: "http://x:1/v1" } },
        },
      },
      {} as never,
    )

    expect(calledUrl).toBe("/api/providers")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({
      slots: {
        stt: { provider_name: "xai" },
        tts: { provider_name: "xai" },
        brain: { provider_name: "local", settings: { server_url: "http://x:1/v1" } },
      },
    })
  })
})

describe("saveProvidersMutationOptions -- onSuccess seeds the providers cache directly, no optimistic update", () => {
  test("the confirmed response replaces the cached slots, never a client-guessed shape", () => {
    const confirmed = { slots: [sampleSlot({ selected: "xai", active: "xai" })], applies_live: false }
    saveProvidersMutationOptions.onSuccess?.(
      confirmed,
      { slots: { stt: { provider_name: "xai" } } },
      undefined,
      {} as never,
    )
    expect(queryClient.getQueryData(PROVIDERS_QUERY_KEY)).toEqual({ slots: confirmed.slots })
  })
})

describe("PROVIDERS_QUERY_KEY -- a stable key", () => {
  test("is the fixed tuple every query/mutation in this module reads and writes", () => {
    expect(PROVIDERS_QUERY_KEY).toEqual(["providers"])
  })
})
