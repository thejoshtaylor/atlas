// D-17 backfill part B (08-10-PLAN.md), Task 1. `mock.module` replaces
// `@/lib/providers` before `ProvidersRoute` is imported -- a dynamic
// `import()` inside the test body is what makes that ordering hold
// (`SessionsRoute.test.tsx`'s established pattern, reused by every
// 08-09/08-10 mount test). Every `mock.module` call returns the full
// named-export set `@/lib/providers` carries, since the replacement is
// process-wide for the rest of this `bun test` run.
//
// Each test builds its own `QueryClient` rather than sharing the app's
// singleton, matching 08-09's own established pattern -- the mocked
// mutation's `onSuccess` calls `setQueryData` on that same test-scoped
// client, so a successful save is observed through a real cache write,
// not assumed.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function sampleSlot(overrides: Record<string, unknown> = {}) {
  return {
    slot: "stt",
    label: "Speech to text",
    selected: "vosk",
    active: "vosk",
    state: "running",
    reason: null,
    wrapped: false,
    measured_ms: null,
    selection_changed_since_boot: false,
    settings: {},
    options: [
      {
        name: "vosk",
        label: "Vosk",
        requires_credential: false,
        credential_set: false,
        wrapped: false,
        licence_note: null,
        needs_server_url: false,
        measured_note: null,
      },
      {
        name: "whisper",
        label: "Whisper",
        requires_credential: false,
        credential_set: false,
        wrapped: false,
        licence_note: null,
        needs_server_url: false,
        measured_note: null,
      },
    ],
    ...overrides,
  }
}

function stubProviders(
  queryClient: QueryClient,
  options: {
    fetchProviders: () => Promise<unknown>
    save?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/providers", () => ({
    PROVIDERS_QUERY_KEY: ["providers"],
    fetchProviders: options.fetchProviders,
    saveProvidersMutationOptions: {
      mutationFn: options.save ?? notStubbed("save"),
      onSuccess: (result: { slots: unknown[] }) => {
        queryClient.setQueryData(["providers"], { slots: result.slots })
      },
    },
  }))
}

function renderRoute(ProvidersRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <ProvidersRoute />
    </QueryClientProvider>,
  )
}

test("each slot renders its currently stored provider and its running status as two separate facts, never merged", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubProviders(queryClient, {
    fetchProviders: async () => ({
      slots: [sampleSlot({ selected: "whisper", active: "vosk", selection_changed_since_boot: false })],
    }),
  })
  const { ProvidersRoute } = await import("./ProvidersRoute")

  renderRoute(ProvidersRoute, queryClient)
  await screen.findByText("Speech to text")

  // The stored (draft) selection: the Whisper radio is checked.
  const whisperRadio = screen.getByRole("radio", { name: "Whisper" }) as HTMLInputElement
  expect(whisperRadio.getAttribute("aria-checked")).toBe("true")

  // The running status: a separate caption naming what actually booted,
  // never folded into the radio selection above.
  expect(screen.getByText("Currently running: Vosk.")).toBeTruthy()
  expect(screen.getByText("Needs restart")).toBeTruthy()
})

test("choosing a different provider for a slot and saving calls the mutation for that slot with the chosen name", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const saveCalls: unknown[] = []
  stubProviders(queryClient, {
    fetchProviders: async () => ({ slots: [sampleSlot()] }),
    save: async (input) => {
      saveCalls.push(input)
      return { slots: [sampleSlot({ selected: "whisper", active: "whisper" })], applies_live: false }
    },
  })
  const { ProvidersRoute } = await import("./ProvidersRoute")

  renderRoute(ProvidersRoute, queryClient)
  await screen.findByText("Speech to text")

  fireEvent.click(screen.getByRole("radio", { name: "Whisper" }))
  fireEvent.click(screen.getByRole("button", { name: "Save provider choices" }))

  await screen.findByText("Saved. Restart the assistant for this to take effect.")
  expect(saveCalls).toEqual([{ slots: { stt: { provider_name: "whisper", settings: {} } } }])
})

test("a successful save renders the restart-required message", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubProviders(queryClient, {
    fetchProviders: async () => ({ slots: [sampleSlot()] }),
    save: async () => ({ slots: [sampleSlot({ selected: "whisper", active: "whisper" })], applies_live: false }),
  })
  const { ProvidersRoute } = await import("./ProvidersRoute")

  renderRoute(ProvidersRoute, queryClient)
  await screen.findByText("Speech to text")

  fireEvent.click(screen.getByRole("radio", { name: "Whisper" }))
  fireEvent.click(screen.getByRole("button", { name: "Save provider choices" }))

  expect(await screen.findByText("Saved. Restart the assistant for this to take effect.")).toBeTruthy()
})

test("a failed save leaves the chosen value on screen rather than reverting it", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubProviders(queryClient, {
    fetchProviders: async () => ({ slots: [sampleSlot()] }),
    save: async () => {
      throw new Error("boom")
    },
  })
  const { ProvidersRoute } = await import("./ProvidersRoute")

  renderRoute(ProvidersRoute, queryClient)
  await screen.findByText("Speech to text")

  fireEvent.click(screen.getByRole("radio", { name: "Whisper" }))
  fireEvent.click(screen.getByRole("button", { name: "Save provider choices" }))

  expect(await screen.findByText("Couldn't save your provider choices. Try again.")).toBeTruthy()
  const whisperRadio = screen.getByRole("radio", { name: "Whisper" }) as HTMLInputElement
  expect(whisperRadio.getAttribute("aria-checked")).toBe("true")
})

test("a degraded slot renders the server's own reason string verbatim, not a paraphrase", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubProviders(queryClient, {
    fetchProviders: async () => ({
      slots: [
        sampleSlot({
          state: "degraded",
          active: null,
          reason: "Missing OPENAI_API_KEY environment variable",
          selection_changed_since_boot: false,
        }),
      ],
    }),
  })
  const { ProvidersRoute } = await import("./ProvidersRoute")

  renderRoute(ProvidersRoute, queryClient)
  await screen.findByText("Speech to text")

  expect(screen.getByText("Missing OPENAI_API_KEY environment variable")).toBeTruthy()
})

test("a slot the operator has not changed is not submitted", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const saveCalls: unknown[] = []
  stubProviders(queryClient, {
    fetchProviders: async () => ({
      slots: [
        sampleSlot({ slot: "stt", label: "Speech to text", selected: "vosk", active: "vosk" }),
        sampleSlot({
          slot: "tts",
          label: "Text to speech",
          selected: "piper",
          active: "piper",
          options: [
            {
              name: "piper",
              label: "Piper",
              requires_credential: false,
              credential_set: false,
              wrapped: false,
              licence_note: null,
              needs_server_url: false,
              measured_note: null,
            },
            {
              name: "elevenlabs",
              label: "ElevenLabs",
              requires_credential: true,
              credential_set: true,
              wrapped: false,
              licence_note: null,
              needs_server_url: false,
              measured_note: null,
            },
          ],
        }),
      ],
    }),
    save: async (input) => {
      saveCalls.push(input)
      return { slots: [], applies_live: false }
    },
  })
  const { ProvidersRoute } = await import("./ProvidersRoute")

  renderRoute(ProvidersRoute, queryClient)
  await screen.findByText("Speech to text")
  await screen.findByText("Text to speech")

  // Only the speech-to-text slot is touched; text-to-speech is left alone.
  fireEvent.click(screen.getByRole("radio", { name: "Whisper" }))
  fireEvent.click(screen.getByRole("button", { name: "Save provider choices" }))

  await screen.findByText("Saved. Restart the assistant for this to take effect.")
  expect(saveCalls).toEqual([{ slots: { stt: { provider_name: "whisper", settings: {} } } }])
})
