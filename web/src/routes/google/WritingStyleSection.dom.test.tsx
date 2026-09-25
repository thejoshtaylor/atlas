// 09-11-PLAN.md Task 1 (status + re-learn) and Task 2 (profile editor,
// samples, signature). `mock.module` replaces the FULL surface of
// `@/lib/google` -- `GoogleAccountRoute.dom.test.tsx`'s own header comment
// explains why: this specifier is shared process-wide across every test
// file in one `bun test` run, so a sibling file's static import of
// `@/lib/google` must still find every export it expects, not only the
// ones this file's own tests exercise.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"

import { LINK_ERROR_MESSAGES, linkErrorMessage } from "@/lib/google"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function sampleStyle(overrides: Record<string, unknown> = {}) {
  return {
    status: "ready",
    status_detail: null,
    profile: "Direct and friendly, short sentences.",
    samples: [],
    signature_text: null,
    messages_scanned: 187,
    learned_at: "2026-09-20T00:00:00Z",
    ...overrides,
  }
}

function stubGoogle(
  queryClient: QueryClient,
  options: {
    fetchGoogleStyle: (id: number) => Promise<unknown>
    saveGoogleStyle?: (input: unknown) => Promise<unknown>
    relearnGoogleStyle?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  const googleStyleQueryKey = (id: number) => ["google", "accounts", id, "style"]
  mock.module("@/lib/google", () => ({
    GOOGLE_CLIENT_QUERY_KEY: ["google", "client"],
    GOOGLE_ACCOUNTS_QUERY_KEY: ["google", "accounts"],
    googleAccountQueryKey: (id: number) => ["google", "accounts", id],
    fetchGoogleClient: notStubbed("fetchGoogleClient"),
    fetchGoogleAccounts: notStubbed("fetchGoogleAccounts"),
    fetchGoogleAccount: notStubbed("fetchGoogleAccount"),
    saveGoogleClientMutationOptions: { mutationFn: notStubbed("saveGoogleClient"), onSuccess: () => {} },
    startGoogleLink: notStubbed("startGoogleLink"),
    updateGoogleAccountMutationOptions: { mutationFn: notStubbed("updateGoogleAccount"), onSuccess: () => {} },
    setCalendarAccessMutationOptions: { mutationFn: notStubbed("setCalendarAccess"), onSuccess: () => {} },
    refreshCalendarsMutationOptions: { mutationFn: notStubbed("refreshCalendars"), onSuccess: () => {} },
    unlinkGoogleAccountMutationOptions: { mutationFn: notStubbed("unlinkGoogleAccount"), onSuccess: () => {} },
    googleStyleQueryKey,
    fetchGoogleStyle: options.fetchGoogleStyle,
    saveGoogleStyleMutationOptions: {
      mutationFn: options.saveGoogleStyle ?? notStubbed("saveGoogleStyle"),
      onSuccess: (style: unknown, input: { accountId: number }) => {
        queryClient.setQueryData(googleStyleQueryKey(input.accountId), style)
      },
    },
    relearnGoogleStyleMutationOptions: {
      mutationFn: options.relearnGoogleStyle ?? notStubbed("relearnGoogleStyle"),
      onSuccess: (style: unknown, input: { accountId: number }) => {
        queryClient.setQueryData(googleStyleQueryKey(input.accountId), style)
      },
    },
    LINK_ERROR_MESSAGES,
    linkErrorMessage,
  }))
}

function renderSection(WritingStyleSection: React.ComponentType<{ accountId: number }>, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <WritingStyleSection accountId={7} />
    </QueryClientProvider>,
  )
}

test("not_learned shows 'Not learned yet.'", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, { fetchGoogleStyle: async () => sampleStyle({ status: "not_learned" }) })
  const { WritingStyleSection } = await import("./WritingStyleSection")

  renderSection(WritingStyleSection, queryClient)

  expect(await screen.findByText("Not learned yet.")).toBeTruthy()
})

test("learning shows 'Learning from your Sent mail...' and disables the re-learn button", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, { fetchGoogleStyle: async () => sampleStyle({ status: "learning" }) })
  const { WritingStyleSection } = await import("./WritingStyleSection")

  renderSection(WritingStyleSection, queryClient)

  expect(await screen.findByText("Learning from your Sent mail...")).toBeTruthy()
  const button = screen.getByRole("button", { name: "Re-learn style" }) as HTMLButtonElement
  expect(button.disabled).toBe(true)
})

test("ready shows the message count and the learned date", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleStyle: async () => sampleStyle({ status: "ready", messages_scanned: 187, learned_at: "2026-09-20T00:00:00Z" }),
  })
  const { WritingStyleSection } = await import("./WritingStyleSection")

  renderSection(WritingStyleSection, queryClient)

  const text = await screen.findByText(/Learned from 187 sent messages on/)
  expect(text).toBeTruthy()
})

test("failed shows the server's status_detail verbatim", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleStyle: async () => sampleStyle({ status: "failed", status_detail: "no Sent messages found" }),
  })
  const { WritingStyleSection } = await import("./WritingStyleSection")

  renderSection(WritingStyleSection, queryClient)

  expect(await screen.findByText("Could not learn the style: no Sent messages found")).toBeTruthy()
})

test("Re-learn style opens a dialog with the replace warning; Cancel sends nothing", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const relearnCalls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleStyle: async () => sampleStyle({ status: "ready" }),
    relearnGoogleStyle: async (input) => {
      relearnCalls.push(input)
      return sampleStyle({ status: "learning" })
    },
  })
  const { WritingStyleSection } = await import("./WritingStyleSection")

  renderSection(WritingStyleSection, queryClient)

  fireEvent.click(await screen.findByRole("button", { name: "Re-learn style" }))
  expect(
    await screen.findByText(
      "This reads the latest Sent mail again and replaces the profile, including your edits.",
    ),
  ).toBeTruthy()

  fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
  expect(relearnCalls).toEqual([])
})

test("confirming Re-learn sends one POST and shows the learning state", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const relearnCalls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleStyle: async () => sampleStyle({ status: "ready" }),
    relearnGoogleStyle: async (input) => {
      relearnCalls.push(input)
      return sampleStyle({ status: "learning" })
    },
  })
  const { WritingStyleSection } = await import("./WritingStyleSection")

  renderSection(WritingStyleSection, queryClient)

  fireEvent.click(await screen.findByRole("button", { name: "Re-learn style" }))
  fireEvent.click(screen.getByRole("button", { name: "Re-learn" }))

  await waitFor(() => expect(relearnCalls).toEqual([{ accountId: 7 }]))
  expect(await screen.findByText("Learning from your Sent mail...")).toBeTruthy()
})

test("a 409 from re-learn shows the server's text verbatim", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleStyle: async () => sampleStyle({ status: "ready" }),
    relearnGoogleStyle: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(409, "a re-learn is already running for this account")
    },
  })
  const { WritingStyleSection } = await import("./WritingStyleSection")

  renderSection(WritingStyleSection, queryClient)

  fireEvent.click(await screen.findByRole("button", { name: "Re-learn style" }))
  fireEvent.click(screen.getByRole("button", { name: "Re-learn" }))

  expect(await screen.findByText("a re-learn is already running for this account")).toBeTruthy()
})
