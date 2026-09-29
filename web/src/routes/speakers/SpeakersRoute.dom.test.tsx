// D-01 through D-04 (11-11-PLAN.md Task 1). `mock.module` replaces
// `@/lib/speakers` before `SpeakersRoute` is imported -- a dynamic
// `import()` inside the test body is what makes that ordering hold
// (`EdgeDevicesRoute.dom.test.tsx`'s established pattern).
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function sampleSpeaker(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    display_name: "Ann",
    linked_user_id: null,
    created_at: "2026-09-01T00:00:00Z",
    enrolled_phrases: 0,
    required_phrases: 5,
    model_id: "cam++",
    ...overrides,
  }
}

function stubSpeakers(
  queryClient: QueryClient,
  options: {
    fetchSpeakers: () => Promise<unknown[]>
    createSpeaker?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/speakers", () => ({
    SPEAKERS_QUERY_KEY: ["speakers"],
    speakersQueryOptions: {
      queryKey: ["speakers"],
      queryFn: options.fetchSpeakers,
    },
    createSpeakerMutationOptions: {
      mutationFn: options.createSpeaker ?? notStubbed("createSpeaker"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["speakers"] })
      },
    },
    deleteSpeakerMutationOptions: {
      mutationFn: notStubbed("deleteSpeaker"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["speakers"] })
      },
    },
  }))
}

function renderRoute(SpeakersRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <SpeakersRoute />
    </QueryClientProvider>,
  )
}

test("a mounted screen with two members shows each name and its enrollment progress, and null model_id reads 'Speaker ID model not set'", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubSpeakers(queryClient, {
    fetchSpeakers: async () => [
      sampleSpeaker({ id: 1, display_name: "Ann", enrolled_phrases: 3, required_phrases: 5 }),
      sampleSpeaker({ id: 2, display_name: "Ben", model_id: null, enrolled_phrases: 2 }),
    ],
  })
  const { SpeakersRoute } = await import("./SpeakersRoute")

  renderRoute(SpeakersRoute, queryClient)

  await screen.findByText("Ann")
  expect(screen.getByText("3 of 5 phrases")).toBeTruthy()
  expect(screen.getByText("Ben")).toBeTruthy()
  expect(screen.getByText("Speaker ID model not set")).toBeTruthy()
})

test("a fully enrolled member shows 'Enrolled'", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubSpeakers(queryClient, {
    fetchSpeakers: async () => [sampleSpeaker({ enrolled_phrases: 5, required_phrases: 5 })],
  })
  const { SpeakersRoute } = await import("./SpeakersRoute")

  renderRoute(SpeakersRoute, queryClient)

  await screen.findByText("Ann")
  expect(screen.getByText("Enrolled")).toBeTruthy()
})

test("an empty list shows the empty state copy", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubSpeakers(queryClient, { fetchSpeakers: async () => [] })
  const { SpeakersRoute } = await import("./SpeakersRoute")

  renderRoute(SpeakersRoute, queryClient)

  expect(await screen.findByText("No household members yet.")).toBeTruthy()
  expect(screen.getByText("Add a member, then record five phrases through the Pi.")).toBeTruthy()
})

test("adding 'Member B' calls the create mutation with {display_name, linked_user_id: null} and the list refetches", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let listCalls = 0
  let calledInput: unknown
  stubSpeakers(queryClient, {
    fetchSpeakers: async () => {
      listCalls += 1
      return []
    },
    createSpeaker: async (input) => {
      calledInput = input
      return sampleSpeaker({ id: 9, display_name: "Member B" })
    },
  })
  const { SpeakersRoute } = await import("./SpeakersRoute")

  renderRoute(SpeakersRoute, queryClient)

  await screen.findByText("No household members yet.")
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Member B" } })
  fireEvent.click(screen.getByRole("button", { name: "Add member" }))

  await waitFor(() => expect(calledInput).toEqual({ displayName: "Member B", linkedUserId: null }))
  await waitFor(() => expect(listCalls).toBeGreaterThan(1))
})

test("a 409 from the create route shows the server's own message under the form", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubSpeakers(queryClient, {
    fetchSpeakers: async () => [],
    createSpeaker: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(409, "a member with this display name already exists")
    },
  })
  const { SpeakersRoute } = await import("./SpeakersRoute")

  renderRoute(SpeakersRoute, queryClient)

  await screen.findByText("No household members yet.")
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Ann" } })
  fireEvent.click(screen.getByRole("button", { name: "Add member" }))

  expect(await screen.findByText("a member with this display name already exists")).toBeTruthy()
})
