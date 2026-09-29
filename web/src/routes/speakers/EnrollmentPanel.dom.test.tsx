// D-02, D-03 (11-11-PLAN.md Task 2). `mock.module` replaces
// `@/lib/edgeDevices` and `@/lib/speakers` before `EnrollmentPanel` is
// imported -- a dynamic `import()` inside the test body is what makes
// that ordering hold (`EdgeDevicesRoute.dom.test.tsx`'s established
// pattern).
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

const PHRASES = ["the quick brown fox", "jumps over", "the lazy dog", "a fourth phrase", "a fifth phrase"]

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

function sampleDevice(overrides: Record<string, unknown> = {}) {
  return {
    id: 7,
    name: "kitchen",
    created_at: "2026-09-01T00:00:00Z",
    revoked: false,
    last_connected_at: null,
    connected: true,
    ...overrides,
  }
}

function stubEdgeDevices(devices: unknown[]) {
  mock.module("@/lib/edgeDevices", () => ({
    EDGE_DEVICES_QUERY_KEY: ["edge-devices"],
    edgeDevicesQueryOptions: {
      queryKey: ["edge-devices"],
      queryFn: async () => devices,
    },
  }))
}

function stubSpeakersLib(queryClient: QueryClient, enrollPhrase: (input: unknown) => Promise<unknown>) {
  mock.module("@/lib/speakers", () => ({
    SPEAKERS_QUERY_KEY: ["speakers"],
    enrollmentPhrasesQueryOptions: {
      queryKey: ["speakers", "enrollment-phrases"],
      queryFn: async () => ({ phrases: PHRASES }),
    },
    enrollPhraseMutationOptions: {
      mutationFn: enrollPhrase,
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["speakers"] })
      },
    },
  }))
}

function renderPanel(
  EnrollmentPanel: React.ComponentType<{ speaker: ReturnType<typeof sampleSpeaker> }>,
  queryClient: QueryClient,
  speaker = sampleSpeaker(),
) {
  return render(
    <QueryClientProvider client={queryClient}>
      <EnrollmentPanel speaker={speaker} />
    </QueryClientProvider>,
  )
}

test("with no connected device, the panel refuses to record and names why", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubEdgeDevices([sampleDevice({ connected: false })])
  stubSpeakersLib(queryClient, async () => {
    throw new Error("enrollPhrase must not be called")
  })
  const { EnrollmentPanel } = await import("./EnrollmentPanel")

  renderPanel(EnrollmentPanel, queryClient)

  expect(
    await screen.findByText("Connect a Pi microphone first. The Edge devices screen shows its status."),
  ).toBeTruthy()
  expect(screen.queryByRole("button", { name: "Record phrase" })).toBeNull()
})

test("picking a device and recording calls the enroll mutation with the right shape, shows Listening, then the recorded length", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubEdgeDevices([sampleDevice()])
  let calledInput: unknown
  let resolveEnroll!: (value: unknown) => void
  stubSpeakersLib(
    queryClient,
    (input) =>
      new Promise((resolve) => {
        calledInput = input
        resolveEnroll = resolve
      }),
  )
  const { EnrollmentPanel } = await import("./EnrollmentPanel")

  renderPanel(EnrollmentPanel, queryClient)

  fireEvent.click(await screen.findByLabelText("kitchen"))
  fireEvent.click(screen.getByRole("button", { name: "Record phrase" }))

  expect(await screen.findByText("Listening...")).toBeTruthy()
  await waitFor(() => expect(calledInput).toEqual({ speakerId: 1, phraseIndex: 0, deviceId: 7 }))

  resolveEnroll({ speaker_id: 1, phrase_index: 0, speech_ms: 2300, enrolled_phrases: 1, required_phrases: 5 })

  expect(await screen.findByText("Recorded 2.3s")).toBeTruthy()
})

test("a 422 shows the message and 'Try again' re-records the same phrase", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubEdgeDevices([sampleDevice()])
  let attempt = 0
  stubSpeakersLib(queryClient, async (input) => {
    attempt += 1
    if (attempt === 1) {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(422, "no speech was heard")
    }
    return {
      speaker_id: 1,
      phrase_index: (input as { phraseIndex: number }).phraseIndex,
      speech_ms: 1000,
      enrolled_phrases: 1,
      required_phrases: 5,
    }
  })
  const { EnrollmentPanel } = await import("./EnrollmentPanel")

  renderPanel(EnrollmentPanel, queryClient)

  fireEvent.click(await screen.findByLabelText("kitchen"))
  fireEvent.click(screen.getByRole("button", { name: "Record phrase" }))

  expect(await screen.findByText("no speech was heard")).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Try again" }))

  await waitFor(() => expect(attempt).toBe(2))
  expect(await screen.findByText("Recorded 1.0s")).toBeTruthy()
})

test("a 409 shows the server's own detail", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubEdgeDevices([sampleDevice()])
  stubSpeakersLib(queryClient, async () => {
    const { ApiError } = await import("@/lib/api")
    throw new ApiError(409, "another enrollment is already recording")
  })
  const { EnrollmentPanel } = await import("./EnrollmentPanel")

  renderPanel(EnrollmentPanel, queryClient)

  fireEvent.click(await screen.findByLabelText("kitchen"))
  fireEvent.click(screen.getByRole("button", { name: "Record phrase" }))

  expect(await screen.findByText("another enrollment is already recording")).toBeTruthy()
})

test("after the required phrase count is reached, the panel shows the enrolled message", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubEdgeDevices([sampleDevice()])
  stubSpeakersLib(queryClient, async () => ({
    speaker_id: 1,
    phrase_index: 0,
    speech_ms: 1500,
    enrolled_phrases: 5,
    required_phrases: 5,
  }))
  const { EnrollmentPanel } = await import("./EnrollmentPanel")

  renderPanel(EnrollmentPanel, queryClient)

  fireEvent.click(await screen.findByLabelText("kitchen"))
  fireEvent.click(screen.getByRole("button", { name: "Record phrase" }))

  fireEvent.click(await screen.findByRole("button", { name: "Continue" }))

  expect(await screen.findByText("Enrolled. This member is recognized on the next turn.")).toBeTruthy()
})
