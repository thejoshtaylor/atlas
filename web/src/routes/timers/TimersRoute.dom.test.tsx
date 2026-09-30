// `mock.module` replaces `@/lib/timers` before `TimersRoute` is imported; the
// dynamic `import()` in each test makes that ordering hold (the pattern of
// `EdgeDevicesRoute.dom.test.tsx`).
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function sampleTimer(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    kind: "timer",
    label: "pasta",
    enabled: true,
    paused: false,
    duration_seconds: 600,
    remaining_seconds: 600,
    time: null,
    days: [],
    next_fire_at: new Date(Date.now() + 600_000).toISOString(),
    created_at: "2027-01-01T12:00:00Z",
    ...overrides,
  }
}

function sampleAlarm(overrides: Record<string, unknown> = {}) {
  return sampleTimer({
    id: 2,
    kind: "alarm",
    label: "work",
    duration_seconds: null,
    remaining_seconds: null,
    time: "07:00",
    days: ["mon", "tue", "wed", "thu", "fri"],
    ...overrides,
  })
}

function stubTimers(
  queryClient: QueryClient,
  options: {
    fetchTimers: () => Promise<unknown[]>
    create?: (input: unknown) => Promise<unknown>
    update?: (input: unknown) => Promise<unknown>
    remove?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["timers"] })
  }
  mock.module("@/lib/timers", () => ({
    TIMERS_QUERY_KEY: ["timers"],
    WEEKDAYS: ["mon", "tue", "wed", "thu", "fri", "sat", "sun"],
    timersQueryOptions: { queryKey: ["timers"], queryFn: options.fetchTimers },
    createTimerMutationOptions: { mutationFn: options.create ?? notStubbed("create"), onSuccess: invalidate },
    updateTimerMutationOptions: { mutationFn: options.update ?? notStubbed("update"), onSuccess: invalidate },
    deleteTimerMutationOptions: { mutationFn: options.remove ?? notStubbed("remove"), onSuccess: invalidate },
  }))
}

function renderRoute(TimersRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <TimersRoute />
    </QueryClientProvider>,
  )
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

test("filling New timer and pressing Start timer creates a 600 second timer", async () => {
  const queryClient = newClient()
  let created: unknown
  stubTimers(queryClient, {
    fetchTimers: async () => [],
    create: async (input) => {
      created = input
      return sampleTimer()
    },
  })
  const { TimersRoute } = await import("./TimersRoute")

  renderRoute(TimersRoute, queryClient)

  await screen.findByText("Timers and alarms")
  await screen.findByText("No timers running.")
  fireEvent.change(screen.getByLabelText("Timer label"), { target: { value: "pasta" } })
  fireEvent.change(screen.getByLabelText("Minutes"), { target: { value: "10" } })
  fireEvent.click(screen.getByRole("button", { name: "Start timer" }))

  await waitFor(() => expect(created).toEqual({ kind: "timer", label: "pasta", duration_seconds: 600 }))
})

test("Pause and Delete call update and delete with the timer's id", async () => {
  const queryClient = newClient()
  let updated: unknown
  let removed: unknown
  stubTimers(queryClient, {
    fetchTimers: async () => [sampleTimer()],
    update: async (input) => {
      updated = input
      return sampleTimer({ paused: true })
    },
    remove: async (input) => {
      removed = input
    },
  })
  const { TimersRoute } = await import("./TimersRoute")

  renderRoute(TimersRoute, queryClient)

  fireEvent.click(await screen.findByRole("button", { name: "Pause" }))
  await waitFor(() => expect(updated).toEqual({ id: 1, changes: { paused: true } }))

  fireEvent.click(screen.getByRole("button", { name: "Delete" }))
  await waitFor(() => expect(removed).toEqual({ id: 1 }))
})

test("an alarm row shows Weekdays, and Turn off calls update with enabled false", async () => {
  const queryClient = newClient()
  let updated: unknown
  stubTimers(queryClient, {
    fetchTimers: async () => [sampleAlarm()],
    update: async (input) => {
      updated = input
      return sampleAlarm({ enabled: false })
    },
  })
  const { TimersRoute } = await import("./TimersRoute")

  renderRoute(TimersRoute, queryClient)

  expect(await screen.findByText(/Weekdays/)).toBeTruthy()
  expect(screen.getByText(/7:00 AM/)).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Turn off" }))
  await waitFor(() => expect(updated).toEqual({ id: 2, changes: { enabled: false } }))
})

test("a failed update shows the error on its own row", async () => {
  const queryClient = newClient()
  const { ApiError } = await import("@/lib/api")
  stubTimers(queryClient, {
    fetchTimers: async () => [sampleTimer()],
    update: async () => {
      throw new ApiError(400, "the time left must stay between 1 second and 86400 seconds")
    },
  })
  const { TimersRoute } = await import("./TimersRoute")

  renderRoute(TimersRoute, queryClient)

  fireEvent.click(await screen.findByRole("button", { name: "Pause" }))
  expect(await screen.findByText("the time left must stay between 1 second and 86400 seconds")).toBeTruthy()
})
