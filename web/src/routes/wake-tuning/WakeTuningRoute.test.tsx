// D-17, the test this screen was adopted for: mounts the real component
// with a stubbed fetch layer, fires a real `change` event on the range
// input, and reads the counts back off the DOM. `mock.module` replaces
// `@/lib/wakeTuning` before `WakeTuningRoute` is imported -- the same
// dynamic-`import()` ordering `SessionDetailRoute.test.tsx` already
// establishes, required because a static import at the top of this file
// would be hoisted ahead of `mock.module`. Every `mock.module` call below
// returns the module's full named-export set (`WAKE_EVENTS_QUERY_KEY`,
// `fetchWakeEvents`, `setWakeThresholdMutationOptions`) -- an incomplete
// stub leaks across the rest of this `bun test` process, including
// sibling files (the same lesson 08-04's and 08-09's harness work
// already recorded).
//
// What this file proves, and what it does not: that dragging the control
// re-partitions the rendered counts with no second network request, that
// the two badges per row are independent facts (moving the control
// changes only one of them), and that the commit button calls the
// mutation once with the previewed value. It does not, and cannot,
// prove that a real drag *feels* smooth in a real browser, or that the
// engine named on screen matches what is actually configured on a real
// deployment -- that is 08-08-PLAN.md's own `<human-check>`, held out for
// end-of-phase UAT per `workflow.human_verify_mode=end-of-phase`.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"

interface StubEvent {
  id: number
  source: string
  engine: string
  score: number | null
  allowed: boolean
  block_reason: string | null
  recorded_at: string
}

function stubEvent(overrides: Partial<StubEvent> = {}): StubEvent {
  return {
    id: 1,
    source: "camera",
    engine: "openwakeword",
    score: 0.5,
    allowed: true,
    block_reason: null,
    recorded_at: "2026-09-21T00:00:00Z",
    ...overrides,
  }
}

interface StubResponse {
  events: StubEvent[]
  engine?: string
  engine_grades?: boolean
  threshold?: number
  not_scored_session_count?: number
  capped?: boolean
}

function stubLib(
  fetchWakeEvents: () => Promise<StubResponse>,
  mutationFn: (input: { threshold: number }) => Promise<{ threshold: number }> = async ({ threshold }) => ({
    threshold,
  }),
) {
  mock.module("@/lib/wakeTuning", () => ({
    WAKE_EVENTS_QUERY_KEY: ["wake-events"],
    fetchWakeEvents: async () => {
      const response = await fetchWakeEvents()
      return {
        events: response.events,
        engine: response.engine ?? "openwakeword",
        engine_grades: response.engine_grades ?? true,
        threshold: response.threshold ?? 0.5,
        not_scored_session_count: response.not_scored_session_count ?? 0,
        capped: response.capped ?? false,
      }
    },
    setWakeThresholdMutationOptions: { mutationFn },
  }))
}

afterEach(() => {
  cleanup()
})

// biome-ignore-start -- test-only indirection, see the module comment above
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderRoute(WakeTuningRoute: any) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <WakeTuningRoute />
    </QueryClientProvider>,
  )
}
// biome-ignore-end

function rangeInput(): HTMLInputElement {
  return screen.getByRole("slider") as HTMLInputElement
}

test("mounting with a set of events spanning several scores renders a count matching the initial threshold", async () => {
  let fetchCount = 0
  stubLib(async () => {
    fetchCount += 1
    return {
      events: [
        stubEvent({ id: 1, score: 0.9 }),
        stubEvent({ id: 2, score: 0.3 }),
        stubEvent({ id: 3, score: 0.6 }),
      ],
      threshold: 0.5,
    }
  })
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  expect(
    await screen.findByText("2 clear this threshold, 1 do not, out of 3 recorded openwakeword wake attempts."),
  ).toBeTruthy()
  expect(fetchCount).toBe(1)
})

test("moving the control to a higher value re-renders with fewer clearing and more below; moving it back moves the counts back -- no network request per drag", async () => {
  let fetchCount = 0
  stubLib(async () => {
    fetchCount += 1
    return {
      events: [
        stubEvent({ id: 1, score: 0.9 }),
        stubEvent({ id: 2, score: 0.3 }),
        stubEvent({ id: 3, score: 0.6 }),
      ],
      threshold: 0.5,
    }
  })
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  await screen.findByText("2 clear this threshold, 1 do not, out of 3 recorded openwakeword wake attempts.")

  fireEvent.change(rangeInput(), { target: { value: "0.95" } })
  expect(
    await screen.findByText("0 clear this threshold, 3 do not, out of 3 recorded openwakeword wake attempts."),
  ).toBeTruthy()

  fireEvent.change(rangeInput(), { target: { value: "0.2" } })
  expect(
    await screen.findByText("3 clear this threshold, 0 do not, out of 3 recorded openwakeword wake attempts."),
  ).toBeTruthy()

  fireEvent.change(rangeInput(), { target: { value: "0.5" } })
  expect(
    await screen.findByText("2 clear this threshold, 1 do not, out of 3 recorded openwakeword wake attempts."),
  ).toBeTruthy()

  // The stubbed fetch was called exactly once, for the initial load --
  // every count above was recomputed locally against the same held
  // event list.
  expect(fetchCount).toBe(1)
})

test("an event whose score clears the previewed threshold but was historically blocked renders both badges; moving the control changes only the score-versus-threshold one", async () => {
  stubLib(async () => ({
    events: [stubEvent({ id: 1, score: 0.9, allowed: false, block_reason: "refractory" })],
    threshold: 0.5,
  }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  await screen.findByText("Clears")
  expect(screen.getByText("Blocked — refractory window")).toBeTruthy()

  fireEvent.change(rangeInput(), { target: { value: "0.99" } })

  await screen.findByText("Below")
  expect(screen.queryByText("Clears")).toBeNull()
  // The historical badge is a fixed fact -- moving the control never
  // touches it (T-08-34).
  expect(screen.getByText("Blocked — refractory window")).toBeTruthy()
})

test("a response reporting an engine that does not grade renders the no-gradient sentence", async () => {
  stubLib(async () => ({
    events: [stubEvent({ engine: "vosk", score: 1.0 })],
    engine: "vosk",
    engine_grades: false,
    threshold: 0.5,
  }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  expect(
    await screen.findByText(
      "The configured wake engine (vosk) reports the same score for every wake it recognizes. Moving this control will not change what it detects.",
    ),
  ).toBeTruthy()
})

test("a response reporting an engine that grades does not render the no-gradient sentence", async () => {
  stubLib(async () => ({
    events: [stubEvent({ engine: "openwakeword" })],
    engine: "openwakeword",
    engine_grades: true,
    threshold: 0.5,
  }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  await screen.findByText(/recorded openwakeword wake attempts\./)
  expect(
    screen.queryByText(
      "The configured wake engine (openwakeword) reports the same score for every wake it recognizes. Moving this control will not change what it detects.",
    ),
  ).toBeNull()
})

test("pressing the commit control calls the mutation exactly once with the previewed value", async () => {
  const mutationCalls: number[] = []
  stubLib(
    async () => ({
      events: [stubEvent({ id: 1, score: 0.9 })],
      threshold: 0.5,
    }),
    async ({ threshold }) => {
      mutationCalls.push(threshold)
      return { threshold }
    },
  )
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  await screen.findByText(/recorded openwakeword wake attempts\./)
  fireEvent.change(rangeInput(), { target: { value: "0.7" } })
  await screen.findByText("1 clear this threshold, 0 do not, out of 1 recorded openwakeword wake attempts.")

  fireEvent.click(screen.getByRole("button", { name: "Set as active threshold" }))

  await screen.findByText("Threshold set to 0.70. This takes effect immediately — no restart needed.")
  expect(mutationCalls).toEqual([0.7])
})

test("this screen never renders the household-audio disclosure", async () => {
  stubLib(async () => ({
    events: [stubEvent()],
    threshold: 0.5,
  }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  await screen.findByText(/recorded openwakeword wake attempts\./)
  expect(screen.queryByText("This shows real speech and audio recorded in your home.")).toBeNull()
})

test("a capped response says so on screen, rather than presenting a partial history as a whole one (WR-06)", async () => {
  stubLib(async () => ({
    events: [stubEvent({ id: 1, score: 0.9 })],
    threshold: 0.5,
    capped: true,
  }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  expect(
    await screen.findByText("Older wake attempts than these exist and aren't included above."),
  ).toBeTruthy()
})

test("an uncapped response says nothing about a cap", async () => {
  stubLib(async () => ({ events: [stubEvent({ id: 1, score: 0.9 })], threshold: 0.5 }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  await screen.findByText("1 clear this threshold, 0 do not, out of 1 recorded openwakeword wake attempts.")
  expect(screen.queryByText("Older wake attempts than these exist and aren't included above.")).toBeNull()
})

test("an unrecognised block reason renders the gate's own string verbatim, never a threshold label (WR-08)", async () => {
  stubLib(async () => ({
    events: [stubEvent({ id: 1, score: 0.9, allowed: false, block_reason: "a_fourth_reason" })],
    threshold: 0.5,
  }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  expect(await screen.findByText("Blocked — a_fourth_reason")).toBeTruthy()
  expect(screen.queryByText("Blocked — below threshold")).toBeNull()
})

test("a blocked row with no reason recorded says so rather than naming one", async () => {
  stubLib(async () => ({
    events: [stubEvent({ id: 1, score: 0.9, allowed: false, block_reason: null })],
    threshold: 0.5,
  }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  expect(await screen.findByText("Blocked — no reason recorded")).toBeTruthy()
  expect(screen.queryByText("Blocked — below threshold")).toBeNull()
})

test("the empty state names the sessions that predate scoring, rather than reporting no history at all (WR-09)", async () => {
  stubLib(async () => ({ events: [], not_scored_session_count: 137 }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  expect(await screen.findByText("No wake attempts recorded yet.")).toBeTruthy()
  expect(
    screen.getByText("137 earlier sessions predate wake-score recording and aren't included above."),
  ).toBeTruthy()
})

test("a genuinely fresh install's empty state says nothing about earlier sessions", async () => {
  stubLib(async () => ({ events: [], not_scored_session_count: 0 }))
  const { WakeTuningRoute } = await import("./WakeTuningRoute")
  renderRoute(WakeTuningRoute)

  expect(await screen.findByText("No wake attempts recorded yet.")).toBeTruthy()
  expect(screen.queryByText(/predate wake-score recording/)).toBeNull()
})
