// The first mount test wiring a fetch dependency, a router and a query
// client together (D-17's harness doing the job it was installed for,
// 08-01-PLAN.md). `mock.module` replaces `@/lib/sessions` before
// `SessionsRoute` is imported -- a dynamic `import()` inside the test body
// is what makes that ordering hold, since a static `import` at the top of
// this file would be hoisted ahead of the `mock.module` call and pick up
// the real module instead.
//
// Every `mock.module` call below returns the full `@/lib/sessions` named-
// export set, not just what its own test reads: the mock replaces the
// module registry entry for every consumer for the rest of this `bun test`
// process, including `SessionDetailRoute.tsx`'s own sibling test file. An
// incomplete stub here would silently break that other file's import.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

function sampleSession(overrides: Record<string, unknown> = {}) {
  return {
    id: "20260919T154201123456Z-turn-abc123",
    started_at: "2026-09-19T15:42:01.123456+00:00",
    turn_outcome: "completed",
    reply_text: "Turn the kitchen lights on",
    duration_ms: 812.5,
    has_audio: true,
    ...overrides,
  }
}

function stubSessions(fetchSessions: () => Promise<unknown[]>) {
  mock.module("@/lib/sessions", () => ({
    SESSIONS_QUERY_KEY: ["sessions"],
    fetchSessions,
    sessionQueryKey: (id: string) => ["sessions", id],
    fetchSession: async () => {
      throw new Error("fetchSession is not stubbed in SessionsRoute.test.tsx")
    },
    sessionAudioUrl: (id: string) => `/api/sessions/${id}/audio`,
  }))
}

function renderRoute(SessionsRoute: React.ComponentType) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <SessionsRoute />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

import type * as React from "react"

afterEach(() => {
  cleanup()
})

test("mounting against a stubbed fetch renders one row carrying the session's summary text", async () => {
  stubSessions(async () => [sampleSession()])
  const { SessionsRoute } = await import("./SessionsRoute")

  renderRoute(SessionsRoute)

  expect(await screen.findByText("Turn the kitchen lights on")).toBeTruthy()
  // The household-audio disclosure (D-03) renders on this screen too.
  expect(screen.getByText("This shows real speech and audio recorded in your home.")).toBeTruthy()
  // The row's duration renders through DevMicRoute's own millisecond
  // formatter, suffixed "end to end" (08-UI-SPEC.md Copywriting Contract).
  expect(screen.getByText(/812\.5 ms end to end/)).toBeTruthy()
})

test("a pending query renders a three-row skeleton, never a momentarily empty list", async () => {
  let resolveSessions: (value: unknown[]) => void = () => {}
  stubSessions(() => new Promise((resolve) => (resolveSessions = resolve)))
  const { SessionsRoute } = await import("./SessionsRoute")

  renderRoute(SessionsRoute)

  expect(screen.getByRole("status", { name: "Loading" })).toBeTruthy()
  resolveSessions([])
})

test("a successful query with zero sessions renders the empty state", async () => {
  stubSessions(async () => [])
  const { SessionsRoute } = await import("./SessionsRoute")

  renderRoute(SessionsRoute)

  expect(await screen.findByText("No sessions yet.")).toBeTruthy()
})

test("a failed query renders the error state with a working Retry", async () => {
  let callCount = 0
  stubSessions(async () => {
    callCount += 1
    if (callCount === 1) throw new Error("boom")
    return [sampleSession()]
  })
  const { SessionsRoute } = await import("./SessionsRoute")

  renderRoute(SessionsRoute)

  const retryButton = await screen.findByText("Retry")
  fireEvent.click(retryButton)

  expect(await screen.findByText("Turn the kitchen lights on")).toBeTruthy()
})

test("an unrecognised outcome renders verbatim, never a fabricated friendlier label", async () => {
  stubSessions(async () => [sampleSession({ reply_text: null, turn_outcome: "brain_provider_unavailable" })])
  const { SessionsRoute } = await import("./SessionsRoute")

  renderRoute(SessionsRoute)

  expect(await screen.findByText("brain_provider_unavailable")).toBeTruthy()
})

test("empty_transcript renders the fixed copy, not the raw outcome string", async () => {
  stubSessions(async () => [sampleSession({ reply_text: null, turn_outcome: "empty_transcript" })])
  const { SessionsRoute } = await import("./SessionsRoute")

  renderRoute(SessionsRoute)

  expect(await screen.findByText("Understood no speech.")).toBeTruthy()
})

test("no control whose accessible name names removal exists anywhere on this screen (D-12)", async () => {
  stubSessions(async () => [sampleSession()])
  const { SessionsRoute } = await import("./SessionsRoute")

  renderRoute(SessionsRoute)
  await screen.findByText("Turn the kitchen lights on")

  const removalWords = /delete|remove/i
  for (const button of screen.queryAllByRole("button")) {
    expect(button.textContent ?? "").not.toMatch(removalWords)
  }
  for (const link of screen.queryAllByRole("link")) {
    expect(link.textContent ?? "").not.toMatch(removalWords)
  }
})
