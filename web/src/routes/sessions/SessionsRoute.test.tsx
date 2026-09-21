// The first mount test wiring a fetch dependency, a router and a query
// client together (D-17's harness doing the job it was installed for,
// 08-01-PLAN.md). `mock.module` replaces `@/lib/sessions` before
// `SessionsRoute` is imported -- a dynamic `import()` inside the test body
// is what makes that ordering hold, since a static `import` at the top of
// this file would be hoisted ahead of the `mock.module` call and pick up
// the real module instead.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

const SAMPLE_SESSION = {
  id: "20260919T154201123456Z-turn-abc123",
  started_at: "2026-09-19T15:42:01.123456+00:00",
  turn_outcome: "completed",
  reply_text: "Turn the kitchen lights on",
  duration_ms: 812.5,
  has_audio: true,
}

mock.module("@/lib/sessions", () => ({
  SESSIONS_QUERY_KEY: ["sessions"],
  fetchSessions: async () => [SAMPLE_SESSION],
}))

afterEach(() => {
  cleanup()
})

test("mounting against a stubbed fetch renders one row carrying the session's summary text", async () => {
  const { SessionsRoute } = await import("./SessionsRoute")
  const queryClient = new QueryClient()

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <SessionsRoute />
      </MemoryRouter>
    </QueryClientProvider>,
  )

  expect(await screen.findByText("Turn the kitchen lights on")).toBeTruthy()
  // The household-audio disclosure (D-03) renders on this screen too.
  expect(screen.getByText("This shows real speech and audio recorded in your home.")).toBeTruthy()
})
