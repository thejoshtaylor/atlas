// Mount tests for the ready, removed and not-found branches (D-04,
// DBG-03). `mock.module` replaces `@/lib/sessions` before
// `SessionDetailRoute` is imported -- the same dynamic-`import()` ordering
// `SessionsRoute.test.tsx` already establishes, required because a static
// import at the top of this file would be hoisted ahead of `mock.module`.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { ApiError } from "@/lib/api"

const SAMPLE_SESSION = {
  id: "20260919T154201123456Z-turn-abc123",
  started_at: "2026-09-19T15:42:01.123456+00:00",
  turn_outcome: "completed",
  transcript: "turn the kitchen lights on",
  reply_text: "Turning the kitchen lights on.",
  stage_durations_ms: { turn_started_at: null, stt_final_at: 120.0 },
  end_of_speech_to_first_audio_ms: 400.0,
  end_of_speech_to_answer_audio_ms: 600.0,
  audio_format: { encoding: "pcm", sample_rate: 8000 },
  has_audio: true,
  timeline: [{ ts: 1.0, kind: "stage", offset_s: 0.0, stage: "turn_started_at" }],
}

afterEach(() => {
  cleanup()
})

// biome-ignore-start -- test-only indirection, see the module comment above
// eslint-disable-next-line @typescript-eslint/no-explicit-any
function renderAt(SessionDetailRoute: any, path: string) {
  // `retry: false` -- an errored fetch (the removed/not-found tests below)
  // must reach `status: "error"` inside `findByText`'s own wait window,
  // not disappear behind React Query's default three-retry backoff.
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[path]}>
        <Routes>
          <Route path="/sessions/:id" element={<SessionDetailRoute />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}
// biome-ignore-end

test("a real session renders the transcript, the reply and the timeline", async () => {
  // The full named-export set, not just what this test reads -- see
  // `SessionsRoute.test.tsx`'s identical comment: `mock.module` replaces
  // the registry entry for every consumer for the rest of this process.
  mock.module("@/lib/sessions", () => ({
    SESSIONS_QUERY_KEY: ["sessions"],
    fetchSessions: async () => [],
    sessionQueryKey: (id: string) => ["sessions", id],
    fetchSession: async () => SAMPLE_SESSION,
  }))
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)

  expect(await screen.findByText(/Heard:/)).toBeTruthy()
  expect(screen.getByText(/turn the kitchen lights on/)).toBeTruthy()
  expect(screen.getByText(/Turning the kitchen lights on\./)).toBeTruthy()
})

test("a session removed by retention renders the removed copy with no Retry button", async () => {
  mock.module("@/lib/sessions", () => ({
    SESSIONS_QUERY_KEY: ["sessions"],
    fetchSessions: async () => [],
    sessionQueryKey: (id: string) => ["sessions", id],
    fetchSession: async () => {
      throw new ApiError(404, "session '20260101T000000000000Z-x' has been removed by the retention sweep")
    },
  }))
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, "/sessions/20260101T000000000000Z-x")

  expect(await screen.findByText("This session was removed.")).toBeTruthy()
  expect(screen.queryByRole("button")).toBeNull()
})

test("a stale or made-up id renders the not-found copy with no Retry button", async () => {
  mock.module("@/lib/sessions", () => ({
    SESSIONS_QUERY_KEY: ["sessions"],
    fetchSessions: async () => [],
    sessionQueryKey: (id: string) => ["sessions", id],
    fetchSession: async () => {
      throw new ApiError(404, "no session with id 'made-up'")
    },
  }))
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, "/sessions/made-up")

  expect(await screen.findByText("Couldn't find this session.")).toBeTruthy()
  expect(screen.queryByRole("button")).toBeNull()
})
