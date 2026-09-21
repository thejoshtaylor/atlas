// Mount tests for the ready, removed and not-found branches (D-04,
// DBG-03). `mock.module` replaces `@/lib/sessions` before
// `SessionDetailRoute` is imported -- the same dynamic-`import()` ordering
// `SessionsRoute.test.tsx` already establishes, required because a static
// import at the top of this file would be hoisted ahead of `mock.module`.
//
// Every `mock.module("@/lib/sessions", ...)` call below returns the full
// named-export set, including `sessionAudioUrl` -- not just what its own
// test reads. `SessionsRoute.test.tsx`'s identical comment applies here
// too: the mock replaces the module registry entry for every consumer for
// the rest of this `bun test` process, including that sibling file.
//
// What the audio/timeline tests below prove, and what they do not
// (08-RESEARCH.md's own probe of happy-dom's `<audio>` element, plan
// 08-05, Task 3): happy-dom's `<audio>` is a structural stub. `currentTime`
// is a real, settable property and a manually dispatched `timeupdate`
// event does reach a listener with that value already set -- so these
// tests genuinely prove that a `timeupdate` listener is attached, that it
// maps a time to the row `activeTimelineIndexAt` names for that time, and
// that clicking a row sets the element's `currentTime`. But `play()`
// never advances the clock on its own, `duration` stays `NaN`, and no real
// media is ever decoded -- so nothing here proves, or could prove, that a
// real browser's real audio decode produces a `timeupdate` cadence a
// person would perceive as in step. A passing test in this file is a
// claim about wiring, never a claim about playback smoothness; the
// `<human-check>` in 08-05-PLAN.md's Task 3 is the only thing that can
// close that second question.
//
// `SAMPLE_SESSION`'s offsets are now shaped like a camera recording (plan
// 08-12, DBG-03): zero is the start of the replayed pre-roll, and the wake
// hit (`turn_started_at`) is `1.5 s` in, not at zero. The phase's original
// fixture invented its own flat offsets starting at zero, which is
// precisely why these mount tests could pass while the product was out of
// step with the corrected server -- 08-VERIFICATION.md's "The DOM harness
// (D-17)" section named this as the harness's own gap. A flat fixture
// cannot tell a corrected route from the defect 08-11 fixed; this one can.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"
import { ApiError } from "@/lib/api"

// A camera turn: `turn_started_at` sits 1.5 s into the recording (the
// replayed pre-roll), and every later mark is measured from that same
// monotonic clock plus the pre-roll shift -- `ts` values are the raw
// recorded-at instants (1.0, 2.2, 5.0), `offset_s` is each `ts` minus
// `turn_started_at`'s own `ts` (1.0) plus `preroll_s` (1.5).
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
  preroll_s: 1.5,
  timeline: [
    { ts: 1.0, kind: "stage", offset_s: 1.5, stage: "turn_started_at" },
    { ts: 2.2, kind: "event", offset_s: 2.7, type: "transcript.partial" },
    { ts: 5.0, kind: "stage", offset_s: 5.5, stage: "stt_final_at" },
  ],
}

// The browser-microphone and WebRTC shape: no `PrerollReplayingSource`
// wrapper, so `preroll_s` is `0` and the timeline starts at `0.0` -- the
// same flat offsets `SAMPLE_SESSION` used before this plan. Proves the
// no-pre-roll path acquired neither a caption nor a shifted highlight.
const NO_PREROLL_SESSION = {
  ...SAMPLE_SESSION,
  id: "20260919T154201123456Z-turn-nopreroll",
  preroll_s: 0,
  timeline: [
    { ts: 1.0, kind: "stage", offset_s: 0.0, stage: "turn_started_at" },
    { ts: 2.5, kind: "event", offset_s: 1.5, type: "transcript.partial" },
    { ts: 5.0, kind: "stage", offset_s: 4.0, stage: "stt_final_at" },
  ],
}

function stubLib(fetchSession: () => Promise<unknown>) {
  mock.module("@/lib/sessions", () => ({
    SESSIONS_QUERY_KEY: ["sessions"],
    fetchSessions: async () => [],
    sessionQueryKey: (id: string) => ["sessions", id],
    fetchSession,
    sessionAudioUrl: (id: string) => `/api/sessions/${id}/audio`,
  }))
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

function findRowButton(label: string): HTMLElement {
  const labelNode = screen.getByText(label)
  const button = labelNode.closest("button")
  if (!button) throw new Error(`no button ancestor found for row labelled "${label}"`)
  return button as HTMLElement
}

test("a real session renders the transcript, the reply and the timeline", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)

  expect(await screen.findByText(/Heard:/)).toBeTruthy()
  expect(screen.getByText(/turn the kitchen lights on/)).toBeTruthy()
  expect(screen.getByText(/Turning the kitchen lights on\./)).toBeTruthy()
})

test("a session removed by retention renders the removed copy with no Retry button", async () => {
  stubLib(async () => {
    throw new ApiError(404, "session '20260101T000000000000Z-x' has been removed by the retention sweep")
  })
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, "/sessions/20260101T000000000000Z-x")

  expect(await screen.findByText("This session was removed.")).toBeTruthy()
  expect(screen.queryByRole("button")).toBeNull()
})

test("a stale or made-up id renders the not-found copy with no Retry button", async () => {
  stubLib(async () => {
    throw new ApiError(404, "no session with id 'made-up'")
  })
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, "/sessions/made-up")

  expect(await screen.findByText("Couldn't find this session.")).toBeTruthy()
  expect(screen.queryByRole("button")).toBeNull()
})

// --- Task 3 of 08-05: the clock drives the row, proven by moving it --------

test("renders one <audio controls> element sourced at this session's audio path", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  const audioElements = document.querySelectorAll("audio")
  expect(audioElements.length).toBe(1)
  expect(audioElements[0].getAttribute("controls")).not.toBeNull()
  expect(audioElements[0].getAttribute("src")).toBe(`/api/sessions/${SAMPLE_SESSION.id}/audio`)
})

test("before any time has been reported, no timeline row is marked current", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  for (const button of screen.getAllByRole("button")) {
    expect(button.getAttribute("aria-current")).toBeNull()
  }
})

test("a clock inside the pre-roll window leaves no row marked current", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  const audioElement = document.querySelector("audio") as HTMLAudioElement
  // 1.0 s is before turn_started_at's own offset_s (1.5) -- room audio
  // captured before the wake word, and no timeline entry exists in that
  // window. Assert the absence across every row, not the presence of one.
  audioElement.currentTime = 1.0
  fireEvent(audioElement, new Event("timeupdate"))

  for (const button of screen.getAllByRole("button")) {
    expect(button.getAttribute("aria-current")).toBeNull()
  }
})

test("setting currentTime past a row marks exactly that row current", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  const audioElement = document.querySelector("audio") as HTMLAudioElement
  audioElement.currentTime = 3.0
  fireEvent(audioElement, new Event("timeupdate"))

  // offset_s 2.7 -> "transcript.partial", the last row at or before 3.0.
  const activeRow = findRowButton("transcript.partial")
  expect(activeRow.getAttribute("aria-current")).toBe("true")

  const otherRows = screen.getAllByRole("button").filter((button) => button !== activeRow)
  for (const button of otherRows) {
    expect(button.getAttribute("aria-current")).toBeNull()
  }
})

test("moving the time forward moves the mark to a later row; moving it back into the pre-roll window clears it", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  const audioElement = document.querySelector("audio") as HTMLAudioElement

  audioElement.currentTime = 6.0
  fireEvent(audioElement, new Event("timeupdate"))
  expect(findRowButton("stt_final_at").getAttribute("aria-current")).toBe("true")

  audioElement.currentTime = 1.0
  fireEvent(audioElement, new Event("timeupdate"))
  expect(findRowButton("stt_final_at").getAttribute("aria-current")).toBeNull()
  for (const button of screen.getAllByRole("button")) {
    expect(button.getAttribute("aria-current")).toBeNull()
  }
})

test("clicking a timeline row sets the audio element's currentTime to that row's offset", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  const audioElement = document.querySelector("audio") as HTMLAudioElement
  fireEvent.click(findRowButton("stt_final_at"))

  expect(audioElement.currentTime).toBe(5.5)
})

// --- Task 1 of 08-12: the pre-roll disclosure sentence -----------------

test("a camera-shaped session renders the pre-roll sentence naming its length", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  expect(await screen.findByText(/1\.5 s before the wake word/)).toBeTruthy()
})

test("a session with no pre-roll renders no pre-roll sentence, and the clock marks the first row at zero", async () => {
  stubLib(async () => NO_PREROLL_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${NO_PREROLL_SESSION.id}`)
  await screen.findByText(/Heard:/)

  expect(screen.queryByText(/before the wake word/)).toBeNull()

  const audioElement = document.querySelector("audio") as HTMLAudioElement
  audioElement.currentTime = 0.0
  fireEvent(audioElement, new Event("timeupdate"))
  expect(findRowButton("turn_started_at").getAttribute("aria-current")).toBe("true")
})

test("an audio error degrades only the player region -- timeline, transcript and reply stay intact", async () => {
  stubLib(async () => SAMPLE_SESSION)
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  const audioElement = document.querySelector("audio") as HTMLAudioElement
  fireEvent(audioElement, new Event("error"))

  expect(
    await screen.findByText("Couldn't load the recording. The rest of this session is still shown below."),
  ).toBeTruthy()
  expect(document.querySelector("audio")).toBeNull()
  // The rest of the screen is untouched: transcript, reply, and every
  // timeline row still render, unhighlighted.
  expect(screen.getByText(/turn the kitchen lights on/)).toBeTruthy()
  expect(screen.getByText(/Turning the kitchen lights on\./)).toBeTruthy()
  expect(screen.getByText("turn_started_at")).toBeTruthy()
  expect(screen.getByText("transcript.partial")).toBeTruthy()
  expect(screen.getByText("stt_final_at")).toBeTruthy()
  for (const button of screen.getAllByRole("button")) {
    expect(button.getAttribute("aria-current")).toBeNull()
  }
})

// IN-02: `has_audio` was fetched, typed, and never read.
test("a session the server says has no audio says so, instead of a player that 404s into a load failure", async () => {
  stubLib(async () => ({ ...SAMPLE_SESSION, has_audio: false }))
  const { SessionDetailRoute } = await import("./SessionDetailRoute")

  renderAt(SessionDetailRoute, `/sessions/${SAMPLE_SESSION.id}`)
  await screen.findByText(/Heard:/)

  expect(screen.getByText("No audio was recorded for this turn.")).toBeTruthy()
  expect(document.querySelectorAll("audio").length).toBe(0)
  // The load-failure copy is reserved for a real failure.
  expect(screen.queryByText(/Couldn't load the recording/)).toBeNull()
  // The rest of the session is untouched by it.
  expect(screen.getByText("Timeline")).toBeTruthy()
})
