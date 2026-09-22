// T-260922-eca. Mocks `global.fetch` directly, the same primitive
// `CalibrationRoute.test.ts`'s own re-entrancy test already mocks in this
// directory -- not `mock.module`, which replaces a module in Bun's
// process-wide registry for the rest of the run and would otherwise leak
// into `CalibrationRoute.test.ts`'s sibling test, which imports
// `runCalibrationMutationOptions` from `@/lib/calibration` statically.
import { afterEach, expect, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { CalibrationRoute } from "./CalibrationRoute"

const originalFetch = global.fetch

afterEach(() => {
  cleanup()
  global.fetch = originalFetch
})

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

function renderRoute(queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <CalibrationRoute />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

test("an echo-cancelled stored result explains why, with no delay or gain shown", async () => {
  global.fetch = (async () =>
    jsonResponse(200, {
      delay_s: 0.0,
      gain: 0.0,
      confidence: 0.02,
      agc_verdict: "absent",
      source: "camera",
      placement_note: "kitchen counter",
      taken_at: "2026-09-22T12:00:00+00:00",
      age_days: 0.01,
      echo_cancelled: true,
    })) as typeof fetch

  renderRoute(new QueryClient({ defaultOptions: { queries: { retry: false } } }))

  await screen.findByText(/No echo came back\./)
  expect(
    screen.getByText(
      "No echo came back. The camera cancels its own speaker from its microphone, so the assistant will not hear itself. If you did not hear the test sound, check the speaker and run the test again.",
    ),
  ).toBeTruthy()
  expect(screen.queryByText("Round-trip delay")).toBeNull()
  expect(screen.queryByText("Arrival level")).toBeNull()
})

test("an ordinary stored result still shows the measured delay and gain", async () => {
  global.fetch = (async () =>
    jsonResponse(200, {
      delay_s: 0.045,
      gain: 0.6,
      confidence: 0.91,
      agc_verdict: "absent",
      source: "camera",
      placement_note: "kitchen counter",
      taken_at: "2026-09-22T12:00:00+00:00",
      age_days: 0.01,
      echo_cancelled: false,
    })) as typeof fetch

  renderRoute(new QueryClient({ defaultOptions: { queries: { retry: false } } }))

  await screen.findByText("Round-trip delay")
  expect(screen.getByText("45 ms")).toBeTruthy()
  expect(screen.getByText("0.60")).toBeTruthy()
  expect(screen.queryByText(/No echo came back\./)).toBeNull()
})
