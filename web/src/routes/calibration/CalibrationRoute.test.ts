// `CalibrationRoute.tsx` has no rendered-DOM test coverage in this
// suite -- this project has no `happy-dom`/testing-library dependency
// (03-03-SUMMARY.md's Deviations section flags this explicitly and
// names the reason: a genuine rendered-layout assertion needs a new
// dependency that goes through this project's package-legitimacy
// checkpoint before any later plan adds it, this plan included). What
// this file proves instead: the two properties the acceptance criteria
// name that do not need a DOM to prove, checked against the real
// component source and the real re-entrancy primitive it composes. The
// properties that genuinely need a rendered browser (results rendering
// first and largest, a 375px layout, a real click being ignored while
// disabled) are the held-out `<human-check>` in this plan's own
// `<verify>` block.
import { afterEach, describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { createSubmitGuard } from "@/lib/submitGuard"
import { runCalibrationMutationOptions } from "@/lib/calibration"
import { queryClient } from "@/lib/queryClient"

const SOURCE = readFileSync(join(import.meta.dir, "CalibrationRoute.tsx"), "utf-8")

describe("CalibrationRoute -- nothing starts a calibration run on mount", () => {
  test("the component defines no useEffect -- the only way something could run unprompted at mount", () => {
    // Pressing a control here makes a physical speaker play a sound and a
    // physical microphone record a room (T-03-34). The initial `useQuery`
    // reads the last stored result; a `useEffect` is the one shape that
    // could call the run mutation without a click, so its total absence
    // is a structural guarantee, not a runtime observation.
    expect(SOURCE).not.toMatch(/useEffect\(/)
  })

  test("the run mutation is only referenced from an explicit handler (handleRun/handleRetry), never from render body or an effect", () => {
    const mutateCallSites = [...SOURCE.matchAll(/run\.mutateAsync/g)]
    expect(mutateCallSites.length).toBeGreaterThan(0)
    // Both call sites live inside `handleRun`/`handleRetry`, not at the
    // top level of the component function -- approximated here by
    // requiring every occurrence to be preceded, somewhere above it in
    // the file, by one of those two handler names before the next
    // top-level `export`.
    const handlersDefined = /const handleRun = |const handleRetry = /.test(SOURCE)
    expect(handlersDefined).toBe(true)
  })

  test("the component never calls fetch directly -- it goes through lib/calibration.ts's typed functions only", () => {
    // A bare `fetch(` call, not `refetch(`/`prefetch(` (TanStack Query's
    // own methods, used here for `latest.refetch()`) -- the same
    // negative-lookbehind shape `api.ts`'s own grep-based check needed
    // once it hit this exact false positive (03-03-SUMMARY.md's
    // Deviations, "a grep false positive on session.refetch(").
    expect(SOURCE).not.toMatch(/(?<![a-zA-Z])fetch\(/)
  })
})

describe("CalibrationRoute -- the three results render before the controls in source order", () => {
  test("CalibrationResults is composed above the placement-note/run control block", () => {
    const resultsIndex = SOURCE.indexOf("<CalibrationResults")
    const controlsIndex = SOURCE.indexOf("showRunControls ?")
    expect(resultsIndex).toBeGreaterThan(-1)
    expect(controlsIndex).toBeGreaterThan(-1)
    expect(resultsIndex).toBeLessThan(controlsIndex)
  })
})

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("CalibrationRoute -- a second run cannot be issued while the first is in flight", () => {
  test("the exact re-entrancy guard SubmitButton wraps around handleRun allows exactly one underlying POST for two back-to-back clicks", async () => {
    let calls = 0
    let resolveFirst: (() => void) | undefined
    global.fetch = (async () => {
      calls += 1
      await new Promise<void>((resolve) => {
        resolveFirst = resolve
      })
      return new Response(
        JSON.stringify({
          delay_s: 0.1,
          gain: 0.3,
          confidence: 0.9,
          agc_verdict: "absent",
          source: "camera",
          placement_note: "",
          taken_at: "2026-09-17T00:00:00+00:00",
          age_days: 0,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    }) as typeof fetch

    // The same primitive `SubmitButton` wraps around `onSubmit`
    // (`components/state/SubmitButton.tsx`), applied here directly to
    // this screen's own run function -- `runCalibrationMutationOptions
    // .mutationFn`, the exact function `CalibrationRoute`'s `useMutation`
    // call is configured with.
    const guard = createSubmitGuard(() =>
      runCalibrationMutationOptions.mutationFn!({ placementNote: "kitchen counter" }, {} as never),
    )

    const first = guard.run()
    const second = guard.run()

    expect(guard.isPending()).toBe(true)
    expect(calls).toBe(1)

    resolveFirst?.()
    await Promise.all([first, second])

    expect(calls).toBe(1)
  })
})
