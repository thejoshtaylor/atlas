import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { WorkflowRun, WorkflowStep } from "@/lib/workflows"
import {
  cancelDialogCopy,
  conflictedStepCount,
  deriveWorkflowsScreenState,
  formatConflictSummary,
  formatRunCount,
  formatStepCount,
  lateCaption,
  runBadges,
  runConflictDisplay,
  type QueryLike,
} from "./deriveWorkflowsScreenState"

function pending(): QueryLike<WorkflowRun[]> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored(error: unknown): QueryLike<WorkflowRun[]> {
  return { status: "error", data: undefined, error }
}
function success(data: WorkflowRun[]): QueryLike<WorkflowRun[]> {
  return { status: "success", data, error: undefined }
}

function step(overrides: Partial<WorkflowStep> = {}): WorkflowStep {
  return {
    id: 1,
    position: 0,
    kind: "wait",
    arguments: {},
    due_at: "2026-09-19T21:00:00Z",
    status: "pending",
    attempts: 0,
    result_detail: null,
    fired_at: null,
    conflict: "ok",
    speak_cached: false,
    ...overrides,
  }
}

function run(overrides: Partial<WorkflowRun> = {}): WorkflowRun {
  return {
    id: 1,
    origin: "webapp",
    status: "pending",
    summary: "turn off the porch light",
    created_at: "2026-09-18T00:00:00Z",
    updated_at: "2026-09-18T00:00:00Z",
    created_by_user_id: 1,
    step_count: 1,
    late: false,
    steps: [step()],
    reply_synthesis_degraded: false,
    reply_synthesis_message: null,
    ...overrides,
  }
}

describe("deriveWorkflowsScreenState", () => {
  test("pending renders loading -- the list must show skeleton rows, never an empty flash", () => {
    expect(deriveWorkflowsScreenState(pending())).toEqual({ kind: "loading" })
  })

  test("a failed load carries the Copywriting Contract's exact fallback text when the server gives none more specific", () => {
    expect(deriveWorkflowsScreenState(errored(new Error("network down")))).toEqual({
      kind: "error",
      message: "Couldn't load pending runs. Try again.",
    })
  })

  test("a failed load with a named server reason surfaces it unmodified", () => {
    expect(deriveWorkflowsScreenState(errored(new ApiError(500, "database unreachable")))).toEqual({
      kind: "error",
      message: "database unreachable",
    })
  })

  test("a successful load renders ready with every run", () => {
    const runs = [run({ id: 1 }), run({ id: 2 })]
    expect(deriveWorkflowsScreenState(success(runs))).toEqual({ kind: "ready", runs })
  })
})

describe("formatRunCount / formatStepCount -- zero-one-many", () => {
  test("1 reads as singular for both", () => {
    expect(formatRunCount(1)).toBe("1 run")
    expect(formatStepCount(1)).toBe("1 step")
  })
  test("0 and >1 read as plural for both", () => {
    expect(formatRunCount(0)).toBe("0 runs")
    expect(formatRunCount(3)).toBe("3 runs")
    expect(formatStepCount(0)).toBe("0 steps")
    expect(formatStepCount(3)).toBe("3 steps")
  })
})

describe("conflictedStepCount -- denied and not_found count, unknown never does", () => {
  test("counts only denied and not_found among remaining (pending) steps", () => {
    const r = run({
      steps: [
        step({ id: 1, conflict: "ok" }),
        step({ id: 2, conflict: "denied" }),
        step({ id: 3, conflict: "not_found" }),
        step({ id: 4, conflict: "unknown" }),
      ],
    })
    expect(conflictedStepCount(r)).toBe(2)
  })

  test("a run with only unknown annotations produces a conflicted count of zero", () => {
    const r = run({ steps: [step({ conflict: "unknown" })] })
    expect(conflictedStepCount(r)).toBe(0)
  })

  test("a step that already fired is never counted, even if its recorded conflict was denied", () => {
    const r = run({ steps: [step({ conflict: "denied", status: "completed" })] })
    expect(conflictedStepCount(r)).toBe(0)
  })
})

describe("runConflictDisplay -- unknown is never silently rendered as clean", () => {
  test("a run with only unknown remaining steps reports 'could not check', not 'none'", () => {
    const r = run({ steps: [step({ conflict: "unknown" })] })
    expect(runConflictDisplay(r)).toEqual({ kind: "unknown", text: "Couldn't check this against the safety policy." })
  })

  test("a run with a denied step and no unknowns reports the conflict summary", () => {
    const r = run({ steps: [step({ conflict: "denied" })] })
    expect(runConflictDisplay(r)).toEqual({ kind: "denied", text: "1 step would be refused right now" })
  })

  test("a run with no conflicted or unknown steps reports none", () => {
    const r = run({ steps: [step({ conflict: "ok" })] })
    expect(runConflictDisplay(r)).toEqual({ kind: "none" })
  })
})

describe("formatConflictSummary -- zero-one-many, verbatim Copywriting Contract text ending 'right now'", () => {
  test("1 reads as singular", () => {
    expect(formatConflictSummary(1)).toBe("1 step would be refused right now")
  })
  test("many reads as plural", () => {
    expect(formatConflictSummary(3)).toBe("3 steps would be refused right now")
  })
})

describe("runBadges -- origin always shown, status badges conditional", () => {
  test("a pending, on-time, spoken run shows only its origin badge", () => {
    expect(runBadges(run({ origin: "spoken", status: "pending", late: false }))).toEqual([{ key: "origin", label: "Spoken" }])
  })

  test("a firing webapp run shows origin and Running now", () => {
    expect(runBadges(run({ origin: "webapp", status: "firing" }))).toEqual([
      { key: "origin", label: "Webapp" },
      { key: "firing", label: "Running now" },
    ])
  })

  test("a late run shows origin and Late", () => {
    expect(runBadges(run({ late: true }))).toEqual([
      { key: "origin", label: "Webapp" },
      { key: "late", label: "Late" },
    ])
  })
})

describe("lateCaption -- 'Was due {relative time} ago.', D-04's lateness made visible", () => {
  test("null when the run is not late", () => {
    expect(lateCaption(run({ late: false }), new Date("2026-09-19T22:00:00Z"))).toBeNull()
  })

  test("minutes ago for a run overdue by less than an hour", () => {
    const r = run({ late: true, steps: [step({ due_at: "2026-09-19T21:45:00Z" })] })
    expect(lateCaption(r, new Date("2026-09-19T22:00:00Z"))).toBe("Was due 15 minutes ago.")
  })

  test("hours ago for a run overdue by more than an hour", () => {
    const r = run({ late: true, steps: [step({ due_at: "2026-09-19T20:00:00Z" })] })
    expect(lateCaption(r, new Date("2026-09-19T22:00:00Z"))).toBe("Was due 2 hours ago.")
  })
})

describe("cancelDialogCopy -- the one copy fork in this document", () => {
  test("a not-yet-firing run says the run will not happen", () => {
    expect(cancelDialogCopy(run({ status: "pending", summary: "turn off the porch light" }))).toEqual({
      body: 'Cancel "turn off the porch light"? This run will not happen, and this cannot be undone.',
      confirmLabel: "Cancel run",
    })
  })

  test("a firing run says an in-flight step will still finish, but no further step will run", () => {
    expect(cancelDialogCopy(run({ status: "firing", summary: "turn off the porch light" }))).toEqual({
      body: 'Cancel "turn off the porch light"? Any step already in progress will still finish, but no further step will run. This cannot be undone.',
      confirmLabel: "Cancel run",
    })
  })
})
