import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { WorkflowRun, WorkflowStep } from "@/lib/workflows"
import type { DraftStep } from "@/stores/workflowDraftStore"
import {
  deriveWorkflowEditorState,
  editorLocked,
  PAST_SCHEDULE_SAVE_BLOCKED_REASON,
  saveBlockedByBlankSummary,
  saveBlockedByPastSchedule,
  saveBlockedByZeroSteps,
  speakStepPrecacheState,
  stepConflictDisplay,
  stepIsValid,
  ZERO_STEPS_SAVE_BLOCKED_REASON,
  type QueryLike,
} from "./deriveWorkflowEditorState"

function draftStep(overrides: Partial<DraftStep> = {}): DraftStep {
  return {
    key: "1",
    kind: "wait",
    durationValue: "",
    durationUnit: "seconds",
    domain: "",
    service: "",
    entityId: "",
    transition: "",
    text: "",
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
    step_count: 0,
    late: false,
    steps: [],
    reply_synthesis_degraded: false,
    reply_synthesis_message: null,
    ...overrides,
  }
}

function step(overrides: Partial<WorkflowStep> = {}): WorkflowStep {
  return {
    id: 1,
    position: 0,
    kind: "speak",
    arguments: { text: "Good night." },
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

describe("deriveWorkflowEditorState", () => {
  test("query: null renders ready with run: null -- a new run never contacts the server", () => {
    expect(deriveWorkflowEditorState(null)).toEqual({ kind: "ready", run: null })
  })

  test("pending renders loading", () => {
    const query: QueryLike<WorkflowRun> = { status: "pending", data: undefined, error: undefined }
    expect(deriveWorkflowEditorState(query)).toEqual({ kind: "loading" })
  })

  test("error renders the exact 'may have already run, been cancelled, or no longer exists' copy by default", () => {
    const query: QueryLike<WorkflowRun> = { status: "error", data: undefined, error: new Error("boom") }
    expect(deriveWorkflowEditorState(query)).toEqual({
      kind: "error",
      message: "Couldn't load this run. It may have already run, been cancelled, or no longer exists.",
    })
  })

  test("a named server reason surfaces unmodified", () => {
    const query: QueryLike<WorkflowRun> = { status: "error", data: undefined, error: new ApiError(404, "no workflow run with id 9") }
    expect(deriveWorkflowEditorState(query)).toEqual({ kind: "error", message: "no workflow run with id 9" })
  })

  test("a successful load renders ready with the run", () => {
    const r = run()
    const query: QueryLike<WorkflowRun> = { status: "success", data: r, error: undefined }
    expect(deriveWorkflowEditorState(query)).toEqual({ kind: "ready", run: r })
  })
})

describe("saveBlockedByBlankSummary -- required so POST /api/workflows never 422s on a missing field", () => {
  test("a blank or whitespace-only summary blocks save", () => {
    expect(saveBlockedByBlankSummary("")).toBe(true)
    expect(saveBlockedByBlankSummary("   ")).toBe(true)
  })
  test("a non-blank summary does not block save", () => {
    expect(saveBlockedByBlankSummary("turn off the porch light")).toBe(false)
  })
})

describe("saveBlockedByZeroSteps -- inline, before submit", () => {
  test("empty list blocks save with the named reason", () => {
    expect(saveBlockedByZeroSteps([])).toBe(ZERO_STEPS_SAVE_BLOCKED_REASON)
  })
  test("a non-empty list does not block save", () => {
    expect(saveBlockedByZeroSteps([draftStep()])).toBeNull()
  })
})

describe("saveBlockedByPastSchedule -- inline, under the Run at field", () => {
  const now = new Date("2026-09-19T12:00:00Z")

  test("a time in the future does not block save", () => {
    expect(saveBlockedByPastSchedule("2026-09-19T13:00", now)).toBeNull()
  })

  test("a time at or before now blocks save with the named reason", () => {
    expect(saveBlockedByPastSchedule("2026-09-19T11:00", now)).toBe(PAST_SCHEDULE_SAVE_BLOCKED_REASON)
  })

  test("a blank value does not block save on this rule (a required-field check is separate)", () => {
    expect(saveBlockedByPastSchedule("", now)).toBeNull()
  })
})

describe("stepIsValid -- per-kind field validity, mirroring routes/workflows.py's own refusals", () => {
  test("a wait step needs a non-negative duration", () => {
    expect(stepIsValid(draftStep({ kind: "wait", durationValue: "10" }))).toBe(true)
    expect(stepIsValid(draftStep({ kind: "wait", durationValue: "" }))).toBe(false)
    expect(stepIsValid(draftStep({ kind: "wait", durationValue: "-1" }))).toBe(false)
  })

  test("a call_service step needs domain and service", () => {
    expect(stepIsValid(draftStep({ kind: "call_service", domain: "light", service: "turn_off", entityId: "light.bedroom" }))).toBe(true)
    expect(stepIsValid(draftStep({ kind: "call_service", domain: "", service: "turn_off" }))).toBe(false)
    expect(stepIsValid(draftStep({ kind: "call_service", domain: "light", service: "" }))).toBe(false)
  })

  test("a call_service step's transition is only valid for the light domain", () => {
    expect(stepIsValid(draftStep({ kind: "call_service", domain: "light", service: "turn_on", transition: "5" }))).toBe(true)
    expect(stepIsValid(draftStep({ kind: "call_service", domain: "switch", service: "turn_on", transition: "5" }))).toBe(false)
  })

  test("a speak step needs non-empty words", () => {
    expect(stepIsValid(draftStep({ kind: "speak", text: "Good night." }))).toBe(true)
    expect(stepIsValid(draftStep({ kind: "speak", text: "" }))).toBe(false)
  })
})

describe("editorLocked -- D-11's append-only-once-firing rule, made visible", () => {
  test("a new (null) run is never locked", () => {
    expect(editorLocked(null)).toBe(false)
  })
  test("a pending run is not locked", () => {
    expect(editorLocked(run({ status: "pending" }))).toBe(false)
  })
  test("a firing run is locked", () => {
    expect(editorLocked(run({ status: "firing" }))).toBe(true)
  })
  test("a terminal run is locked", () => {
    expect(editorLocked(run({ status: "cancelled" }))).toBe(true)
  })
})

describe("stepConflictDisplay -- the three-state mapping, unknown distinct from ok", () => {
  test("ok carries no text", () => {
    expect(stepConflictDisplay("ok")).toEqual({ kind: "ok" })
  })
  test("denied uses the phase's own weaker, honest wording", () => {
    expect(stepConflictDisplay("denied")).toEqual({ kind: "denied", text: "Denied for control · would be refused right now" })
  })
  test("not_found reuses the existing string verbatim", () => {
    expect(stepConflictDisplay("not_found")).toEqual({ kind: "not_found", text: "Not found in Home Assistant" })
  })
  test("unknown is a distinct state, never folded into ok", () => {
    expect(stepConflictDisplay("unknown")).toEqual({ kind: "unknown", text: "Couldn't check this against the safety policy." })
  })
})

describe("speakStepPrecacheState -- delegates to deriveReplyPrecacheState (PA-01), never re-derived", () => {
  test("a cached speak step reads cached", () => {
    expect(speakStepPrecacheState(step({ speak_cached: true }), null)).toBe("cached")
  })
  test("an uncached speak step reads absent", () => {
    expect(speakStepPrecacheState(step({ speak_cached: false }), null)).toBe("absent")
  })
  test("a synthesis failure message takes precedence over the step's own cached flag", () => {
    expect(speakStepPrecacheState(step({ speak_cached: true }), "couldn't prepare it")).toBe("failed")
  })
  test("no step at all (a non-speak row) reads absent", () => {
    expect(speakStepPrecacheState(null, null)).toBe("absent")
  })
})
