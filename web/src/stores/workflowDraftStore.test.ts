import { beforeEach, describe, expect, test } from "bun:test"
import type { WorkflowRun } from "@/lib/workflows"
import {
  COMPOSER_KEY,
  draftStepFrom,
  draftStepsToInput,
  formatStepSummary,
  isFirstStep,
  isLastStep,
  useWorkflowDraftStore,
  type DraftStep,
} from "./workflowDraftStore"

function sampleRun(overrides: Partial<WorkflowRun> = {}): WorkflowRun {
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
    steps: [
      { id: 10, position: 0, kind: "wait", arguments: { duration_s: 60 }, due_at: "2026-09-19T21:00:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
    ],
    reply_synthesis_degraded: false,
    reply_synthesis_message: null,
    ...overrides,
  }
}

beforeEach(() => {
  useWorkflowDraftStore.getState().loadBlank()
})

describe("loadRun / loadBlank -- server data seeds the draft, and only these two actions ever do", () => {
  test("loadRun copies summary and steps, and starts clean (not dirty)", () => {
    useWorkflowDraftStore.getState().loadRun(sampleRun())
    const state = useWorkflowDraftStore.getState()
    expect(state.summary).toBe("turn off the porch light")
    expect(state.steps).toEqual([
      { key: "10", kind: "wait", durationValue: "60", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "" },
    ])
    expect(state.dirty).toBe(false)
  })

  test("loadBlank clears everything for a new, unsaved run", () => {
    useWorkflowDraftStore.getState().loadRun(sampleRun())
    useWorkflowDraftStore.getState().loadBlank()
    const state = useWorkflowDraftStore.getState()
    expect(state.summary).toBe("")
    expect(state.steps).toEqual([])
    expect(state.dirty).toBe(false)
  })
})

describe("addStep -- appends the composer's current values as a new, frozen row", () => {
  test("addStep appends the composer and resets it to blank, dirtying the draft", () => {
    useWorkflowDraftStore.getState().setStepKind(COMPOSER_KEY, "speak")
    useWorkflowDraftStore.getState().updateStep(COMPOSER_KEY, { text: "Good night." })
    useWorkflowDraftStore.getState().addStep()
    const state = useWorkflowDraftStore.getState()
    expect(state.steps).toHaveLength(1)
    expect(state.steps[0]).toMatchObject({ kind: "speak", text: "Good night." })
    expect(state.composer).toMatchObject({ kind: "speak", text: "" })
    expect(state.dirty).toBe(true)
  })
})

describe("setStepKind -- switching a row's kind clears the fields belonging to the previous kind", () => {
  test("switching the composer from call_service to speak drops the call_service fields", () => {
    useWorkflowDraftStore.getState().setStepKind(COMPOSER_KEY, "call_service")
    useWorkflowDraftStore.getState().updateStep(COMPOSER_KEY, { domain: "light", service: "turn_off", entityId: "light.bedroom", transition: "5" })
    useWorkflowDraftStore.getState().setStepKind(COMPOSER_KEY, "speak")
    const composer = useWorkflowDraftStore.getState().composer
    expect(composer.kind).toBe("speak")
    expect(composer).toMatchObject({ domain: "", service: "", entityId: "", transition: "", text: "" })
  })

  test("switching an already-added row's kind clears its previous fields without changing its position", () => {
    useWorkflowDraftStore.getState().loadRun(
      sampleRun({
        steps: [
          { id: 1, position: 0, kind: "wait", arguments: { duration_s: 30 }, due_at: "2026-09-19T21:00:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
          { id: 2, position: 1, kind: "call_service", arguments: { domain: "light", service: "turn_off", entity_id: "light.bedroom" }, due_at: "2026-09-19T21:01:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
        ],
      }),
    )
    useWorkflowDraftStore.getState().setStepKind("2", "speak")
    const state = useWorkflowDraftStore.getState()
    expect(state.steps.map((s) => s.key)).toEqual(["1", "2"])
    expect(state.steps[1]).toMatchObject({ kind: "speak", domain: "", service: "", entityId: "" })
  })
})

describe("remove / reorder -- edits the draft only, dirties it, contacts no server", () => {
  test("removeStep drops exactly the targeted step by key", () => {
    useWorkflowDraftStore.getState().loadRun(sampleRun())
    useWorkflowDraftStore.getState().removeStep("10")
    expect(useWorkflowDraftStore.getState().steps).toEqual([])
  })

  test("moving the first step up is a no-op", () => {
    useWorkflowDraftStore.getState().loadRun(
      sampleRun({
        steps: [
          { id: 1, position: 0, kind: "wait", arguments: {}, due_at: "2026-09-19T21:00:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
          { id: 2, position: 1, kind: "wait", arguments: {}, due_at: "2026-09-19T21:01:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
        ],
      }),
    )
    const before = useWorkflowDraftStore.getState().steps
    useWorkflowDraftStore.getState().moveStepUp("1")
    const after = useWorkflowDraftStore.getState()
    expect(after.steps).toEqual(before)
    expect(after.dirty).toBe(false)
  })

  test("moving the last step down is a no-op", () => {
    useWorkflowDraftStore.getState().loadRun(
      sampleRun({
        steps: [
          { id: 1, position: 0, kind: "wait", arguments: {}, due_at: "2026-09-19T21:00:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
          { id: 2, position: 1, kind: "wait", arguments: {}, due_at: "2026-09-19T21:01:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
        ],
      }),
    )
    const before = useWorkflowDraftStore.getState().steps
    useWorkflowDraftStore.getState().moveStepDown("2")
    const after = useWorkflowDraftStore.getState()
    expect(after.steps).toEqual(before)
    expect(after.dirty).toBe(false)
  })

  test("moving a middle step up swaps it with its predecessor and dirties the draft", () => {
    useWorkflowDraftStore.getState().loadRun(
      sampleRun({
        steps: [
          { id: 1, position: 0, kind: "wait", arguments: {}, due_at: "2026-09-19T21:00:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
          { id: 2, position: 1, kind: "wait", arguments: {}, due_at: "2026-09-19T21:01:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
          { id: 3, position: 2, kind: "wait", arguments: {}, due_at: "2026-09-19T21:02:00Z", status: "pending", attempts: 0, result_detail: null, fired_at: null, conflict: "ok", speak_cached: false },
        ],
      }),
    )
    useWorkflowDraftStore.getState().moveStepUp("2")
    const state = useWorkflowDraftStore.getState()
    expect(state.steps.map((s) => s.key)).toEqual(["2", "1", "3"])
    expect(state.dirty).toBe(true)
  })
})

describe("isFirstStep / isLastStep -- the boundary facts the reorder controls disable on", () => {
  const steps: DraftStep[] = [
    { key: "a", kind: "wait", durationValue: "", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "" },
    { key: "b", kind: "wait", durationValue: "", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "" },
    { key: "c", kind: "wait", durationValue: "", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "" },
  ]

  test("the first row is first, and nothing else is", () => {
    expect(isFirstStep(steps, "a")).toBe(true)
    expect(isFirstStep(steps, "b")).toBe(false)
    expect(isFirstStep(steps, "c")).toBe(false)
  })

  test("the last row is last, and nothing else is", () => {
    expect(isLastStep(steps, "c")).toBe(true)
    expect(isLastStep(steps, "b")).toBe(false)
    expect(isLastStep(steps, "a")).toBe(false)
  })

  test("a single-step list is both first and last -- both controls stay disabled", () => {
    const single: DraftStep[] = [{ key: "only", kind: "wait", durationValue: "", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "" }]
    expect(isFirstStep(single, "only")).toBe(true)
    expect(isLastStep(single, "only")).toBe(true)
  })
})

describe("formatStepSummary -- 05-UI-SPEC.md's Copywriting Contract, verbatim, for all three kinds", () => {
  test("a wait step reads 'Wait {n} {unit}'", () => {
    expect(
      formatStepSummary({ key: "1", kind: "wait", durationValue: "10", durationUnit: "minutes", domain: "", service: "", entityId: "", transition: "", text: "" }),
    ).toBe("Wait 10 minutes")
  })

  test("a wait step of 1 unit reads the singular form", () => {
    expect(
      formatStepSummary({ key: "1", kind: "wait", durationValue: "1", durationUnit: "hours", domain: "", service: "", entityId: "", transition: "", text: "" }),
    ).toBe("Wait 1 hour")
  })

  test("a call_service step without a transition reads '{domain}.{service} → {entity_id}'", () => {
    expect(
      formatStepSummary({ key: "1", kind: "call_service", durationValue: "", durationUnit: "seconds", domain: "light", service: "turn_off", entityId: "light.bedroom", transition: "", text: "" }),
    ).toBe("light.turn_off → light.bedroom")
  })

  test("a call_service step with a transition appends ', transition {n}s'", () => {
    expect(
      formatStepSummary({ key: "1", kind: "call_service", durationValue: "", durationUnit: "seconds", domain: "light", service: "turn_on", entityId: "light.bedroom", transition: "5", text: "" }),
    ).toBe("light.turn_on → light.bedroom, transition 5s")
  })

  test("a speak step reads 'Speak: \"{text}\"'", () => {
    expect(
      formatStepSummary({ key: "1", kind: "speak", durationValue: "", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "Good night." }),
    ).toBe('Speak: "Good night."')
  })
})

describe("draftStepFrom -- loading a saved step tolerates arguments missing any of the expected fields", () => {
  test("a step with unrelated arguments loads with blank fields, not a crash", () => {
    const draft = draftStepFrom({ id: 5, kind: "speak", arguments: { unrelated: "value" } })
    expect(draft).toEqual({ key: "5", kind: "speak", durationValue: "", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "" })
  })
})

describe("draftStepsToInput -- the draft's kind-specific fields fold back into kind/arguments", () => {
  test("a wait step converts its duration to seconds regardless of the chosen unit", () => {
    const steps: DraftStep[] = [
      { key: "1", kind: "wait", durationValue: "2", durationUnit: "minutes", domain: "", service: "", entityId: "", transition: "", text: "" },
    ]
    expect(draftStepsToInput(steps)).toEqual([{ kind: "wait", arguments: { duration_s: 120 } }])
  })

  test("a call_service step omits transition entirely when left blank", () => {
    const steps: DraftStep[] = [
      { key: "1", kind: "call_service", durationValue: "", durationUnit: "seconds", domain: "light", service: "turn_off", entityId: "light.bedroom", transition: "", text: "" },
    ]
    expect(draftStepsToInput(steps)).toEqual([
      { kind: "call_service", arguments: { domain: "light", service: "turn_off", entity_id: "light.bedroom" } },
    ])
  })

  test("a call_service step includes transition as a number when set", () => {
    const steps: DraftStep[] = [
      { key: "1", kind: "call_service", durationValue: "", durationUnit: "seconds", domain: "light", service: "turn_on", entityId: "light.bedroom", transition: "5", text: "" },
    ]
    expect(draftStepsToInput(steps)).toEqual([
      { kind: "call_service", arguments: { domain: "light", service: "turn_on", entity_id: "light.bedroom", transition: 5 } },
    ])
  })

  test("a speak step round-trips into { text }", () => {
    const steps: DraftStep[] = [
      { key: "1", kind: "speak", durationValue: "", durationUnit: "seconds", domain: "", service: "", entityId: "", transition: "", text: "Good night." },
    ]
    expect(draftStepsToInput(steps)).toEqual([{ kind: "speak", arguments: { text: "Good night." } }])
  })
})
