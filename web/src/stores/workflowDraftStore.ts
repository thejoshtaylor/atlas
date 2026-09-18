import { create } from "zustand"
import type { WorkflowRun, WorkflowStepInput, WorkflowStepKind } from "@/lib/workflows"

// The workflow editor's unsaved draft -- a summary, the schedule field's
// value, and an ordered DraftStep[] (05-06-PLAN.md's own key link: "the
// same reason those files exist for policy and macros" --
// `macroDraftStore.ts` is the file this one is built against for the
// ordered-list half). The new part, per that plan, is that a step is one
// of three shapes rather than one: a `DraftStep` carries every kind's
// fields at once (never a discriminated union of three separate shapes)
// so `updateStep`/`setStepKind` can operate on any row by key alone,
// exactly like `macroDraftStore.ts`'s own `updateAction`.
//
// 05-UI-SPEC.md is binding on the shape of an ALREADY-ADDED step row:
// two lines, a truncating summary plus Remove, then a right-aligned
// Move pair -- never a live-editable form. The kind picker and the
// kind's own fields therefore belong to the "Add step" composer only, a
// single in-progress draft kept apart from the ordered `steps` list
// until "Add step" is pressed -- the moment `steps` gains an entry, that
// entry is frozen to summary + Remove + Move, matching the macro
// editor's own "remove + re-add is the only edit path" precedent
// (04-07-SUMMARY.md's key-decision). The composer still uses the exact
// same `updateStep`/`setStepKind` functions the plan names, dispatched
// by key, so there is one code path for "a fact about a draft step's
// fields," not two.
//
// The loaded run itself is server state and lives in the query cache,
// never here -- `loadRun`/`loadBlank` are the only two ways this store's
// fields are ever set from outside an operator's own typing, matching
// `macroDraftStore.ts`'s own rule.

export type DurationUnit = "seconds" | "minutes" | "hours"

export interface DraftStep {
  /** A stable client-side key for React list rendering and for
   * targeting remove/move-up/move-down -- never sent to the server. An
   * existing step's own `id` (stringified) once loaded from a saved run,
   * or a locally-minted key for one added in this session. */
  key: string
  kind: WorkflowStepKind
  // wait
  durationValue: string
  durationUnit: DurationUnit
  // call_service
  domain: string
  service: string
  entityId: string
  transition: string
  // speak
  text: string
}

/** The one client-side key the "Add step" composer uses -- never sent
 * to the server, and never collides with a real step's own persisted
 * `id` (stringified) because no database id is ever this literal
 * string. */
export const COMPOSER_KEY = "new-step"

function blankStep(key: string, kind: WorkflowStepKind = "wait"): DraftStep {
  return {
    key,
    kind,
    durationValue: "",
    durationUnit: "seconds",
    domain: "",
    service: "",
    entityId: "",
    transition: "",
    text: "",
  }
}

interface WorkflowDraftState {
  summary: string
  runAt: string
  steps: DraftStep[]
  composer: DraftStep
  /** Set by any edit, cleared by `loadRun`/`loadBlank` -- reordering,
   * adding or removing a step marks the draft dirty without contacting
   * the server (matches `macroDraftStore.ts`'s own rule). */
  dirty: boolean

  loadRun: (run: WorkflowRun) => void
  loadBlank: () => void

  setSummary: (summary: string) => void
  setRunAt: (runAt: string) => void

  updateStep: (key: string, patch: Partial<Omit<DraftStep, "key" | "kind">>) => void
  setStepKind: (key: string, kind: WorkflowStepKind) => void

  addStep: () => void
  removeStep: (key: string) => void
  moveStepUp: (key: string) => void
  moveStepDown: (key: string) => void
}

let nextDraftKey = 0
function freshKey(): string {
  nextDraftKey += 1
  return `draft-step-${nextDraftKey}`
}

function stringField(args: Record<string, unknown>, field: string): string {
  const value = args[field]
  return typeof value === "string" ? value : ""
}

function numberField(args: Record<string, unknown>, field: string): number | null {
  const value = args[field]
  return typeof value === "number" && Number.isFinite(value) ? value : null
}

/** A saved `WorkflowStep` -> a `DraftStep`, tolerating `arguments`
 * missing any expected field -- a step of an unexpected shape loads
 * blank rather than crashing, `draftActionFrom`'s own precedent in
 * `macroDraftStore.ts`. */
export function draftStepFrom(step: { id: number; kind: WorkflowStepKind; arguments: Record<string, unknown> }): DraftStep {
  const args = step.arguments ?? {}
  const durationSeconds = numberField(args, "duration_s")
  const transition = numberField(args, "transition")
  return {
    key: String(step.id),
    kind: step.kind,
    durationValue: durationSeconds !== null ? String(durationSeconds) : "",
    durationUnit: "seconds",
    domain: stringField(args, "domain"),
    service: stringField(args, "service"),
    entityId: stringField(args, "entity_id"),
    transition: transition !== null ? String(transition) : "",
    text: stringField(args, "text"),
  }
}

export const useWorkflowDraftStore = create<WorkflowDraftState>((set) => ({
  summary: "",
  runAt: "",
  steps: [],
  composer: blankStep(COMPOSER_KEY),
  dirty: false,

  loadRun: (run) =>
    set({
      summary: run.summary,
      // `run_at` is never sent back by the server (only each step's own
      // `due_at` is) and `replace_steps` (05-04) accepts no new anchor
      // moment for an existing run -- the editor shows the existing
      // schedule read-only from the run's own first step, not through
      // this field, which stays blank for an existing run (PA-03).
      runAt: "",
      steps: run.steps.map(draftStepFrom),
      composer: blankStep(COMPOSER_KEY),
      dirty: false,
    }),

  loadBlank: () => set({ summary: "", runAt: "", steps: [], composer: blankStep(COMPOSER_KEY), dirty: false }),

  setSummary: (summary) => set({ summary, dirty: true }),
  setRunAt: (runAt) => set({ runAt, dirty: true }),

  updateStep: (key, patch) =>
    set((state) => {
      if (key === COMPOSER_KEY) return { composer: { ...state.composer, ...patch }, dirty: true }
      return { steps: state.steps.map((step) => (step.key === key ? { ...step, ...patch } : step)), dirty: true }
    }),

  // Switching a row's kind clears the fields belonging to the previous
  // kind (05-06-PLAN.md's own instruction) -- a full reset to that
  // kind's blank shape except `key`, so e.g. an entity id left behind
  // from a call_service row is never silently submitted under a speak
  // row.
  setStepKind: (key, kind) =>
    set((state) => {
      if (key === COMPOSER_KEY) return { composer: blankStep(COMPOSER_KEY, kind), dirty: true }
      return {
        steps: state.steps.map((step) => (step.key === key ? blankStep(step.key, kind) : step)),
        dirty: true,
      }
    }),

  // Appends the composer's current values as a new, frozen row -- "moves
  // a row between kinds without losing its position" (the plan's own
  // wording) describes the composer itself, which keeps its own kind
  // across edits until this call gives it a permanent position at the
  // end of the ordered list.
  addStep: () =>
    set((state) => ({
      steps: [...state.steps, { ...state.composer, key: freshKey() }],
      composer: blankStep(COMPOSER_KEY, state.composer.kind),
      dirty: true,
    })),

  removeStep: (key) => set((state) => ({ steps: state.steps.filter((step) => step.key !== key), dirty: true })),

  // Moving the first item up, or the last item down, is a no-op --
  // matches `macroDraftStore.ts`'s own `moveActionUp`/`moveActionDown`.
  moveStepUp: (key) =>
    set((state) => {
      const index = state.steps.findIndex((step) => step.key === key)
      if (index <= 0) return state
      const steps = [...state.steps]
      const [item] = steps.splice(index, 1)
      steps.splice(index - 1, 0, item)
      return { steps, dirty: true }
    }),

  moveStepDown: (key) =>
    set((state) => {
      const index = state.steps.findIndex((step) => step.key === key)
      if (index === -1 || index >= state.steps.length - 1) return state
      const steps = [...state.steps]
      const [item] = steps.splice(index, 1)
      steps.splice(index + 1, 0, item)
      return { steps, dirty: true }
    }),
}))

/** A fact about the list, not about any one row -- used to disable the
 * up control at the top and the down control at the bottom, matching
 * `isFirstAction`/`isLastAction` (`macroDraftStore.ts`). Pure functions
 * rather than store methods so they stay trivially testable. */
export function isFirstStep(steps: DraftStep[], key: string): boolean {
  return steps.length > 0 && steps[0].key === key
}

export function isLastStep(steps: DraftStep[], key: string): boolean {
  return steps.length > 0 && steps[steps.length - 1].key === key
}

function unitLabel(unit: DurationUnit, amount: number): string {
  const singular: Record<DurationUnit, string> = { seconds: "second", minutes: "minute", hours: "hour" }
  return amount === 1 ? singular[unit] : unit
}

/** 05-UI-SPEC.md's Copywriting Contract row-summary formats, verbatim,
 * for all three kinds -- computed here, not in JSX, the same reason
 * every other list fact in this app lives in a store or derive module
 * (this project has no rendered-DOM test infrastructure). */
export function formatStepSummary(step: DraftStep): string {
  if (step.kind === "wait") {
    const amount = Number(step.durationValue) || 0
    return `Wait ${amount} ${unitLabel(step.durationUnit, amount)}`
  }
  if (step.kind === "call_service") {
    const base = `${step.domain}.${step.service} → ${step.entityId}`
    const transition = step.transition.trim()
    return transition === "" ? base : `${base}, transition ${transition}s`
  }
  return `Speak: "${step.text}"`
}

function durationSeconds(value: string, unit: DurationUnit): number {
  const amount = Number(value) || 0
  if (unit === "minutes") return amount * 60
  if (unit === "hours") return amount * 3600
  return amount
}

/** The draft's ordered step list, shaped for a create/replace request
 * body -- the one place a `DraftStep`'s kind-specific fields fold back
 * into the `kind`/`arguments` shape `routes/workflows.py`'s own
 * `WorkflowStepInput` expects. A blank `transition` is simply not sent,
 * matching the Copywriting Contract's own "left blank, it is simply not
 * sent" rule. */
export function draftStepsToInput(steps: DraftStep[]): WorkflowStepInput[] {
  return steps.map((step) => {
    if (step.kind === "wait") {
      return { kind: "wait" as const, arguments: { duration_s: durationSeconds(step.durationValue, step.durationUnit) } }
    }
    if (step.kind === "call_service") {
      const args: Record<string, unknown> = { domain: step.domain, service: step.service, entity_id: step.entityId }
      const transition = step.transition.trim()
      if (transition !== "") args.transition = Number(transition)
      return { kind: "call_service" as const, arguments: args }
    }
    return { kind: "speak" as const, arguments: { text: step.text } }
  })
}
