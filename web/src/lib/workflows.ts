// The pending-runs list and workflow editor's fetch surface (FLOW-09,
// FLOW-10). Every shape here matches `src/atlas/routes/workflows.py`'s
// own response models field for field, the same discipline `lib/macros.ts`
// states at its own top of file -- a renamed or reshaped field here is a
// silent drift from what the server actually sends.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"
import type { ConflictAnnotation } from "./macros"

// Imported, not redeclared (05-06-PLAN.md's own acceptance criterion) --
// `routes/conflict.py` is the one place either route module's conflict
// check lives, and `ConflictAnnotation`'s four values are exactly the
// same four values on a workflow step.
export type { ConflictAnnotation }

export type WorkflowStepKind = "wait" | "call_service" | "speak"

/** `WorkflowStepResponse`'s exact shape. `speak_cached` is `false` for
 * every non-`speak` step -- there is nothing to report about a step with
 * no words to synthesize (PA-01). */
export interface WorkflowStep {
  id: number
  position: number
  kind: WorkflowStepKind
  arguments: Record<string, unknown>
  due_at: string
  status: string
  attempts: number
  result_detail: Record<string, unknown> | null
  fired_at: string | null
  conflict: ConflictAnnotation
  speak_cached: boolean
}

/** `WorkflowRunResponse`'s exact shape. `late` is a still-pending step
 * whose `due_at` has passed (D-04's "and says so," made visible).
 * `reply_synthesis_degraded`/`reply_synthesis_message` are only ever
 * meaningful on a create/replace response -- `false`/`null` on every
 * list/read/cancel response, mirroring `Macro`'s own shape. */
export interface WorkflowRun {
  id: number
  origin: string
  status: string
  summary: string
  created_at: string
  updated_at: string
  created_by_user_id: number | null
  step_count: number
  late: boolean
  steps: WorkflowStep[]
  reply_synthesis_degraded: boolean
  reply_synthesis_message: string | null
}

export const WORKFLOWS_QUERY_KEY = ["workflows"] as const

export function workflowQueryKey(runId: number) {
  return ["workflows", runId] as const
}

export function fetchWorkflows(): Promise<WorkflowRun[]> {
  return apiFetch<WorkflowRun[]>("/api/workflows")
}

export function fetchWorkflow(runId: number): Promise<WorkflowRun> {
  return apiFetch<WorkflowRun>(`/api/workflows/${runId}`)
}

/** `WorkflowStepInput`'s exact shape -- what a create/replace request
 * sends per step. */
export interface WorkflowStepInput {
  kind: WorkflowStepKind
  arguments: Record<string, unknown>
}

/** `CreateWorkflowRequest`'s exact shape. `run_at` is a `datetime-local`
 * string (PA-03) -- this module performs no zone handling of its own;
 * the server resolves it against its own configured zone. */
export interface CreateWorkflowInput {
  summary: string
  run_at: string
  steps: WorkflowStepInput[]
}

/**
 * No optimistic update, the same reasoning `lib/macros.ts` records for
 * its own write mutations: the server performs a real side effect that
 * can fail (the schedule resolution, the speak-step precache) -- a run
 * shown before the server confirmed it would be a run the operator
 * believes is scheduled when it might not be. `onSuccess` seeds both the
 * list and the single-run cache directly from the response, since the
 * response already carries everything a GET would return.
 */
export const createWorkflowMutationOptions: UseMutationOptions<WorkflowRun, unknown, CreateWorkflowInput> = {
  mutationFn: (input) => apiFetch<WorkflowRun>("/api/workflows", { method: "POST", body: input }),
  onSuccess: (run) => {
    void queryClient.invalidateQueries({ queryKey: WORKFLOWS_QUERY_KEY })
    queryClient.setQueryData(workflowQueryKey(run.id), run)
  },
}

/** `ReplaceWorkflowStepsRequest`'s exact shape -- `steps` only. No
 * `summary`, no `run_at`: 05-04's own key-decision records that
 * `replace_steps` recomputes every step's `due_at` from the run's own
 * existing first step, with no parameter for a new schedule start --
 * moving the run's own anchor moment is not in this route's contract. */
export interface ReplaceWorkflowStepsInput {
  runId: number
  steps: WorkflowStepInput[]
}

export const replaceWorkflowStepsMutationOptions: UseMutationOptions<WorkflowRun, unknown, ReplaceWorkflowStepsInput> = {
  mutationFn: ({ runId, steps }) => apiFetch<WorkflowRun>(`/api/workflows/${runId}`, { method: "PUT", body: { steps } }),
  onSuccess: (run) => {
    void queryClient.invalidateQueries({ queryKey: WORKFLOWS_QUERY_KEY })
    queryClient.setQueryData(workflowQueryKey(run.id), run)
  },
}

export interface CancelWorkflowInput {
  runId: number
}

/** Cancel is a status transition, never a delete (D-12) -- the response
 * is the run's own new (terminal) snapshot, so `onSuccess` seeds the
 * single-run cache with it directly rather than evicting the entry the
 * way `deleteMacroMutationOptions` evicts a deleted macro's. */
export const cancelWorkflowMutationOptions: UseMutationOptions<WorkflowRun, unknown, CancelWorkflowInput> = {
  mutationFn: ({ runId }) => apiFetch<WorkflowRun>(`/api/workflows/${runId}/cancel`, { method: "POST" }),
  onSuccess: (run) => {
    void queryClient.invalidateQueries({ queryKey: WORKFLOWS_QUERY_KEY })
    queryClient.setQueryData(workflowQueryKey(run.id), run)
  },
}
