import { afterEach, describe, expect, test } from "bun:test"
import { queryClient } from "./queryClient"
import {
  WORKFLOWS_QUERY_KEY,
  cancelWorkflowMutationOptions,
  createWorkflowMutationOptions,
  fetchWorkflow,
  fetchWorkflows,
  replaceWorkflowStepsMutationOptions,
  workflowQueryKey,
  type WorkflowRun,
} from "./workflows"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

function sampleRun(overrides: Partial<WorkflowRun> = {}): WorkflowRun {
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

describe("workflows.ts -- calls the real routes/workflows.py paths", () => {
  test("fetchWorkflows calls GET /api/workflows", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [sampleRun()])
    }) as typeof fetch
    await fetchWorkflows()
    expect(calledUrl).toBe("/api/workflows")
  })

  test("fetchWorkflow calls GET /api/workflows/{id}", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, sampleRun({ id: 5 }))
    }) as typeof fetch
    await fetchWorkflow(5)
    expect(calledUrl).toBe("/api/workflows/5")
  })

  test("createWorkflowMutationOptions POSTs the save shape to /api/workflows", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(201, sampleRun())
    }) as typeof fetch
    await createWorkflowMutationOptions.mutationFn!(
      { summary: "turn off the porch light", run_at: "2026-09-19T21:00", steps: [] },
      {} as never,
    )
    expect(calledUrl).toBe("/api/workflows")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({
      summary: "turn off the porch light",
      run_at: "2026-09-19T21:00",
      steps: [],
    })
  })

  test("replaceWorkflowStepsMutationOptions PUTs to /api/workflows/{id}, with runId stripped from the body", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, sampleRun({ id: 7 }))
    }) as typeof fetch
    await replaceWorkflowStepsMutationOptions.mutationFn!({ runId: 7, steps: [] }, {} as never)
    expect(calledUrl).toBe("/api/workflows/7")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({ steps: [] })
  })

  test("cancelWorkflowMutationOptions POSTs /api/workflows/{id}/cancel, never a DELETE (D-12)", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return jsonResponse(200, sampleRun({ id: 9, status: "cancelled" }))
    }) as typeof fetch
    await cancelWorkflowMutationOptions.mutationFn!({ runId: 9 }, {} as never)
    expect(calledUrl).toBe("/api/workflows/9/cancel")
    expect(calledMethod).toBe("POST")
  })
})

describe("mutation onSuccess -- seeds the single-run cache directly, no second fetch needed to see it", () => {
  test("create seeds workflowQueryKey(id) with the response run", () => {
    const created = sampleRun({ id: 42 })
    createWorkflowMutationOptions.onSuccess?.(
      created,
      { summary: "x", run_at: "2026-09-19T21:00", steps: [] },
      undefined,
      {} as never,
    )
    expect(queryClient.getQueryData(workflowQueryKey(42))).toEqual(created)
  })

  test("cancel seeds workflowQueryKey(id) with the run's own terminal snapshot -- never evicted (D-12)", () => {
    const cancelled = sampleRun({ id: 9, status: "cancelled" })
    cancelWorkflowMutationOptions.onSuccess?.(cancelled, { runId: 9 }, undefined, {} as never)
    expect(queryClient.getQueryData(workflowQueryKey(9))).toEqual(cancelled)
  })
})

describe("workflowQueryKey / WORKFLOWS_QUERY_KEY -- distinct, stable keys", () => {
  test("the list key and a single-run key never collide", () => {
    expect(workflowQueryKey(1)).not.toEqual(WORKFLOWS_QUERY_KEY)
    expect(workflowQueryKey(1)).toEqual(["workflows", 1])
  })
})
