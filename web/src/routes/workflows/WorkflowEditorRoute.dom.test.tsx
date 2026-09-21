// D-17 backfill part B (08-10-PLAN.md), Task 2. `WorkflowEditorRoute`
// has never had a screen-level test of any kind before this file -- only
// its pure derivation (`deriveWorkflowEditorState.ts`) is covered by
// `deriveWorkflowEditorState.test.ts`. `mock.module` replaces
// `@/lib/workflows` before `WorkflowEditorRoute` is imported -- a
// dynamic `import()` inside the test body is what makes that ordering
// hold (`SessionsRoute.test.tsx`'s established pattern). Every
// `mock.module` call returns the full named-export set `@/lib/workflows`
// carries, since the replacement is process-wide for the rest of this
// `bun test` run.
//
// `useWorkflowDraftStore` is a real module-level Zustand store, not
// mocked (08-09-SUMMARY.md's Pattern 2): each test resets it in
// `afterEach` via `loadBlank()`.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"

import { useWorkflowDraftStore } from "@/stores/workflowDraftStore"

import type * as React from "react"

afterEach(() => {
  cleanup()
  useWorkflowDraftStore.getState().loadBlank()
})

function stubWorkflows(
  queryClient: QueryClient,
  options: {
    create?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/workflows", () => ({
    WORKFLOWS_QUERY_KEY: ["workflows"],
    workflowQueryKey: (id: number) => ["workflows", id],
    fetchWorkflows: notStubbed("fetchWorkflows"),
    fetchWorkflow: notStubbed("fetchWorkflow"),
    createWorkflowMutationOptions: {
      mutationFn: options.create ?? notStubbed("createWorkflow"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["workflows"] })
      },
    },
    replaceWorkflowStepsMutationOptions: { mutationFn: notStubbed("replaceSteps"), onSuccess: () => {} },
    cancelWorkflowMutationOptions: { mutationFn: notStubbed("cancelWorkflow"), onSuccess: () => {} },
  }))
}

function renderRoute(WorkflowEditorRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/workflows/new"]}>
        <Routes>
          <Route path="/workflows/new" element={<WorkflowEditorRoute />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

test("adding a step appends one row to the editor's step list", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubWorkflows(queryClient, {})
  const { WorkflowEditorRoute } = await import("./WorkflowEditorRoute")

  renderRoute(WorkflowEditorRoute, queryClient)
  await screen.findByLabelText("Duration")

  expect(screen.queryAllByRole("listitem")).toHaveLength(0)

  fireEvent.change(screen.getByLabelText("Duration"), { target: { value: "5" } })
  fireEvent.click(screen.getByRole("button", { name: "Add step" }))

  expect(await screen.findByText("Wait 5 seconds")).toBeTruthy()
  expect(screen.getAllByRole("listitem")).toHaveLength(1)
})

test("the delay field accepts a value and carries it into the submitted payload", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const createCalls: { steps: { kind: string; arguments: Record<string, unknown> }[] }[] = []
  stubWorkflows(queryClient, {
    create: async (input) => {
      createCalls.push(input as (typeof createCalls)[number])
      return {
        id: 1,
        origin: "manual",
        status: "pending",
        summary: "test run",
        created_at: "2026-01-01T00:00:00+00:00",
        updated_at: "2026-01-01T00:00:00+00:00",
        created_by_user_id: 1,
        step_count: 1,
        late: false,
        steps: [],
        reply_synthesis_degraded: false,
        reply_synthesis_message: null,
      }
    },
  })
  const { WorkflowEditorRoute } = await import("./WorkflowEditorRoute")

  renderRoute(WorkflowEditorRoute, queryClient)
  await screen.findByLabelText("Duration")

  fireEvent.change(screen.getByLabelText("Summary"), { target: { value: "Lights off later" } })
  fireEvent.change(screen.getByLabelText("Run at"), { target: { value: "2099-01-01T12:00" } })
  fireEvent.change(screen.getByLabelText("Duration"), { target: { value: "45" } })
  fireEvent.click(screen.getByRole("button", { name: "Add step" }))
  await screen.findByText("Wait 45 seconds")

  fireEvent.click(screen.getByRole("button", { name: "Save workflow" }))

  await waitFor(() => expect(createCalls).toHaveLength(1))
  expect(createCalls[0]?.steps).toEqual([{ kind: "wait", arguments: { duration_s: 45 } }])
})
