// D-17 backfill part B (08-10-PLAN.md), Task 2. `mock.module` replaces
// `@/lib/macros` before `MacroEditorRoute` is imported -- a dynamic
// `import()` inside the test body is what makes that ordering hold
// (`SessionsRoute.test.tsx`'s established pattern). Every `mock.module`
// call returns the full named-export set `@/lib/macros` carries, since
// the replacement is process-wide for the rest of this `bun test` run.
//
// `useMacroDraftStore` is a real module-level Zustand store, not mocked
// (08-09-SUMMARY.md's Pattern 2): each test resets it in `afterEach` via
// `loadBlank()` rather than stubbing the store module, since the store
// holds no network dependency of its own.
//
// The macro editor builds an ORDERED action list -- the one property a
// source-text regex is least able to check. These tests assert the
// submitted payload's order directly, not just the rendered order, per
// 08-10-PLAN.md's own instruction: the gap between "renders in order"
// and "submits in order" is exactly the kind of bug this harness exists
// to find.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"

import { useMacroDraftStore } from "@/stores/macroDraftStore"

import type * as React from "react"

afterEach(() => {
  cleanup()
  useMacroDraftStore.getState().loadBlank()
})

function sampleAction(overrides: Record<string, unknown> = {}) {
  return {
    id: 10,
    position: 0,
    tool: "ha_call_service",
    arguments: { domain: "light", service: "turn_off", entity_id: "light.kitchen" },
    conflict: "ok",
    ...overrides,
  }
}

function sampleMacro(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    phrase: "turn off the lights",
    aliases: [],
    reply: "Okay.",
    actions: [sampleAction()],
    created_at: "2026-01-01T00:00:00+00:00",
    updated_at: "2026-01-01T00:00:00+00:00",
    created_by_user_id: 1,
    reply_cached: true,
    reply_synthesis_degraded: false,
    reply_synthesis_message: null,
    ...overrides,
  }
}

function stubMacros(
  queryClient: QueryClient,
  options: {
    fetchMacro: () => Promise<unknown>
    fetchMacros?: () => Promise<unknown[]>
    update?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/macros", () => ({
    MACROS_QUERY_KEY: ["macros"],
    macroQueryKey: (id: number) => ["macros", id],
    fetchMacros: options.fetchMacros ?? (async () => []),
    fetchMacro: options.fetchMacro,
    createMacroMutationOptions: { mutationFn: notStubbed("createMacro"), onSuccess: () => {} },
    updateMacroMutationOptions: {
      mutationFn: options.update ?? notStubbed("updateMacro"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["macros"] })
      },
    },
    deleteMacroMutationOptions: { mutationFn: notStubbed("deleteMacro"), onSuccess: () => {} },
  }))
}

function renderRoute(MacroEditorRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/macros/1"]}>
        <Routes>
          <Route path="/macros/:id" element={<MacroEditorRoute />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

test("adding an action appends one row to the action list", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubMacros(queryClient, { fetchMacro: async () => sampleMacro() })
  const { MacroEditorRoute } = await import("./MacroEditorRoute")

  renderRoute(MacroEditorRoute, queryClient)
  await screen.findByText("light.turn_off → light.kitchen")
  expect(screen.getAllByRole("listitem")).toHaveLength(1)

  fireEvent.change(screen.getByLabelText("Domain"), { target: { value: "switch" } })
  fireEvent.change(screen.getByLabelText("Service"), { target: { value: "turn_on" } })
  fireEvent.change(screen.getByLabelText("Entity id"), { target: { value: "switch.fan" } })
  fireEvent.click(screen.getByRole("button", { name: "Add action" }))

  expect(await screen.findByText("switch.turn_on → switch.fan")).toBeTruthy()
  expect(screen.getAllByRole("listitem")).toHaveLength(2)
})

test("removing an action removes that row and leaves the others in their original order", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubMacros(queryClient, {
    fetchMacro: async () =>
      sampleMacro({
        actions: [
          sampleAction({ id: 10, arguments: { domain: "light", service: "turn_off", entity_id: "light.kitchen" } }),
          sampleAction({ id: 11, arguments: { domain: "cover", service: "close_cover", entity_id: "cover.blinds" } }),
          sampleAction({ id: 12, arguments: { domain: "lock", service: "lock", entity_id: "lock.front_door" } }),
        ],
      }),
  })
  const { MacroEditorRoute } = await import("./MacroEditorRoute")

  renderRoute(MacroEditorRoute, queryClient)
  await screen.findByText("cover.close_cover → cover.blinds")

  const middleRow = screen.getByText("cover.close_cover → cover.blinds").closest("li")
  if (!middleRow) throw new Error("expected the middle row")
  fireEvent.click(within(middleRow).getByRole("button", { name: "Remove" }))

  await waitFor(() => expect(screen.queryByText("cover.close_cover → cover.blinds")).toBeNull())
  const remaining = screen.getAllByRole("listitem")
  expect(remaining).toHaveLength(2)
  expect(remaining[0]?.textContent).toContain("light.kitchen")
  expect(remaining[1]?.textContent).toContain("lock.front_door")
})

test("saving submits the actions in the order shown on screen", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const updateCalls: { actions: { arguments: Record<string, unknown> }[] }[] = []
  stubMacros(queryClient, {
    fetchMacro: async () =>
      sampleMacro({
        actions: [
          sampleAction({ id: 10, arguments: { domain: "light", service: "turn_off", entity_id: "light.kitchen" } }),
          sampleAction({ id: 11, arguments: { domain: "lock", service: "lock", entity_id: "lock.front_door" } }),
        ],
      }),
    update: async (input) => {
      updateCalls.push(input as (typeof updateCalls)[number])
      return sampleMacro()
    },
  })
  const { MacroEditorRoute } = await import("./MacroEditorRoute")

  renderRoute(MacroEditorRoute, queryClient)
  await screen.findByText("light.turn_off → light.kitchen")

  // Move the first row (light) down, so lock is now shown first.
  fireEvent.click(screen.getAllByRole("button", { name: "Move down" })[0]!)
  const rows = screen.getAllByRole("listitem")
  expect(rows[0]?.textContent).toContain("lock.front_door")
  expect(rows[1]?.textContent).toContain("light.kitchen")

  fireEvent.click(screen.getByRole("button", { name: "Save macro" }))

  await waitFor(() => expect(updateCalls).toHaveLength(1))
  expect(updateCalls[0]?.actions.map((action) => action.arguments.entity_id)).toEqual([
    "lock.front_door",
    "light.kitchen",
  ])
})

test("removing the last action blocks save and renders the zero-actions message", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubMacros(queryClient, { fetchMacro: async () => sampleMacro() })
  const { MacroEditorRoute } = await import("./MacroEditorRoute")

  renderRoute(MacroEditorRoute, queryClient)
  await screen.findByText("light.turn_off → light.kitchen")

  fireEvent.click(screen.getByRole("button", { name: "Remove" }))

  expect(await screen.findByText("Add at least one action before saving.")).toBeTruthy()
  const saveButton = screen.getByRole("button", { name: "Save macro" }) as HTMLButtonElement
  expect(saveButton.disabled).toBe(true)
})
