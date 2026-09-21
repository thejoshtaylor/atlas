// D-17 backfill (08-09-PLAN.md), Task 2. `mock.module` replaces
// `@/lib/policy` before `PolicyRoute` is imported -- a dynamic
// `import()` inside the test body is what makes that ordering hold,
// since a static `import` at the top of this file would be hoisted
// ahead of `mock.module` and pick up the real module instead
// (`SessionsRoute.test.tsx`'s own established pattern).
//
// Every `mock.module` call below returns the full `@/lib/policy` named-
// export set, not just what its own test reads -- the mock replaces the
// module registry entry for the rest of this `bun test` process,
// including `PolicyRoute.test.ts`'s own (regex-only, no runtime import)
// sibling. An incomplete stub here would silently break any future
// runtime consumer of this module in this directory.
//
// Each test builds its own `QueryClient` rather than sharing the app's
// singleton (`@/lib/queryClient`) -- the mutation stubs below call
// `invalidateQueries` on that same test-scoped client, matching the real
// module's `onSuccess` shape exactly, so a successful add or remove
// triggers a real refetch without leaking query cache state between
// tests via the app-wide singleton.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"

import { usePolicyDraftStore } from "@/stores/policyDraftStore"

import type * as React from "react"

afterEach(() => {
  cleanup()
  usePolicyDraftStore.getState().clear()
})

function sampleRule(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    kind: "deny_entity",
    value: "light.kitchen",
    note: null,
    created_at: "2026-01-01T00:00:00+00:00",
    resolved: true,
    ...overrides,
  }
}

function samplePolicy(overrides: Record<string, unknown> = {}) {
  return {
    mode: "allow_all_except_denylist",
    rules: [sampleRule()],
    applies_live: true,
    ...overrides,
  }
}

function stubPolicy(
  queryClient: QueryClient,
  options: {
    fetchPolicy: () => Promise<unknown>
    addRule?: (input: unknown) => Promise<unknown>
    removeRule?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/policy", () => ({
    POLICY_QUERY_KEY: ["policy"],
    fetchPolicy: options.fetchPolicy,
    addRuleMutationOptions: {
      mutationFn: options.addRule ?? notStubbed("addRule"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["policy"] })
      },
    },
    removeRuleMutationOptions: {
      mutationFn: options.removeRule ?? notStubbed("removeRule"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["policy"] })
      },
    },
    setModeMutationOptions: {
      mutationFn: notStubbed("setMode"),
      onSuccess: () => {},
    },
  }))
}

function renderRoute(PolicyRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <PolicyRoute />
    </QueryClientProvider>,
  )
}

test("renders the rules the stubbed fetch returns, one row per rule", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubPolicy(queryClient, {
    fetchPolicy: async () =>
      samplePolicy({
        rules: [sampleRule({ id: 1, value: "light.kitchen" }), sampleRule({ id: 2, value: "lock.front_door" })],
      }),
  })
  const { PolicyRoute } = await import("./PolicyRoute")

  renderRoute(PolicyRoute, queryClient)

  expect(await screen.findByText("light.kitchen")).toBeTruthy()
  expect(screen.getByText("lock.front_door")).toBeTruthy()
})

test("a pending query renders the loading skeleton, not an empty list", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubPolicy(queryClient, { fetchPolicy: () => new Promise(() => {}) })
  const { PolicyRoute } = await import("./PolicyRoute")

  renderRoute(PolicyRoute, queryClient)

  expect(screen.getByRole("status", { name: "Loading" })).toBeTruthy()
})

test("adding a rule calls the add mutation with the typed value, and the new rule appears once the query returns it", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let fetchCount = 0
  const addCalls: unknown[] = []
  stubPolicy(queryClient, {
    fetchPolicy: async () => {
      fetchCount += 1
      return fetchCount === 1
        ? samplePolicy({ rules: [sampleRule({ id: 1, value: "light.kitchen" })] })
        : samplePolicy({
            rules: [sampleRule({ id: 1, value: "light.kitchen" }), sampleRule({ id: 2, value: "lock.front_door" })],
          })
    },
    addRule: async (input) => {
      addCalls.push(input)
      return sampleRule({ id: 2, value: "lock.front_door" })
    },
  })
  const { PolicyRoute } = await import("./PolicyRoute")

  renderRoute(PolicyRoute, queryClient)
  await screen.findByText("light.kitchen")

  fireEvent.change(screen.getByLabelText("Entity id"), { target: { value: "lock.front_door" } })
  fireEvent.click(screen.getByRole("button", { name: "Add to denylist" }))

  expect(await screen.findByText("lock.front_door")).toBeTruthy()
  expect(addCalls).toEqual([{ kind: "deny_entity", value: "lock.front_door" }])
})

test("submitting the add form with an empty required field does not call the mutation", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const addCalls: unknown[] = []
  stubPolicy(queryClient, {
    fetchPolicy: async () => samplePolicy(),
    addRule: async (input) => {
      addCalls.push(input)
      return sampleRule()
    },
  })
  const { PolicyRoute } = await import("./PolicyRoute")

  renderRoute(PolicyRoute, queryClient)
  await screen.findByText("light.kitchen")

  const addButton = screen.getByRole("button", { name: "Add to denylist" })
  expect((addButton as HTMLButtonElement).disabled).toBe(true)

  fireEvent.click(addButton)
  expect(addCalls).toEqual([])
})

test("removing a rule calls the remove mutation for that rule and not another", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const removeCalls: unknown[] = []
  stubPolicy(queryClient, {
    fetchPolicy: async () =>
      samplePolicy({
        rules: [sampleRule({ id: 1, value: "light.kitchen" }), sampleRule({ id: 2, value: "lock.front_door" })],
      }),
    removeRule: async (input) => {
      removeCalls.push(input)
      return undefined
    },
  })
  const { PolicyRoute } = await import("./PolicyRoute")

  renderRoute(PolicyRoute, queryClient)
  await screen.findByText("lock.front_door")

  const rows = screen.getAllByRole("listitem")
  const targetRow = rows.find((row) => row.textContent?.includes("lock.front_door"))
  if (!targetRow) throw new Error("expected to find the lock.front_door row")
  fireEvent.click(within(targetRow).getByRole("button", { name: "Remove" }))

  fireEvent.click(await screen.findByRole("button", { name: "Remove from denylist" }))

  await waitFor(() => expect(removeCalls).toEqual([{ ruleId: 2 }]))
})

test("a failed load renders the error state, and Retry refetches successfully", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let callCount = 0
  stubPolicy(queryClient, {
    fetchPolicy: async () => {
      callCount += 1
      if (callCount === 1) throw new Error("boom")
      return samplePolicy()
    },
  })
  const { PolicyRoute } = await import("./PolicyRoute")

  renderRoute(PolicyRoute, queryClient)

  const retryButton = await screen.findByText("Retry")
  fireEvent.click(retryButton)

  expect(await screen.findByText("light.kitchen")).toBeTruthy()
})

test("every interactive mode row carries the touch-target class", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubPolicy(queryClient, { fetchPolicy: async () => samplePolicy() })
  const { PolicyRoute } = await import("./PolicyRoute")

  const { container } = renderRoute(PolicyRoute, queryClient)
  await screen.findByText("light.kitchen")

  const touchTargets = container.querySelectorAll(".touch-target")
  expect(touchTargets.length).toBeGreaterThanOrEqual(2)
  for (const element of touchTargets) {
    expect(element.querySelector('[role="radio"]')).toBeTruthy()
  }
})
