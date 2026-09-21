// D-17 backfill (08-09-PLAN.md), Task 3. This list carries the only
// destructive confirmation dialog in this application, and a
// confirmation dialog is the archetypal thing a regular expression
// cannot check: the source containing a dialog component proves
// nothing about whether dismissing it actually cancels. `mock.module`
// replaces `@/lib/plugins` before `PluginsRoute` is imported -- a
// dynamic `import()` inside the test body is what makes that ordering
// hold (`SessionsRoute.test.tsx`'s established pattern). Every
// `mock.module` call returns the full named-export set `@/lib/plugins`
// carries, since the replacement is process-wide for the rest of this
// `bun test` run, including `PluginEditorRoute.dom.test.tsx`'s own
// sibling file.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function samplePlugin(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    slug: "home-assistant",
    display_name: "Home Assistant",
    transport: "command",
    args: [],
    url: null,
    enabled: true,
    builtin: false,
    timeout_ms: 5000,
    state: "running",
    reason: null,
    tools: [],
    config_values: [],
    applies_live: true,
    ...overrides,
  }
}

function stubPlugins(
  queryClient: QueryClient,
  options: {
    fetchPlugins: () => Promise<unknown[]>
    setEnabled?: (input: unknown) => Promise<unknown>
    deletePlugin?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/plugins", () => ({
    PLUGINS_QUERY_KEY: ["plugins"],
    pluginQueryKey: (id: number) => ["plugins", id],
    PLUGIN_CATALOG_QUERY_KEY: ["plugins", "catalog"],
    fetchPlugins: options.fetchPlugins,
    fetchPlugin: notStubbed("fetchPlugin"),
    fetchPluginCatalog: async () => [],
    installPluginMutationOptions: { mutationFn: notStubbed("installPlugin"), onSuccess: () => {} },
    setPluginEnabledMutationOptions: {
      mutationFn: options.setEnabled ?? notStubbed("setEnabled"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["plugins"] })
      },
    },
    savePluginConfigMutationOptions: { mutationFn: notStubbed("saveConfig"), onSuccess: () => {} },
    deletePluginMutationOptions: {
      mutationFn: options.deletePlugin ?? notStubbed("deletePlugin"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["plugins"] })
      },
    },
  }))
}

function renderRoute(PluginsRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter>
        <PluginsRoute />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

test("renders one row per plugin, each with its own runtime status badge", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubPlugins(queryClient, {
    fetchPlugins: async () => [
      samplePlugin({ id: 1, display_name: "Home Assistant", state: "running" }),
      samplePlugin({ id: 2, display_name: "Weather", state: "degraded", reason: "No API key configured" }),
    ],
  })
  const { PluginsRoute } = await import("./PluginsRoute")

  renderRoute(PluginsRoute, queryClient)

  expect(await screen.findByText("Home Assistant")).toBeTruthy()
  expect(screen.getByText("Weather")).toBeTruthy()
  expect(screen.getByText("Running")).toBeTruthy()
  expect(screen.getByText("Degraded — won't start")).toBeTruthy()
})

test("the row control reads Disable for an enabled plugin and Enable for a disabled one", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubPlugins(queryClient, {
    fetchPlugins: async () => [
      samplePlugin({ id: 1, display_name: "Home Assistant", enabled: true }),
      samplePlugin({ id: 2, display_name: "Weather", enabled: false, state: "disabled" }),
    ],
  })
  const { PluginsRoute } = await import("./PluginsRoute")

  renderRoute(PluginsRoute, queryClient)
  await screen.findByText("Home Assistant")

  const enabledRow = screen.getByText("Home Assistant").closest("li")
  const disabledRow = screen.getByText("Weather").closest("li")
  if (!enabledRow || !disabledRow) throw new Error("expected both rows")

  expect(within(enabledRow).getByRole("button", { name: "Disable" })).toBeTruthy()
  expect(within(disabledRow).getByRole("button", { name: "Enable" })).toBeTruthy()
})

test("toggling a plugin's enabled control calls the enable mutation for that plugin with the new value", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const calls: unknown[] = []
  stubPlugins(queryClient, {
    fetchPlugins: async () => [samplePlugin({ id: 1, display_name: "Home Assistant", enabled: true })],
    setEnabled: async (input) => {
      calls.push(input)
      return samplePlugin({ id: 1, display_name: "Home Assistant", enabled: false })
    },
  })
  const { PluginsRoute } = await import("./PluginsRoute")

  renderRoute(PluginsRoute, queryClient)
  await screen.findByText("Home Assistant")

  fireEvent.click(screen.getByRole("button", { name: "Disable" }))

  await waitFor(() => expect(calls).toEqual([{ pluginId: 1, enabled: false }]))
})

test("dismissing the delete confirmation calls no mutation; confirming calls delete exactly once", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const deleteCalls: unknown[] = []
  stubPlugins(queryClient, {
    fetchPlugins: async () => [samplePlugin({ id: 1, display_name: "Home Assistant", builtin: false })],
    deletePlugin: async (input) => {
      deleteCalls.push(input)
      return undefined
    },
  })
  const { PluginsRoute } = await import("./PluginsRoute")

  renderRoute(PluginsRoute, queryClient)
  const row = (await screen.findByText("Home Assistant")).closest("li")
  if (!row) throw new Error("expected a row")

  // Dismiss: opens then cancels -- no mutation should fire.
  fireEvent.click(within(row).getByRole("button", { name: "Delete" }))
  fireEvent.click(await screen.findByRole("button", { name: "Cancel" }))
  expect(deleteCalls).toEqual([])

  // Confirm: opens again and confirms -- exactly one mutation call.
  fireEvent.click(within(row).getByRole("button", { name: "Delete" }))
  fireEvent.click(await screen.findByRole("button", { name: "Delete plugin" }))
  await waitFor(() => expect(deleteCalls).toEqual([{ pluginId: 1 }]))
})

test("a built-in plugin shows no destructive control at all", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubPlugins(queryClient, {
    fetchPlugins: async () => [samplePlugin({ id: 1, display_name: "Home Assistant", builtin: true })],
  })
  const { PluginsRoute } = await import("./PluginsRoute")

  renderRoute(PluginsRoute, queryClient)
  const row = (await screen.findByText("Home Assistant")).closest("li")
  if (!row) throw new Error("expected a row")

  expect(within(row).queryByRole("button", { name: "Delete" })).toBeNull()
})
