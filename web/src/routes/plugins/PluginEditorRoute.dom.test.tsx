// D-17 backfill (08-09-PLAN.md), Task 3. The editor is the largest
// screen in this application; D-17 says this backfill is not an
// exhaustive suite for every prior screen. Covers its two load-bearing
// interactions only: a validation failure that blocks submission, and a
// successful save carrying the edited values. `mock.module` replaces
// `@/lib/plugins` before `PluginEditorRoute` is imported -- see
// `PluginsRoute.dom.test.tsx`'s own header comment for why the ordering
// and the full-export-set requirement both matter here too.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { usePluginDraftStore } from "@/stores/pluginDraftStore"

import type * as React from "react"

afterEach(() => {
  cleanup()
  usePluginDraftStore.getState().loadBlank()
})

function stubPluginsForEditor(options: { install?: (input: unknown) => Promise<unknown> }) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/plugins", () => ({
    PLUGINS_QUERY_KEY: ["plugins"],
    pluginQueryKey: (id: number) => ["plugins", id],
    PLUGIN_CATALOG_QUERY_KEY: ["plugins", "catalog"],
    fetchPlugins: notStubbed("fetchPlugins"),
    fetchPlugin: notStubbed("fetchPlugin"),
    fetchPluginCatalog: async () => [],
    installPluginMutationOptions: {
      mutationFn: options.install ?? notStubbed("installPlugin"),
      onSuccess: () => {},
    },
    setPluginEnabledMutationOptions: { mutationFn: notStubbed("setEnabled"), onSuccess: () => {} },
    savePluginConfigMutationOptions: { mutationFn: notStubbed("saveConfig"), onSuccess: () => {} },
    deletePluginMutationOptions: { mutationFn: notStubbed("deletePlugin"), onSuccess: () => {} },
  }))
}

function renderEditor(PluginEditorRoute: React.ComponentType) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/plugins/new"]}>
        <PluginEditorRoute />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

test("a blank required field renders a validation message and blocks submission", async () => {
  stubPluginsForEditor({})
  const { PluginEditorRoute } = await import("./PluginEditorRoute")

  renderEditor(PluginEditorRoute)

  fireEvent.click(screen.getByRole("radio", { name: "Custom" }))
  fireEvent.change(screen.getByRole("textbox", { name: "Command" }), { target: { value: "python plugin.py" } })
  // Display name left blank on purpose -- the one required field this
  // test does not fill in.

  const messages = await screen.findAllByText("Add a name before installing.")
  expect(messages.length).toBeGreaterThanOrEqual(1)
  expect((screen.getByRole("button", { name: "Install plugin" }) as HTMLButtonElement).disabled).toBe(true)
})

test("a successful save calls the mutation with the edited values", async () => {
  const calls: unknown[] = []
  stubPluginsForEditor({
    install: async (input) => {
      calls.push(input)
      return { id: 42 }
    },
  })
  const { PluginEditorRoute } = await import("./PluginEditorRoute")

  renderEditor(PluginEditorRoute)

  fireEvent.click(screen.getByRole("radio", { name: "Custom" }))
  fireEvent.change(screen.getByRole("textbox", { name: "Command" }), { target: { value: "python plugin.py" } })
  fireEvent.change(screen.getByRole("textbox", { name: "Name" }), { target: { value: "My plugin" } })

  fireEvent.click(screen.getByRole("button", { name: "Install plugin" }))

  await waitFor(() =>
    expect(calls).toEqual([
      {
        display_name: "My plugin",
        transport: "command",
        command: "python plugin.py",
        url: null,
        timeout_ms: 5000,
        config_values: {},
      },
    ]),
  )
})
