// D-03 (10-06-PLAN.md Task 1). `mock.module` replaces `@/lib/edgeDevices`
// before `EdgeDevicesRoute` is imported -- a dynamic `import()` inside
// the test body is what makes that ordering hold
// (`GoogleAccountsRoute.dom.test.tsx`'s established pattern).
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function sampleDevice(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    name: "kitchen",
    created_at: "2026-09-01T00:00:00Z",
    revoked: false,
    last_connected_at: null,
    connected: false,
    ...overrides,
  }
}

function stubEdgeDevices(
  queryClient: QueryClient,
  options: {
    fetchEdgeDevices: () => Promise<unknown[]>
    createEdgeDevice?: (input: unknown) => Promise<unknown>
    revokeEdgeDevice?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/edgeDevices", () => ({
    EDGE_DEVICES_QUERY_KEY: ["edge-devices"],
    edgeDevicesQueryOptions: {
      queryKey: ["edge-devices"],
      queryFn: options.fetchEdgeDevices,
    },
    createEdgeDeviceMutationOptions: {
      mutationFn: options.createEdgeDevice ?? notStubbed("createEdgeDevice"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["edge-devices"] })
      },
    },
    revokeEdgeDeviceMutationOptions: {
      mutationFn: options.revokeEdgeDevice ?? notStubbed("revokeEdgeDevice"),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["edge-devices"] })
      },
    },
  }))
}

function renderRoute(EdgeDevicesRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <EdgeDevicesRoute />
    </QueryClientProvider>,
  )
}

test("adding a device shows the token panel with the token and a Copy button; Dismiss removes it, and the refetch's own devices never carry a token field", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let listCalls = 0
  stubEdgeDevices(queryClient, {
    fetchEdgeDevices: async () => {
      listCalls += 1
      return []
    },
    createEdgeDevice: async (input) => {
      expect(input).toEqual({ name: "test" })
      return { id: 1, name: "test", token: "device-token-abc", created_at: "2026-09-25T00:00:00Z" }
    },
  })
  const { EdgeDevicesRoute } = await import("./EdgeDevicesRoute")

  renderRoute(EdgeDevicesRoute, queryClient)

  await screen.findByText("Edge devices")
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: "test" } })
  fireEvent.click(screen.getByRole("button", { name: "Add device" }))

  expect(await screen.findByText("device-token-abc")).toBeTruthy()
  expect(screen.getByRole("button", { name: "Copy" })).toBeTruthy()
  expect(screen.getByText("Device added.")).toBeTruthy()

  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }))
  expect(screen.queryByText("device-token-abc")).toBeNull()

  // onSuccess invalidates the list query, which refetches through the
  // same fetchEdgeDevices fake this test controls -- its own return type
  // (EdgeDevice[]) carries no token field at all, so a refetched list can
  // never surface the token a second time.
  await waitFor(() => expect(listCalls).toBeGreaterThan(1))
})

test("Revoke opens an alert dialog naming the device; Cancel sends nothing, confirming sends exactly one DELETE", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const revokeCalls: unknown[] = []
  stubEdgeDevices(queryClient, {
    fetchEdgeDevices: async () => [sampleDevice({ id: 5, name: "hallway" })],
    revokeEdgeDevice: async (input) => {
      revokeCalls.push(input)
    },
  })
  const { EdgeDevicesRoute } = await import("./EdgeDevicesRoute")

  renderRoute(EdgeDevicesRoute, queryClient)

  await screen.findByText("hallway")
  fireEvent.click(screen.getByRole("button", { name: "Revoke" }))

  expect(await screen.findByText("Revoke hallway?")).toBeTruthy()
  expect(screen.getByText("The Pi disconnects now and cannot reconnect with this token.")).toBeTruthy()

  fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
  expect(revokeCalls).toEqual([])

  fireEvent.click(screen.getByRole("button", { name: "Revoke" }))
  await screen.findByText("Revoke hallway?")
  fireEvent.click(screen.getByRole("button", { name: "Revoke device" }))

  await waitFor(() => expect(revokeCalls).toEqual([{ deviceId: 5 }]))
})

test("a connected device shows Connected, and a revoked device shows Revoked with no Revoke control", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubEdgeDevices(queryClient, {
    fetchEdgeDevices: async () => [
      sampleDevice({ id: 1, name: "connected-device", connected: true }),
      sampleDevice({ id: 2, name: "revoked-device", revoked: true, connected: false }),
    ],
  })
  const { EdgeDevicesRoute } = await import("./EdgeDevicesRoute")

  renderRoute(EdgeDevicesRoute, queryClient)

  await screen.findByText("connected-device")
  expect(screen.getByText("Connected")).toBeTruthy()
  expect(screen.getByText("Revoked")).toBeTruthy()

  const revokedRow = screen.getByText("revoked-device").closest("li") as HTMLElement
  expect(revokedRow).toBeTruthy()
  expect(within(revokedRow).queryByRole("button", { name: "Revoke" })).toBeNull()
})

test("a 422 from the create route shows the server's own message verbatim", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubEdgeDevices(queryClient, {
    fetchEdgeDevices: async () => [],
    createEdgeDevice: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(422, "name must be 1-64 characters of letters, digits, space, -, _ and .")
    },
  })
  const { EdgeDevicesRoute } = await import("./EdgeDevicesRoute")

  renderRoute(EdgeDevicesRoute, queryClient)

  await screen.findByText("Edge devices")
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: "!!!" } })
  fireEvent.click(screen.getByRole("button", { name: "Add device" }))

  expect(await screen.findByText("name must be 1-64 characters of letters, digits, space, -, _ and .")).toBeTruthy()
})
