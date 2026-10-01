// `mock.module` replaces `@/lib/desktopDevices` before the route is
// imported. A dynamic `import()` inside each test keeps that ordering
// (the EdgeDevicesRoute test uses the same pattern).
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
    name: "desk",
    created_at: "2026-10-01T00:00:00Z",
    revoked: false,
    last_seen_at: null,
    connected: false,
    edge_device_id: null,
    is_default: false,
    ...overrides,
  }
}

function stubDesktopDevices(
  queryClient: QueryClient,
  options: {
    fetchDesktopDevices: () => Promise<unknown[]>
    createDesktopDevice?: (input: unknown) => Promise<unknown>
    location?: { host: string; protocol: string }
  },
) {
  const location = options.location ?? { host: "svr.test", protocol: "https:" }
  mock.module("@/lib/desktopDevices", () => ({
    DESKTOP_DEVICES_QUERY_KEY: ["desktop-devices"],
    desktopDevicesQueryOptions: {
      queryKey: ["desktop-devices"],
      queryFn: options.fetchDesktopDevices,
    },
    createDesktopDeviceMutationOptions: {
      mutationFn:
        options.createDesktopDevice ??
        (async () => {
          throw new Error("createDesktopDevice is not stubbed in this test")
        }),
      onSuccess: () => {
        void queryClient.invalidateQueries({ queryKey: ["desktop-devices"] })
      },
    },
    buildPairLink: (host: string, token: string) =>
      `atlas://pair?server=${encodeURIComponent(host)}&token=${encodeURIComponent(token)}`,
    pageLocation: () => location,
  }))
}

function renderRoute(DesktopDevicesRoute: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <DesktopDevicesRoute />
    </QueryClientProvider>,
  )
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

function stubClipboard(writeText: (text: string) => Promise<void>) {
  const original = navigator.clipboard
  Object.defineProperty(navigator, "clipboard", { value: { writeText }, configurable: true })
  return () => Object.defineProperty(navigator, "clipboard", { value: original, configurable: true })
}

async function addMac(name = "desk") {
  await screen.findByText("Macs")
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: name } })
  fireEvent.click(screen.getByRole("button", { name: "Add Mac" }))
}

const created = { id: 1, name: "desk", token: "t-1", created_at: "2026-10-01T00:00:00Z" }

test("Add Mac is disabled while Name is empty, and the Name input allows 64 characters", async () => {
  const queryClient = newClient()
  stubDesktopDevices(queryClient, { fetchDesktopDevices: async () => [] })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")
  renderRoute(DesktopDevicesRoute, queryClient)

  await screen.findByText("No Macs paired yet.")
  const button = screen.getByRole("button", { name: "Add Mac" }) as HTMLButtonElement
  expect(button.disabled).toBe(true)
  expect((screen.getByLabelText("Name") as HTMLInputElement).maxLength).toBe(64)

  fireEvent.change(screen.getByLabelText("Name"), { target: { value: "desk" } })
  expect(button.disabled).toBe(false)
})

test("a pending create shows Adding… and the panel then shows the message, link, token, buttons and anchor", async () => {
  const queryClient = newClient()
  let release: (value: unknown) => void = () => {}
  stubDesktopDevices(queryClient, {
    fetchDesktopDevices: async () => [],
    createDesktopDevice: (input) => {
      expect(input).toEqual({ name: "desk" })
      return new Promise((resolve) => {
        release = resolve
      })
    },
  })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")
  renderRoute(DesktopDevicesRoute, queryClient)

  await screen.findByText("No Macs paired yet.")
  await addMac()

  const pending = (await screen.findByRole("button", { name: "Adding…" })) as HTMLButtonElement
  expect(pending.disabled).toBe(true)

  release(created)
  expect(await screen.findByText("Mac added.")).toBeTruthy()
  expect(
    screen.getByText("This token will not be shown again. Open the pair link on that Mac, or paste the token into the ATLAS app."),
  ).toBeTruthy()
  expect(screen.getByText("atlas://pair?server=svr.test&token=t-1")).toBeTruthy()
  expect(screen.getByText("t-1")).toBeTruthy()
  expect(screen.getByRole("button", { name: "Copy link" })).toBeTruthy()
  expect(screen.getByRole("button", { name: "Copy token" })).toBeTruthy()
  expect(screen.getByRole("button", { name: "Dismiss" })).toBeTruthy()
  const open = screen.getByRole("link", { name: "Open in ATLAS" })
  expect(open.getAttribute("href")).toBe("atlas://pair?server=svr.test&token=t-1")
  expect(screen.queryByText(/not on https/)).toBeNull()
})

test("Copy link turns to Copied only after writeText resolves, and the two buttons keep separate states", async () => {
  const queryClient = newClient()
  stubDesktopDevices(queryClient, {
    fetchDesktopDevices: async () => [],
    createDesktopDevice: async () => created,
  })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")

  let resolveWrite: () => void = () => {}
  const written: string[] = []
  const restore = stubClipboard(
    (text) =>
      new Promise<void>((resolve) => {
        written.push(text)
        resolveWrite = resolve
      }),
  )
  try {
    renderRoute(DesktopDevicesRoute, queryClient)
    await addMac()
    await screen.findByText("Mac added.")

    fireEvent.click(screen.getByRole("button", { name: "Copy link" }))
    expect(screen.queryByText("Copied")).toBeNull()
    expect(screen.getByRole("button", { name: "Copy link" })).toBeTruthy()

    resolveWrite()
    expect(await screen.findByRole("button", { name: "Copied" })).toBeTruthy()
    expect(written).toEqual(["atlas://pair?server=svr.test&token=t-1"])
    expect(screen.getByRole("button", { name: "Copy token" })).toBeTruthy()
  } finally {
    restore()
  }
})

test("a rejected clipboard write shows the manual-copy line and never Copied", async () => {
  const queryClient = newClient()
  stubDesktopDevices(queryClient, {
    fetchDesktopDevices: async () => [],
    createDesktopDevice: async () => created,
  })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")

  const restore = stubClipboard(async () => {
    throw new Error("denied")
  })
  try {
    renderRoute(DesktopDevicesRoute, queryClient)
    await addMac()
    await screen.findByText("Mac added.")

    fireEvent.click(screen.getByRole("button", { name: "Copy token" }))
    expect(
      await screen.findByText("Could not copy automatically. Select the text above and copy it manually."),
    ).toBeTruthy()
    expect(screen.queryByText("Copied")).toBeNull()
    // The link button kept its own idle state.
    expect(screen.getByRole("button", { name: "Copy link" })).toBeTruthy()
  } finally {
    restore()
  }
})

test("a list refetch keeps the panel, and Dismiss removes the token from the document", async () => {
  const queryClient = newClient()
  let listCalls = 0
  stubDesktopDevices(queryClient, {
    fetchDesktopDevices: async () => {
      listCalls += 1
      return []
    },
    createDesktopDevice: async () => created,
  })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")
  renderRoute(DesktopDevicesRoute, queryClient)

  await addMac()
  await screen.findByText("t-1")
  await waitFor(() => expect(listCalls).toBeGreaterThan(1))
  expect(screen.getByText("t-1")).toBeTruthy()

  fireEvent.click(screen.getByRole("button", { name: "Dismiss" }))
  expect(screen.queryByText("t-1")).toBeNull()
  expect(document.body.textContent).not.toContain("t-1")
})

test("the wss warning shows on http and not on https", async () => {
  const queryClient = newClient()
  stubDesktopDevices(queryClient, {
    fetchDesktopDevices: async () => [],
    createDesktopDevice: async () => created,
    location: { host: "svr.test", protocol: "http:" },
  })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")
  renderRoute(DesktopDevicesRoute, queryClient)

  await addMac()
  expect(
    await screen.findByText(
      "This page is not on https. The Mac app pairs over wss:// only, so the server must be reachable over https.",
    ),
  ).toBeTruthy()
})

test("a 409 shows the server message under the field, and any other failure shows the fallback", async () => {
  const queryClient = newClient()
  const { ApiError } = await import("@/lib/api")
  let attempt = 0
  stubDesktopDevices(queryClient, {
    fetchDesktopDevices: async () => [],
    createDesktopDevice: async () => {
      attempt += 1
      if (attempt === 1) throw new ApiError(409, "another active Mac already uses this name")
      throw new Error("boom")
    },
  })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")
  renderRoute(DesktopDevicesRoute, queryClient)

  await addMac()
  expect(await screen.findByText("another active Mac already uses this name")).toBeTruthy()

  fireEvent.click(screen.getByRole("button", { name: "Add Mac" }))
  expect(await screen.findByText("Could not add the Mac. Try again.")).toBeTruthy()
  expect(screen.queryByText("another active Mac already uses this name")).toBeNull()
})

test("rows show their badge and last-seen line, online first, with Default and no line for revoked-never-seen", async () => {
  const queryClient = newClient()
  const recently = new Date(Date.now() - 5 * 60_000).toISOString()
  stubDesktopDevices(queryClient, {
    fetchDesktopDevices: async () => [
      sampleDevice({ id: 1, name: "old-revoked", revoked: true }),
      sampleDevice({ id: 2, name: "never", last_seen_at: null }),
      sampleDevice({ id: 3, name: "away", last_seen_at: recently }),
      sampleDevice({ id: 4, name: "live", connected: true, is_default: true }),
    ],
  })
  const { DesktopDevicesRoute } = await import("./DesktopDevicesRoute")
  renderRoute(DesktopDevicesRoute, queryClient)

  await screen.findByText("live")
  const names = screen.getAllByRole("listitem").map((li) => li.querySelector("span")?.textContent)
  expect(names).toEqual(["live", "away", "never", "old-revoked"])

  const live = screen.getByText("live").closest("li") as HTMLElement
  expect(within(live).getByText("Online")).toBeTruthy()
  expect(within(live).getByText("Connected now")).toBeTruthy()
  expect(within(live).getByText("Default")).toBeTruthy()

  const away = screen.getByText("away").closest("li") as HTMLElement
  expect(within(away).getByText("Offline")).toBeTruthy()
  expect(within(away).getByText("Last seen 5 minutes ago")).toBeTruthy()

  const revoked = screen.getByText("old-revoked").closest("li") as HTMLElement
  expect(within(revoked).getByText("Revoked")).toBeTruthy()
  expect(within(revoked).getByText("Never connected")).toBeTruthy()
})
