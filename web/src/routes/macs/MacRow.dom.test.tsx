// `mock.module` replaces `@/lib/desktopDevices` before `MacRow` is imported.
// A dynamic `import()` inside each test keeps that ordering (the
// EdgeDevicesRoute test uses the same pattern).
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"

afterEach(() => {
  cleanup()
})

function sampleDevice(overrides: Record<string, unknown> = {}) {
  return {
    id: 7,
    name: "Desk Mac",
    created_at: "2026-10-01T00:00:00Z",
    revoked: false,
    last_seen_at: null,
    connected: true,
    edge_device_id: null,
    is_default: false,
    ...overrides,
  }
}

interface Handlers {
  update?: (input: unknown) => Promise<unknown>
  revoke?: (input: unknown) => Promise<unknown>
  test?: (input: unknown) => Promise<unknown>
}

function stubDesktopDevices(queryClient: QueryClient, handlers: Handlers) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  const invalidate = () => {
    void queryClient.invalidateQueries({ queryKey: ["desktop-devices"] })
  }
  mock.module("@/lib/desktopDevices", () => ({
    DESKTOP_DEVICES_QUERY_KEY: ["desktop-devices"],
    desktopDevicesQueryOptions: { queryKey: ["desktop-devices"], queryFn: async () => [] },
    updateDesktopDeviceMutationOptions: {
      mutationFn: handlers.update ?? notStubbed("update"),
      onSuccess: invalidate,
    },
    revokeDesktopDeviceMutationOptions: {
      mutationFn: handlers.revoke ?? notStubbed("revoke"),
      onSuccess: invalidate,
    },
    testDesktopDeviceMutationOptions: { mutationFn: handlers.test ?? notStubbed("test") },
  }))
}

function newClient() {
  return new QueryClient({ defaultOptions: { queries: { retry: false } } })
}

async function renderRow(
  device: ReturnType<typeof sampleDevice>,
  handlers: Handlers,
  controlsDisabled = false,
) {
  const queryClient = newClient()
  stubDesktopDevices(queryClient, handlers)
  const { MacRow } = await import("./MacRow")
  return render(
    <QueryClientProvider client={queryClient}>
      <ul>
        <MacRow device={device} now={new Date()} controlsDisabled={controlsDisabled} />
      </ul>
    </QueryClientProvider>,
  )
}

function lineText(text: string) {
  return screen.getByText((_, element) => element?.tagName === "P" && element.textContent === text)
}

test("a revoked row renders no Rename, Test, Revoke, Default Mac or Room control", async () => {
  await renderRow(sampleDevice({ revoked: true, connected: false }), {})
  expect(screen.getByText("Revoked")).toBeTruthy()
  expect(screen.queryByRole("button", { name: "Rename" })).toBeNull()
  expect(screen.queryByRole("button", { name: "Test" })).toBeNull()
  expect(screen.queryByRole("button", { name: "Revoke" })).toBeNull()
  expect(screen.queryByRole("checkbox")).toBeNull()
  expect(screen.queryByText("Default Mac")).toBeNull()
  expect(screen.queryByText("Room")).toBeNull()
})

test("Rename opens a prefilled focused input, Enter sends only the name", async () => {
  const calls: unknown[] = []
  await renderRow(sampleDevice(), {
    update: async (input) => {
      calls.push(input)
      return sampleDevice({ name: "Studio" })
    },
  })

  fireEvent.click(screen.getByRole("button", { name: "Rename" }))
  const input = screen.getByLabelText("Mac name") as HTMLInputElement
  expect(input.value).toBe("Desk Mac")
  expect(input.maxLength).toBe(64)
  await waitFor(() => expect(document.activeElement).toBe(input))

  fireEvent.change(input, { target: { value: "Studio" } })
  fireEvent.keyDown(input, { key: "Enter" })
  await waitFor(() => expect(calls).toEqual([{ deviceId: 7, changes: { name: "Studio" } }]))
  await waitFor(() => expect(screen.queryByLabelText("Mac name")).toBeNull())
})

test("Escape closes the rename input without a request and focus returns to Rename", async () => {
  const calls: unknown[] = []
  await renderRow(sampleDevice(), {
    update: async (input) => {
      calls.push(input)
      return sampleDevice()
    },
  })

  const rename = screen.getByRole("button", { name: "Rename" })
  fireEvent.click(rename)
  const input = screen.getByLabelText("Mac name")
  fireEvent.change(input, { target: { value: "Other" } })
  fireEvent.keyDown(input, { key: "Escape" })

  expect(screen.queryByLabelText("Mac name")).toBeNull()
  expect(screen.getByText("Desk Mac")).toBeTruthy()
  await waitFor(() => expect(document.activeElement).toBe(screen.getByRole("button", { name: "Rename" })))
  expect(calls).toEqual([])
})

test("a rename 409 shows the server message, and any other failure shows the fallback", async () => {
  const { ApiError } = await import("@/lib/api")
  let attempt = 0
  await renderRow(sampleDevice(), {
    update: async () => {
      attempt += 1
      if (attempt === 1) throw new ApiError(409, "another active Mac already uses this name")
      throw new Error("boom")
    },
  })

  fireEvent.click(screen.getByRole("button", { name: "Rename" }))
  fireEvent.change(screen.getByLabelText("Mac name"), { target: { value: "Studio" } })
  fireEvent.click(screen.getByRole("button", { name: "Save" }))
  expect(await screen.findByText("another active Mac already uses this name")).toBeTruthy()

  fireEvent.click(screen.getByRole("button", { name: "Save" }))
  expect(await screen.findByText("Could not rename the Mac. Try again.")).toBeTruthy()
  expect(screen.queryByText("another active Mac already uses this name")).toBeNull()
})

test("Default Mac sends is_default true when checked and false when unchecked", async () => {
  const calls: unknown[] = []
  const handlers: Handlers = {
    update: async (input) => {
      calls.push(input)
      return sampleDevice()
    },
  }
  const { unmount } = await renderRow(sampleDevice({ is_default: false }), handlers)
  fireEvent.click(screen.getByRole("checkbox", { name: "Default Mac" }))
  await waitFor(() => expect(calls).toEqual([{ deviceId: 7, changes: { is_default: true } }]))
  unmount()

  await renderRow(sampleDevice({ is_default: true }), handlers)
  expect(screen.getByText("Default")).toBeTruthy()
  fireEvent.click(screen.getByRole("checkbox", { name: "Default Mac" }))
  await waitFor(() => expect(calls[1]).toEqual({ deviceId: 7, changes: { is_default: false } }))
})

test("a failed default change shows the fallback line", async () => {
  await renderRow(sampleDevice(), {
    update: async () => {
      throw new Error("boom")
    },
  })
  fireEvent.click(screen.getByRole("checkbox", { name: "Default Mac" }))
  expect(await screen.findByText("Could not change the default Mac. Try again.")).toBeTruthy()
})

test("Test is disabled on an offline row and the offline line shows", async () => {
  await renderRow(sampleDevice({ connected: false }), {})
  expect((screen.getByRole("button", { name: "Test" }) as HTMLButtonElement).disabled).toBe(true)
  expect(screen.getByText("An offline Mac cannot be tested.")).toBeTruthy()
})

test("Test shows Testing… while pending, then Answered in 42 ms", async () => {
  let release: (value: unknown) => void = () => {}
  await renderRow(sampleDevice(), {
    test: (input) => {
      expect(input).toEqual({ deviceId: 7 })
      return new Promise((resolve) => {
        release = resolve
      })
    },
  })

  fireEvent.click(screen.getByRole("button", { name: "Test" }))
  const pending = (await screen.findByRole("button", { name: "Testing…" })) as HTMLButtonElement
  expect(pending.disabled).toBe(true)

  release({ answered: true, rtt_ms: 42 })
  await waitFor(() => lineText("Answered in 42 ms"))
  expect(screen.getByRole("button", { name: "Test" })).toBeTruthy()
})

test("Test shows the timeout line for answered false, and the request-failed line for a rejection", async () => {
  const { ApiError } = await import("@/lib/api")
  let attempt = 0
  await renderRow(sampleDevice(), {
    test: async () => {
      attempt += 1
      if (attempt === 1) return { answered: false, rtt_ms: null }
      throw new ApiError(409, "this Mac is not connected")
    },
  })

  fireEvent.click(screen.getByRole("button", { name: "Test" }))
  const timeout = await screen.findByText(
    "No answer after 5 seconds. Make sure the Mac is awake and online, then try again.",
  )
  expect(timeout.className).toContain("text-destructive")

  fireEvent.click(screen.getByRole("button", { name: "Test" }))
  const failed = await screen.findByText("Could not run the test. Try again.")
  expect(failed.className).toContain("text-destructive")
  expect(screen.queryByText(/No answer after 5 seconds/)).toBeNull()
})

test("Revoke opens a dialog naming the Mac; Cancel sends nothing, Revoke Mac sends one revoke", async () => {
  const calls: unknown[] = []
  await renderRow(sampleDevice(), {
    revoke: async (input) => {
      calls.push(input)
    },
  })

  fireEvent.click(screen.getByRole("button", { name: "Revoke" }))
  expect(await screen.findByText("Revoke Desk Mac?")).toBeTruthy()
  expect(
    screen.getByText("The Mac disconnects now and cannot reconnect with this token. It will ask to pair again."),
  ).toBeTruthy()
  fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
  await waitFor(() => expect(screen.queryByText("Revoke Desk Mac?")).toBeNull())
  expect(calls).toEqual([])

  fireEvent.click(screen.getByRole("button", { name: "Revoke" }))
  await screen.findByText("Revoke Desk Mac?")
  fireEvent.click(screen.getByRole("button", { name: "Revoke Mac" }))
  await waitFor(() => expect(calls).toEqual([{ deviceId: 7 }]))
})

test("a failed revoke shows the fallback line under the row", async () => {
  await renderRow(sampleDevice(), {
    revoke: async () => {
      throw new Error("boom")
    },
  })
  fireEvent.click(screen.getByRole("button", { name: "Revoke" }))
  await screen.findByText("Revoke Desk Mac?")
  fireEvent.click(screen.getByRole("button", { name: "Revoke Mac" }))
  expect(await screen.findByText("Could not revoke the Mac. Try again.")).toBeTruthy()
})

test("every control is disabled while the screen is not ready", async () => {
  await renderRow(sampleDevice(), {}, true)
  const row = screen.getByRole("listitem")
  for (const name of ["Rename", "Test", "Revoke"]) {
    expect((within(row).getByRole("button", { name }) as HTMLButtonElement).disabled).toBe(true)
  }
  expect(within(row).getByRole("checkbox", { name: "Default Mac" }).hasAttribute("disabled")).toBe(true)
})
