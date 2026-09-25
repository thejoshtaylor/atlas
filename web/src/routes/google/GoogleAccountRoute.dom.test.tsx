// GOOG-01, GOOG-02 (09-10-PLAN.md Task 2). `mock.module` replaces
// `@/lib/google` before `GoogleAccountRoute` is imported -- see
// `GoogleAccountsRoute.dom.test.tsx`'s own header comment for why the
// ordering and the full-export-set requirement both matter. Every
// mutation's `onSuccess` in `stubGoogle` writes to the SAME per-test
// `QueryClient` the tree renders with, so an invalidate-then-refetch (or
// a direct `setQueryData`) is real, the Phase 8 D-17 backfill pattern
// this plan's own action text names.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor, within } from "@testing-library/react"
import { MemoryRouter, Route, Routes, useLocation } from "react-router-dom"

// Imported statically, before any `mock.module` call in this file runs, so
// the mock factory below can re-export the module's full, real surface
// (`mock.module` replaces the module process-wide for the rest of this
// `bun test` run -- a sibling file's static `import ... from "@/lib/google"`
// must still find every export it expects, not only the ones this file's
// own tests exercise).
import { LINK_ERROR_MESSAGES, linkErrorMessage } from "@/lib/google"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function sampleCalendar(overrides: Record<string, unknown> = {}) {
  return {
    id: 10,
    calendar_id: "primary",
    name: "Home",
    is_primary: true,
    can_write: true,
    access: "off",
    ...overrides,
  }
}

function sampleAccount(overrides: Record<string, unknown> = {}) {
  return {
    id: 7,
    label: "work",
    email: "operator@example.com",
    is_default: false,
    status: "ok",
    status_detail: null,
    refresh_token_expires_at: null,
    linked_at: "2026-09-01T00:00:00Z",
    calendars: [sampleCalendar()],
    plugin_state: null,
    ...overrides,
  }
}

function stubGoogle(
  queryClient: QueryClient,
  options: {
    fetchGoogleAccount: (id: number) => Promise<unknown>
    updateGoogleAccount?: (input: unknown) => Promise<unknown>
    setCalendarAccess?: (input: unknown) => Promise<unknown>
    refreshCalendars?: (input: unknown) => Promise<unknown>
    unlinkGoogleAccount?: (input: unknown) => Promise<unknown>
    startGoogleLink?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  const googleAccountQueryKey = (id: number) => ["google", "accounts", id]
  mock.module("@/lib/google", () => ({
    GOOGLE_CLIENT_QUERY_KEY: ["google", "client"],
    GOOGLE_ACCOUNTS_QUERY_KEY: ["google", "accounts"],
    googleAccountQueryKey,
    fetchGoogleClient: notStubbed("fetchGoogleClient"),
    fetchGoogleAccounts: notStubbed("fetchGoogleAccounts"),
    fetchGoogleAccount: options.fetchGoogleAccount,
    saveGoogleClientMutationOptions: { mutationFn: notStubbed("saveGoogleClient"), onSuccess: () => {} },
    startGoogleLink: options.startGoogleLink ?? notStubbed("startGoogleLink"),
    updateGoogleAccountMutationOptions: {
      mutationFn: options.updateGoogleAccount ?? notStubbed("updateGoogleAccount"),
      onSuccess: (account: { id: number }) => {
        queryClient.setQueryData(googleAccountQueryKey(account.id), account)
        void queryClient.invalidateQueries({ queryKey: ["google", "accounts"], exact: true })
      },
    },
    setCalendarAccessMutationOptions: {
      mutationFn: options.setCalendarAccess ?? notStubbed("setCalendarAccess"),
      onSuccess: (account: { id: number }) => {
        queryClient.setQueryData(googleAccountQueryKey(account.id), account)
      },
    },
    refreshCalendarsMutationOptions: {
      mutationFn: options.refreshCalendars ?? notStubbed("refreshCalendars"),
      onSuccess: (account: { id: number }) => {
        queryClient.setQueryData(googleAccountQueryKey(account.id), account)
      },
    },
    unlinkGoogleAccountMutationOptions: {
      mutationFn: options.unlinkGoogleAccount ?? notStubbed("unlinkGoogleAccount"),
      onSuccess: (_data: unknown, input: { accountId: number }) => {
        queryClient.removeQueries({ queryKey: googleAccountQueryKey(input.accountId) })
        void queryClient.invalidateQueries({ queryKey: ["google", "accounts"], exact: true })
      },
    },
    LINK_ERROR_MESSAGES,
    linkErrorMessage,
  }))
}

function renderRoute(GoogleAccountRoute: React.ComponentType, queryClient: QueryClient, initialEntry: string) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <Routes>
          <Route path="/google/accounts/:id" element={<GoogleAccountRoute />} />
          <Route path="/google" element={<div>google accounts list</div>} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

test("loads one account and shows its label, address, and status; ?linked=1 shows the linked banner", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work", email: "work@example.com", status: "ok" }),
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7?linked=1")

  expect(await screen.findByText("work@example.com")).toBeTruthy()
  expect(screen.getByText("Linked")).toBeTruthy()
  expect(
    screen.getByText("Linked. Every calendar starts off -- choose what ATLAS may see below."),
  ).toBeTruthy()
})

test("editing the label and saving sends PATCH with only {label}; a 409 shows the server's text verbatim", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const patchCalls: unknown[] = []
  let shouldFail = false
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work" }),
    updateGoogleAccount: async (input) => {
      patchCalls.push(input)
      if (shouldFail) {
        const { ApiError } = await import("@/lib/api")
        throw new ApiError(409, "another account already uses this label")
      }
      return sampleAccount({ id: 7, label: "home" })
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  const labelInput = await screen.findByLabelText("Label")
  fireEvent.change(labelInput, { target: { value: "home" } })
  fireEvent.click(screen.getByRole("button", { name: "Save label" }))

  await waitFor(() => expect(patchCalls).toEqual([{ accountId: 7, label: "home" }]))

  cleanup()

  // Second render: the save fails with a 409, whose text must reach the
  // screen verbatim.
  shouldFail = true
  const queryClient2 = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient2, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work" }),
    updateGoogleAccount: async (input) => {
      patchCalls.push(input)
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(409, "another account already uses this label")
    },
  })
  const { GoogleAccountRoute: GoogleAccountRoute2 } = await import("./GoogleAccountRoute")
  renderRoute(GoogleAccountRoute2, queryClient2, "/google/accounts/7")

  const labelInput2 = await screen.findByLabelText("Label")
  fireEvent.change(labelInput2, { target: { value: "duplicate" } })
  fireEvent.click(screen.getByRole("button", { name: "Save label" }))

  expect(await screen.findByText("another account already uses this label")).toBeTruthy()
})

test("toggling default saves {is_default: true}, clearing it saves {is_default: false}", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const calls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, is_default: false }),
    updateGoogleAccount: async (input) => {
      calls.push(input)
      return sampleAccount({ id: 7, is_default: (input as { is_default: boolean }).is_default })
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  const checkbox = await screen.findByRole("checkbox", { name: "Use for new events when I don't name an account" })
  fireEvent.click(checkbox)
  await waitFor(() => expect(calls).toEqual([{ accountId: 7, is_default: true }]))

  // Wait for the cache-driven re-render to actually show the box checked
  // before clicking again -- otherwise the second click can land on the
  // still-unchecked element and send `is_default: true` a second time.
  const checkedBox = await screen.findByRole("checkbox", {
    name: "Use for new events when I don't name an account",
    checked: true,
  })
  fireEvent.click(checkedBox)
  await waitFor(() =>
    expect(calls).toEqual([
      { accountId: 7, is_default: true },
      { accountId: 7, is_default: false },
    ]),
  )
})

test("choosing a calendar option sends exactly one PUT with that access value; Read and write is disabled with a note when Google marks the calendar read-only", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const puts: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () =>
      sampleAccount({
        id: 7,
        calendars: [
          sampleCalendar({ id: 10, name: "Home", access: "off", can_write: true }),
          sampleCalendar({ id: 11, name: "Work (shared)", access: "off", can_write: false }),
        ],
      }),
    setCalendarAccess: async (input) => {
      puts.push(input)
      return sampleAccount({
        id: 7,
        calendars: [
          sampleCalendar({ id: 10, name: "Home", access: "read_only", can_write: true }),
          sampleCalendar({ id: 11, name: "Work (shared)", access: "off", can_write: false }),
        ],
      })
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  await screen.findByText("Home")
  const homeRow = screen.getByText("Home").closest("div[class]")!.parentElement as HTMLElement
  fireEvent.click(within(homeRow).getByRole("radio", { name: "Read only" }))

  await waitFor(() =>
    expect(puts).toEqual([{ accountId: 7, calendarId: 10, access: "read_only" }]),
  )
  expect(puts.length).toBe(1)

  const sharedRow = screen.getByText("Work (shared)").closest("div[class]")!.parentElement as HTMLElement
  expect((within(sharedRow).getByRole("radio", { name: "Read and write" }) as HTMLInputElement).disabled).toBe(true)
  expect(within(sharedRow).getByText("Google shares this calendar read-only")).toBeTruthy()
})

test("a 503 from a calendar access change shows the server's text verbatim", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, calendars: [sampleCalendar({ id: 10, name: "Home" })] }),
    setCalendarAccess: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(
        503,
        "the setting was saved, but the running google tools could not be updated to match and were stopped -- retry this action to bring them back",
      )
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  await screen.findByText("Home")
  fireEvent.click(screen.getByRole("radio", { name: "Read only" }))

  expect(
    await screen.findByText(
      "the setting was saved, but the running google tools could not be updated to match and were stopped -- retry this action to bring them back",
    ),
  ).toBeTruthy()
})

test("Find new calendars calls the refresh mutation and the list refetches", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const refreshCalls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, calendars: [sampleCalendar({ id: 10, name: "Home" })] }),
    refreshCalendars: async (input) => {
      refreshCalls.push(input)
      return sampleAccount({
        id: 7,
        calendars: [sampleCalendar({ id: 10, name: "Home" }), sampleCalendar({ id: 12, name: "Errands", is_primary: false })],
      })
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  await screen.findByText("Home")
  fireEvent.click(screen.getByRole("button", { name: "Find new calendars" }))

  await waitFor(() => expect(refreshCalls).toEqual([{ accountId: 7 }]))
  expect(await screen.findByText("Errands")).toBeTruthy()
})

test("C-WR-03: 'Link again' is https-gated the same way the primary Link flow is", async () => {
  Object.defineProperty(window, "location", {
    value: { ...window.location, protocol: "http:" },
    writable: true,
    configurable: true,
  })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work", status: "needs_relink" }),
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  expect(await screen.findByText(/Linking needs this page served over https/)).toBeTruthy()
  const relinkButton = (await screen.findByRole("button", { name: "Link again" })) as HTMLButtonElement
  expect(relinkButton.disabled).toBe(true)
})

test("C-WR-02: a failed 'Link again' shows the server's own text verbatim, not a generic message", async () => {
  Object.defineProperty(window, "location", {
    value: { ...window.location, protocol: "https:" },
    writable: true,
    configurable: true,
  })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work", status: "needs_relink" }),
    startGoogleLink: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(503, "No Google OAuth client is configured. Set one up below, then link again.")
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  fireEvent.click(await screen.findByRole("button", { name: "Link again" }))

  expect(
    await screen.findByText("No Google OAuth client is configured. Set one up below, then link again."),
  ).toBeTruthy()
})

test("a needs_relink account shows Link again, starting the flow with this account's label and id", async () => {
  const assignCalls: string[] = []
  Object.defineProperty(window, "location", {
    value: { ...window.location, protocol: "https:", assign: (url: string) => assignCalls.push(url) },
    writable: true,
    configurable: true,
  })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const startCalls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work", status: "needs_relink" }),
    startGoogleLink: async (input) => {
      startCalls.push(input)
      return { authorization_url: "https://accounts.google.com/o/oauth2/v2/auth?relink=1" }
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  fireEvent.click(await screen.findByRole("button", { name: "Link again" }))

  await waitFor(() => expect(startCalls).toEqual([{ label: "work", relink_account_id: 7 }]))
  await waitFor(() => expect(assignCalls).toEqual(["https://accounts.google.com/o/oauth2/v2/auth?relink=1"]))
})

test("Unlink opens a dialog naming the account; Cancel sends nothing, confirming sends DELETE and navigates to /google", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const deleteCalls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work" }),
    unlinkGoogleAccount: async (input) => {
      deleteCalls.push(input)
      return undefined
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  fireEvent.click(await screen.findByRole("button", { name: "Unlink" }))
  expect(await screen.findByText("Unlink work?")).toBeTruthy()
  expect(
    screen.getByText("ATLAS loses access to this account's calendars and mail. Drafts already in Gmail stay there."),
  ).toBeTruthy()

  fireEvent.click(screen.getByRole("button", { name: "Cancel" }))
  expect(deleteCalls).toEqual([])

  fireEvent.click(screen.getByRole("button", { name: "Unlink" }))
  const dialogUnlinkButtons = screen.getAllByRole("button", { name: "Unlink work" })
  fireEvent.click(dialogUnlinkButtons[dialogUnlinkButtons.length - 1])

  await waitFor(() => expect(deleteCalls).toEqual([{ accountId: 7 }]))
  expect(await screen.findByText("google accounts list")).toBeTruthy()
})

test("C-WR-01: the default checkbox is disabled while its own mutation is pending, so a second click can't overlap it", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let resolveUpdate: (value: unknown) => void = () => {}
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, is_default: false }),
    updateGoogleAccount: (input) =>
      new Promise((resolve) => {
        resolveUpdate = () => resolve(sampleAccount({ id: 7, is_default: (input as { is_default: boolean }).is_default }))
      }),
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  const checkbox = (await screen.findByRole("checkbox", {
    name: "Use for new events when I don't name an account",
  })) as HTMLButtonElement
  fireEvent.click(checkbox)

  await waitFor(() => expect(checkbox.disabled).toBe(true))

  resolveUpdate(undefined)
  await waitFor(() => expect(checkbox.disabled).toBe(false))
})

test("C-WR-01: a calendar-access radio group is disabled while its own mutation is pending, so a second click can't overlap it", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let resolveSetAccess: (value: unknown) => void = () => {}
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () =>
      sampleAccount({ id: 7, calendars: [sampleCalendar({ id: 10, name: "Home", access: "off" })] }),
    setCalendarAccess: () =>
      new Promise((resolve) => {
        resolveSetAccess = () =>
          resolve(
            sampleAccount({
              id: 7,
              calendars: [sampleCalendar({ id: 10, name: "Home", access: "read_only" })],
            }),
          )
      }),
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  await screen.findByText("Home")
  fireEvent.click(screen.getByRole("radio", { name: "Read only" }))

  const writeRadio = (await screen.findByRole("radio", { name: "Read and write" })) as HTMLInputElement
  expect(writeRadio.disabled).toBe(true)

  resolveSetAccess(undefined)
  await waitFor(() => expect((screen.getByRole("radio", { name: "Read and write" }) as HTMLInputElement).disabled).toBe(false))
})

test("a failed unlink shows the server's error text verbatim (C-CR-01, R2-WR-09)", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work" }),
    unlinkGoogleAccount: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(500, "the database could not be reached")
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/7")

  fireEvent.click(await screen.findByRole("button", { name: "Unlink" }))
  const dialogUnlinkButtons = screen.getAllByRole("button", { name: "Unlink work" })
  fireEvent.click(dialogUnlinkButtons[dialogUnlinkButtons.length - 1])

  // The error survives `AlertDialogAction` closing the dialog (C-CR-01),
  // and it is the server's own text, not a generic sentence (R2-WR-09).
  expect(await screen.findByText("the database could not be reached")).toBeTruthy()
})

test("R3-WR-02: a 503 from the delete route whose refetch 404s means the account is really gone -- the page leaves it and never offers a retry", async () => {
  // R2-WR-09's own case, told the R3-WR-02 way: the fixed status code (503)
  // is no longer trusted alone -- a refetch of the account is, and it 404s
  // here because the DELETE really did commit before the route's own
  // reconcile-failed 503.
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  let fetchCalls = 0
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => {
      fetchCalls += 1
      if (fetchCalls === 1) return sampleAccount({ id: 7, label: "work" })
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(404, "no google account with id 7")
    },
    unlinkGoogleAccount: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(
        503,
        "the setting was saved, but the running google tools could not be updated to match and were stopped -- retry this action to bring them back",
      )
    },
  })
  queryClient.setQueryData(["google", "accounts"], [sampleAccount({ id: 7, label: "work" })])
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/google/accounts/7"]}>
        <Routes>
          <Route path="/google/accounts/:id" element={<GoogleAccountRoute />} />
          <Route path="/google" element={<NoticeProbe />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  fireEvent.click(await screen.findByRole("button", { name: "Unlink" }))
  const dialogUnlinkButtons = screen.getAllByRole("button", { name: "Unlink work" })
  fireEvent.click(dialogUnlinkButtons[dialogUnlinkButtons.length - 1])

  expect(
    await screen.findByText("Unlinked work. Google tools restart on the next account change."),
  ).toBeTruthy()
  expect(screen.queryByText(/retry this action/)).toBeNull()
  expect(queryClient.getQueryState(["google", "accounts", 7])).toBeUndefined()
  expect(queryClient.getQueryState(["google", "accounts"])?.isInvalidated).toBe(true)
})

test("R3-WR-02: a 503 from a proxy that never reached the server shows an error and stays on the page", async () => {
  // The DELETE never reached the backend (a plain-text 503 from an
  // ingress/reverse proxy with no ready endpoint has no `detail` field at
  // all, unlike the route's own JSON error). The account still exists --
  // the refetch below proves it by succeeding, not 404ing -- so the page
  // must show the error and must never claim the account is gone.
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => sampleAccount({ id: 7, label: "work" }),
    unlinkGoogleAccount: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(503, "The server is unavailable.")
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={["/google/accounts/7"]}>
        <Routes>
          <Route path="/google/accounts/:id" element={<GoogleAccountRoute />} />
          <Route path="/google" element={<NoticeProbe />} />
        </Routes>
      </MemoryRouter>
    </QueryClientProvider>,
  )

  fireEvent.click(await screen.findByRole("button", { name: "Unlink" }))
  const dialogUnlinkButtons = screen.getAllByRole("button", { name: "Unlink work" })
  fireEvent.click(dialogUnlinkButtons[dialogUnlinkButtons.length - 1])

  // The error is shown, and it is the server's own text -- not the
  // "Unlinked..." notice, which would be a false claim here.
  expect(await screen.findByText("The server is unavailable.")).toBeTruthy()
  expect(screen.queryByText(/^Unlinked work\./)).toBeNull()

  // The page never left, and the account's own cache entry is untouched --
  // the next list fetch would still show this account, which is correct,
  // because it is still linked.
  expect(screen.getByText("work")).toBeTruthy()
  expect(queryClient.getQueryState(["google", "accounts", 7])).toBeDefined()
})

function NoticeProbe() {
  const location = useLocation()
  const notice = (location.state as { notice?: unknown } | null)?.notice
  return <div>{typeof notice === "string" ? notice : "google accounts list"}</div>
}

test("an unknown id shows the not-found state with no Retry button", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleAccount: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(404, "no google account with id 999")
    },
  })
  const { GoogleAccountRoute } = await import("./GoogleAccountRoute")

  renderRoute(GoogleAccountRoute, queryClient, "/google/accounts/999")

  await waitFor(() => expect(screen.queryByRole("status")).toBeNull())
  expect(screen.queryByRole("button", { name: "Retry" })).toBeNull()
})
