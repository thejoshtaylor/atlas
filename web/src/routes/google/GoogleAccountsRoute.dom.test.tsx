// GOOG-01, GOOG-02 (09-10-PLAN.md Task 1). `mock.module` replaces
// `@/lib/google` before `GoogleAccountsRoute` is imported -- a dynamic
// `import()` inside the test body is what makes that ordering hold
// (`PluginsRoute.dom.test.tsx`'s established pattern). `LINK_ERROR_MESSAGES`
// and `linkErrorMessage` are imported statically, before any `mock.module`
// call runs, and re-exported through the mock factory verbatim -- so these
// tests assert the real fixed sentences rather than a second, hand-copied
// literal that could silently drift from the real ones.
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"

import { LINK_ERROR_MESSAGES, linkErrorMessage } from "@/lib/google"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function stubLocation(overrides: { protocol?: string; origin?: string; assign?: (url: string) => void } = {}) {
  Object.defineProperty(window, "location", {
    value: {
      ...window.location,
      protocol: overrides.protocol ?? "https:",
      origin: overrides.origin ?? "https://atlas.example",
      assign: overrides.assign ?? (() => {}),
    },
    writable: true,
    configurable: true,
  })
}

function sampleClient(overrides: Record<string, unknown> = {}) {
  return {
    configured: false,
    client_id: null,
    updated_at: null,
    redirect_path: "/api/google/oauth/callback",
    ...overrides,
  }
}

function sampleAccount(overrides: Record<string, unknown> = {}) {
  return {
    id: 1,
    label: "work",
    email: "operator@example.com",
    is_default: false,
    status: "ok",
    status_detail: null,
    refresh_token_expires_at: null,
    linked_at: "2026-09-01T00:00:00Z",
    calendars: [],
    plugin_state: null,
    ...overrides,
  }
}

function stubGoogle(
  queryClient: QueryClient,
  options: {
    fetchGoogleClient: () => Promise<unknown>
    fetchGoogleAccounts: () => Promise<unknown[]>
    saveGoogleClient?: (input: unknown) => Promise<unknown>
    startGoogleLink?: (input: unknown) => Promise<unknown>
  },
) {
  const notStubbed = (name: string) => async () => {
    throw new Error(`${name} is not stubbed in this test`)
  }
  mock.module("@/lib/google", () => ({
    GOOGLE_CLIENT_QUERY_KEY: ["google", "client"],
    GOOGLE_ACCOUNTS_QUERY_KEY: ["google", "accounts"],
    googleAccountQueryKey: (id: number) => ["google", "accounts", id],
    fetchGoogleClient: options.fetchGoogleClient,
    fetchGoogleAccounts: options.fetchGoogleAccounts,
    fetchGoogleAccount: notStubbed("fetchGoogleAccount"),
    saveGoogleClientMutationOptions: {
      mutationFn: options.saveGoogleClient ?? notStubbed("saveGoogleClient"),
      onSuccess: (client: unknown) => {
        queryClient.setQueryData(["google", "client"], client)
      },
    },
    startGoogleLink: options.startGoogleLink ?? notStubbed("startGoogleLink"),
    updateGoogleAccountMutationOptions: { mutationFn: notStubbed("updateGoogleAccount"), onSuccess: () => {} },
    setCalendarAccessMutationOptions: { mutationFn: notStubbed("setCalendarAccess"), onSuccess: () => {} },
    refreshCalendarsMutationOptions: { mutationFn: notStubbed("refreshCalendars"), onSuccess: () => {} },
    unlinkGoogleAccountMutationOptions: { mutationFn: notStubbed("unlinkGoogleAccount"), onSuccess: () => {} },
    LINK_ERROR_MESSAGES,
    linkErrorMessage,
  }))
}

function renderRoute(
  GoogleAccountsRoute: React.ComponentType,
  queryClient: QueryClient,
  initialEntry: string | { pathname: string; state: unknown } = "/google",
) {
  return render(
    <QueryClientProvider client={queryClient}>
      <MemoryRouter initialEntries={[initialEntry]}>
        <GoogleAccountsRoute />
      </MemoryRouter>
    </QueryClientProvider>,
  )
}

test("with no client configured, shows the setup steps including the In production sentence and the redirect URI", async () => {
  stubLocation({ protocol: "https:", origin: "https://atlas.example" })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: false, redirect_path: "/api/google/oauth/callback" }),
    fetchGoogleAccounts: async () => [],
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  expect(
    await screen.findByText(
      "Set the app's publishing status to In production. While it is in Testing, Google ends every link after 7 days, and every account would unlink once a week.",
    ),
  ).toBeTruthy()
  expect(screen.getByText("https://atlas.example/api/google/oauth/callback")).toBeTruthy()
})

test("saving a client id and secret calls the save mutation once, then shows the client id and Secret saved with no input holding the secret", async () => {
  stubLocation()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const saveCalls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: false }),
    fetchGoogleAccounts: async () => [],
    saveGoogleClient: async (input) => {
      saveCalls.push(input)
      return sampleClient({ configured: true, client_id: "my-client-id", updated_at: "2026-09-24T00:00:00Z" })
    },
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  await screen.findByLabelText("Client ID")
  fireEvent.change(screen.getByLabelText("Client ID"), { target: { value: "my-client-id" } })
  fireEvent.change(screen.getByLabelText("Client secret"), { target: { value: "shh-pw" } })
  fireEvent.click(screen.getByRole("button", { name: "Save client" }))

  await waitFor(() => expect(saveCalls).toEqual([{ client_id: "my-client-id", client_secret: "shh-pw" }]))
  expect(await screen.findByText("Secret saved")).toBeTruthy()
  expect(screen.getByText("my-client-id")).toBeTruthy()
  expect(screen.queryByLabelText("Client secret")).toBeNull()
  expect(screen.queryByDisplayValue("shh-pw")).toBeNull()
})

test("C-WR-02: a failed client save shows the server's own text verbatim, not a generic message", async () => {
  stubLocation()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: false }),
    fetchGoogleAccounts: async () => [],
    saveGoogleClient: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(400, "client_id and client_secret must both be non-empty")
    },
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  await screen.findByLabelText("Client ID")
  fireEvent.change(screen.getByLabelText("Client ID"), { target: { value: "my-client-id" } })
  fireEvent.change(screen.getByLabelText("Client secret"), { target: { value: "shh-pw" } })
  fireEvent.click(screen.getByRole("button", { name: "Save client" }))

  expect(await screen.findByText("client_id and client_secret must both be non-empty")).toBeTruthy()
})

test("C-WR-02: a failed Link start shows the server's own text verbatim, not a generic message", async () => {
  stubLocation({ protocol: "https:", origin: "https://atlas.example" })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "abc" }),
    fetchGoogleAccounts: async () => [],
    startGoogleLink: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(409, "Another account already uses that label. Choose a different one.")
    },
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  await screen.findByRole("button", { name: "Link" })
  fireEvent.change(screen.getByLabelText("Label"), { target: { value: "work" } })
  fireEvent.click(screen.getByRole("button", { name: "Link" }))

  expect(
    await screen.findByText("Another account already uses that label. Choose a different one."),
  ).toBeTruthy()
})

test("the Link button is disabled on an http: page and enabled on an https: page", async () => {
  stubLocation({ protocol: "http:", origin: "http://atlas.example" })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "abc" }),
    fetchGoogleAccounts: async () => [],
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  expect(await screen.findByText(/Linking needs this page served over https/)).toBeTruthy()
  fireEvent.change(screen.getByLabelText("Label"), { target: { value: "work" } })
  expect((screen.getByRole("button", { name: "Link" }) as HTMLButtonElement).disabled).toBe(true)
})

test("entering label work and pressing Link calls the start mutation, then navigates to the authorization URL", async () => {
  const assignCalls: string[] = []
  stubLocation({ protocol: "https:", origin: "https://atlas.example", assign: (url) => assignCalls.push(url) })
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const startCalls: unknown[] = []
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "abc" }),
    fetchGoogleAccounts: async () => [],
    startGoogleLink: async (input) => {
      startCalls.push(input)
      return { authorization_url: "https://accounts.google.com/o/oauth2/v2/auth?x=1" }
    },
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  await screen.findByRole("button", { name: "Link" })
  fireEvent.change(screen.getByLabelText("Label"), { target: { value: "work" } })
  fireEvent.click(screen.getByRole("button", { name: "Link" }))

  await waitFor(() => expect(startCalls).toEqual([{ label: "work", relink_account_id: null }]))
  await waitFor(() => expect(assignCalls).toEqual(["https://accounts.google.com/o/oauth2/v2/auth?x=1"]))
})

test("link_error=scopes_missing shows its own named sentence, verbatim", async () => {
  stubLocation()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "abc" }),
    fetchGoogleAccounts: async () => [],
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient, "/google?link_error=scopes_missing")

  expect(await screen.findByText(LINK_ERROR_MESSAGES.scopes_missing)).toBeTruthy()
})

test("an unknown link_error code shows a generic sentence, never the raw code alone", async () => {
  stubLocation()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "abc" }),
    fetchGoogleAccounts: async () => [],
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient, "/google?link_error=totally_bogus")

  await screen.findByText("Google accounts")
  expect(screen.queryByText("totally_bogus")).toBeNull()
  expect(screen.getByText(linkErrorMessage("totally_bogus")!)).toBeTruthy()
})

test("each account card shows its label, address, badges, link-expires date, and links to its detail page", async () => {
  stubLocation()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "abc" }),
    fetchGoogleAccounts: async () => [
      sampleAccount({ id: 1, label: "work", email: "work@example.com", is_default: true, status: "ok" }),
      sampleAccount({ id: 2, label: "personal", email: "personal@example.com", status: "needs_relink" }),
      sampleAccount({ id: 3, label: "spare", email: "spare@example.com", status: "unreachable" }),
      sampleAccount({
        id: 4,
        label: "expiring",
        email: "expiring@example.com",
        refresh_token_expires_at: "2026-10-01T00:00:00Z",
      }),
    ],
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  expect(await screen.findByText("work")).toBeTruthy()
  expect(screen.getByText("work@example.com")).toBeTruthy()
  expect(screen.getByText("Default")).toBeTruthy()
  expect(screen.getByText("Needs re-link")).toBeTruthy()
  expect(screen.getByText("Unreachable")).toBeTruthy()
  expect(screen.getByText(/^Link expires/)).toBeTruthy()

  const workLink = screen.getByText("work").closest("a")
  expect(workLink?.getAttribute("href")).toBe("/google/accounts/1")
})

// Orchestrator-directed correction (09-UI-SPEC.md's Read This First /
// UI Considerations table, ⚠ unresolved row): zero linked accounts must
// show `EmptyState`, matching this project's own Macros/Sessions/
// Wake-Tuning precedent -- not a silently empty `<ul>`.
test("with a configured client and zero linked accounts, shows the empty state instead of a bare list", async () => {
  stubLocation()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "abc" }),
    fetchGoogleAccounts: async () => [],
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient)

  expect(await screen.findByText("No accounts linked yet.")).toBeTruthy()
  expect(
    screen.getByText("Set up your OAuth client above, then link your first Google account."),
  ).toBeTruthy()
})

test("R2-WR-09: a notice handed over by the account page's unlink is shown above the list", async () => {
  stubLocation()
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubGoogle(queryClient, {
    fetchGoogleClient: async () => sampleClient({ configured: true, client_id: "my-client-id" }),
    fetchGoogleAccounts: async () => [],
  })
  const { GoogleAccountsRoute } = await import("./GoogleAccountsRoute")

  renderRoute(GoogleAccountsRoute, queryClient, {
    pathname: "/google",
    state: { notice: "Unlinked work. The Google tools stopped and did not start again." },
  })

  expect(await screen.findByText("Unlinked work. The Google tools stopped and did not start again.")).toBeTruthy()
})
