// D-17 backfill part B (08-10-PLAN.md), Task 3 -- deliberately the LAST
// file this milestone adds. The shell's navigation list grows by three
// entries across plans 08-04, 08-06 and 08-08; a test written before
// those landed would have had to be rewritten three times.
//
// This file never asserts a total entry count. A count is a test that
// breaks every time a future phase adds a screen, for no gain -- D-17's
// backfill is supposed to leave later phases easier to work in, not
// harder. Every entry is named and checked individually instead, by its
// accessible name, so a phase that adds a fourteenth nav entry never has
// to touch this file.
//
// This file checks the CONVENIENCE layer only, not the access boundary.
// `AppShell.tsx`'s own comment already states it: hiding a link from a
// role is convenience, never a permission boundary -- the route
// (`RequireRole`) and the backend (`require_role`) both refuse a role
// that cannot use a screen regardless of what this navigation shows or
// hides. A future reader finding a role-keyed test in the navigation
// file is exactly the person likely to mistake it for a boundary check;
// `tests/test_auth_roles.py` and the route guards are the tests that
// actually hold that line.
//
// `mock.module` replaces `@/hooks/useSession` before `AppShell` is
// imported -- a dynamic `import()` inside the test body is what makes
// that ordering hold (`SessionsRoute.test.tsx`'s established pattern).
// Every `mock.module` call returns the full named-export set
// `@/hooks/useSession` carries.
import { afterEach, describe, expect, mock, test } from "bun:test"
import { cleanup, fireEvent, render, screen, within } from "@testing-library/react"
import { MemoryRouter, Route, Routes } from "react-router-dom"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

const OPERATOR_LABELS = ["Safety policy", "Macros", "Scheduled", "Developer mic", "Live", "Sessions", "Wake threshold"]
const ADMIN_LABELS = ["Accounts", "Settings", "Plugins", "Providers", "Google accounts"]

function stubSession(role: "viewer" | "operator" | "admin") {
  mock.module("@/hooks/useSession", () => ({
    useSession: () => ({
      data: { id: 1, email: "operator@example.com", display_name: "Operator", role },
    }),
    SetupIncompleteError: class SetupIncompleteError extends Error {},
  }))
}

function renderShell(AppShell: React.ComponentType) {
  return render(
    <MemoryRouter initialEntries={["/"]}>
      <Routes>
        <Route element={<AppShell />}>
          <Route index element={<div>screen content</div>} />
        </Route>
      </Routes>
    </MemoryRouter>,
  )
}

function openNav() {
  fireEvent.click(screen.getByRole("button", { name: "Open navigation" }))
}

describe("AppShell -- the navigation toggle", () => {
  test("opens the navigation and a second activation closes it", async () => {
    stubSession("viewer")
    const { AppShell } = await import("./AppShell")
    renderShell(AppShell)

    expect(screen.queryByRole("navigation", { name: "Primary" })).toBeNull()

    openNav()
    expect(screen.getByRole("navigation", { name: "Primary" })).toBeTruthy()

    fireEvent.click(screen.getByRole("button", { name: "Close navigation" }))
    expect(screen.queryByRole("navigation", { name: "Primary" })).toBeNull()
  })
})

describe("AppShell -- role-visible entries, named individually, never counted", () => {
  test("a viewer session sees the home entry and no operator-only or admin-only entry", async () => {
    stubSession("viewer")
    const { AppShell } = await import("./AppShell")
    renderShell(AppShell)
    openNav()
    const nav = screen.getByRole("navigation", { name: "Primary" })

    expect(within(nav).getByRole("link", { name: "Home" })).toBeTruthy()
    for (const label of [...OPERATOR_LABELS, ...ADMIN_LABELS]) {
      expect(within(nav).queryByRole("link", { name: label })).toBeNull()
    }
  })

  test("an operator session sees every operator entry and no admin entry", async () => {
    stubSession("operator")
    const { AppShell } = await import("./AppShell")
    renderShell(AppShell)
    openNav()
    const nav = screen.getByRole("navigation", { name: "Primary" })

    expect(within(nav).getByRole("link", { name: "Home" })).toBeTruthy()
    for (const label of OPERATOR_LABELS) {
      expect(within(nav).getByRole("link", { name: label })).toBeTruthy()
    }
    for (const label of ADMIN_LABELS) {
      expect(within(nav).queryByRole("link", { name: label })).toBeNull()
    }
  })

  test("an admin session sees every entry an operator sees, plus the four admin entries", async () => {
    stubSession("admin")
    const { AppShell } = await import("./AppShell")
    renderShell(AppShell)
    openNav()
    const nav = screen.getByRole("navigation", { name: "Primary" })

    expect(within(nav).getByRole("link", { name: "Home" })).toBeTruthy()
    for (const label of [...OPERATOR_LABELS, ...ADMIN_LABELS]) {
      expect(within(nav).getByRole("link", { name: label })).toBeTruthy()
    }
  })
})

describe("AppShell -- every navigation entry is a link with an accessible name", () => {
  test("each link rendered for an admin session has a non-empty accessible name", async () => {
    stubSession("admin")
    const { AppShell } = await import("./AppShell")
    renderShell(AppShell)
    openNav()
    const nav = screen.getByRole("navigation", { name: "Primary" })

    const links = within(nav).getAllByRole("link")
    for (const link of links) {
      expect(link.textContent?.trim().length).toBeGreaterThan(0)
    }
  })
})
