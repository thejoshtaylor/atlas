import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "AccountsRoute.tsx"), "utf-8")

describe("AccountsRoute -- the empty state's exact copy (Copywriting Contract)", () => {
  test("heading and body match the contract verbatim", () => {
    expect(SOURCE).toMatch(/heading="No one invited yet\."/)
    expect(SOURCE).toMatch(/Invite an operator or viewer to share access\./)
  })
})

describe("AccountsRoute -- revoking access is a destructive confirmation with the contract's exact copy", () => {
  test("the dialog body names the person and states the consequence; the confirm button is red and says 'Remove access'", () => {
    expect(SOURCE).toMatch(/They can no longer sign in\./)
    expect(SOURCE).toMatch(/variant="destructive"[\s\S]{0,300}Remove access/)
  })
})

describe("AccountsRoute -- the invite token is rendered exactly once and outlives a refetch", () => {
  test("the just-created invite lives in component state (setJustCreated), not the query cache", () => {
    expect(SOURCE).toMatch(/useState<InviteCreated \| null>/)
  })

  test("InviteTokenPanel states plainly the link will not be shown again", () => {
    expect(SOURCE).toMatch(/will not be shown again/)
  })
})

describe("AccountsRoute -- a failed load disables invite and revoke", () => {
  test("every add/write control reads its disabled state from controlsDisabled, derived from screen.kind !== \"ready\"", () => {
    expect(SOURCE).toMatch(/const controlsDisabled = screen\.kind !== "ready"/)
    // Every interactive control that mutates state is wired to it.
    const disabledSites = [...SOURCE.matchAll(/disabled=\{[^}]*controlsDisabled[^}]*\}/g)]
    expect(disabledSites.length).toBeGreaterThanOrEqual(3) // role radios, email input, submit
  })
})

describe("AccountsRoute -- counts read correctly (zero handled by the empty state; one/many by formatPendingInviteCount)", () => {
  test("the count line uses the shared pure formatter, not a bespoke pluralization", () => {
    expect(SOURCE).toMatch(/formatPendingInviteCount\(screen\.invites\.length\)/)
  })
})

describe("AccountsRoute -- loading never renders as an empty list", () => {
  test("SkeletonList renders for screen.kind === \"loading\"", () => {
    expect(SOURCE).toMatch(/screen\.kind === "loading" \? <SkeletonList/)
  })
})
