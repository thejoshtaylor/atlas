import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "AcceptInviteRoute.tsx"), "utf-8")

describe("AcceptInviteRoute -- no role field, ever (T-03-31)", () => {
  test("no useState hook manages a role value, and the accept call site names no role field", () => {
    expect(SOURCE).not.toMatch(/useState.*[Rr]ole/)
    const callSite = SOURCE.match(/accept\.mutateAsync\(\{[^}]*\}\)/)
    expect(callSite).not.toBeNull()
    expect(callSite![0]).not.toMatch(/role/)
  })
})

describe("AcceptInviteRoute -- accepting does not sign the new account in", () => {
  test("a successful accept navigates to /sign-in, not into the authenticated shell", () => {
    const acceptIndex = SOURCE.indexOf("accept.mutateAsync")
    const navigateIndex = SOURCE.indexOf('navigate("/sign-in"')
    expect(acceptIndex).toBeGreaterThan(-1)
    expect(navigateIndex).toBeGreaterThan(-1)
    expect(acceptIndex).toBeLessThan(navigateIndex)
  })
})

describe("AcceptInviteRoute -- blank fields never reach the server", () => {
  test("handleSubmit returns before calling accept.mutateAsync when a field is blank", () => {
    const blankCheckIndex = SOURCE.indexOf("!displayName.trim() || !password")
    const mutateCallIndex = SOURCE.indexOf("accept.mutateAsync")
    expect(blankCheckIndex).toBeGreaterThan(-1)
    expect(blankCheckIndex).toBeLessThan(mutateCallIndex)
  })
})
