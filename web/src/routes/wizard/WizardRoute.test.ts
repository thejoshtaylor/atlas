import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "WizardRoute.tsx"), "utf-8")

describe("WizardRoute -- routes to the first unfinished step, never step one for a returning admin", () => {
  test("admin_account routes with no session check -- there is no session to have yet", () => {
    const adminAccountBranch = SOURCE.indexOf('nextStep === "admin_account"')
    const sessionCheck = SOURCE.indexOf("session.isLoading")
    expect(adminAccountBranch).toBeGreaterThan(-1)
    expect(sessionCheck).toBeGreaterThan(-1)
    expect(adminAccountBranch).toBeLessThan(sessionCheck)
  })

  test("every other step redirects to sign-in with no session, carrying /setup as the return address", () => {
    expect(SOURCE).toMatch(/Navigate to="\/sign-in" replace state=\{\{ from: \{ pathname: location\.pathname \} \}\}/)
  })

  test("the routing decision itself is the pure firstUnfinishedStep function, not re-derived inline", () => {
    expect(SOURCE).toMatch(/firstUnfinishedStep\(status\.data/)
  })
})
