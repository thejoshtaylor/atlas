// Source-text assertions, matching CalibrationRoute.test.ts's own
// pattern (03-06) -- this project has no rendered-component test
// infrastructure. What a DOM render would need is the held-out
// `<human-check>` this plan's own Task 3 names.
import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "SignInRoute.tsx"), "utf-8")

describe("SignInRoute -- blank fields never reach the server", () => {
  test("handleSubmit returns before calling login.mutateAsync when a field is blank", () => {
    const blankCheckIndex = SOURCE.indexOf("!email.trim() || !password")
    const mutateCallIndex = SOURCE.indexOf("login.mutateAsync")
    expect(blankCheckIndex).toBeGreaterThan(-1)
    expect(mutateCallIndex).toBeGreaterThan(-1)
    expect(blankCheckIndex).toBeLessThan(mutateCallIndex)
  })
})

describe("SignInRoute -- a second submit cannot be issued while the first is in flight", () => {
  test("the form's submit control is SubmitButton, the shared re-entrancy guard", () => {
    expect(SOURCE).toMatch(/<SubmitButton\b/)
  })
})

describe("SignInRoute -- signing in never distinguishes an unknown email from a wrong password", () => {
  test("every server-side login failure routes through classifyLoginError, the one function that collapses them", () => {
    expect(SOURCE).toMatch(/classifyLoginError\(caught\)/)
    // No second, ad-hoc 401 branch reintroducing a distinction
    // classifyLoginError already removed.
    expect(SOURCE).not.toMatch(/caught\.status === 401/)
  })
})
