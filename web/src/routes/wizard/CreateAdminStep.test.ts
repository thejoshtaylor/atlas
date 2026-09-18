// Source-text assertions, matching CalibrationRoute.test.ts/SignInRoute.test.ts's
// own pattern (03-06/03-08) -- this project has no rendered-component test
// infrastructure.
import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "CreateAdminStep.tsx"), "utf-8")

describe("CreateAdminStep -- the only screen a clean install offers", () => {
  test("states plainly that there is no default password and nothing is printed", () => {
    expect(SOURCE).toMatch(/no default password/)
    expect(SOURCE).toMatch(/nothing is ever printed/)
  })

  test("the primary control reads the contract's word: Continue", () => {
    expect(SOURCE).toMatch(/>\s*Continue\s*<\/SubmitButton>/)
  })

  test("a second create-admin attempt renders the server's own message unmodified, never paraphrased", () => {
    expect(SOURCE).toMatch(/caught instanceof Error \? caught\.message/)
  })

  test("the submit control is SubmitButton, the shared re-entrancy guard -- no step can be submitted twice", () => {
    expect(SOURCE).toMatch(/<SubmitButton\b/)
  })

  test("blank fields never reach the mutation", () => {
    const blankCheckIndex = SOURCE.indexOf("!email.trim() || !displayName.trim() || !password")
    const mutateCallIndex = SOURCE.indexOf("createAdmin.mutateAsync")
    expect(blankCheckIndex).toBeGreaterThan(-1)
    expect(mutateCallIndex).toBeGreaterThan(-1)
    expect(blankCheckIndex).toBeLessThan(mutateCallIndex)
  })
})
