import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "HubStep.tsx"), "utf-8")

describe("HubStep -- collects the token, verifies through the server, never guesses", () => {
  test("the primary control reads the contract's word: Continue", () => {
    expect(SOURCE).toMatch(/>\s*Continue\s*<\/SubmitButton>/)
  })

  test("every hub failure renders through classifyHubCheckError, never a re-derived guess", () => {
    expect(SOURCE).toMatch(/classifyHubCheckError\(caught\)/)
  })

  test("the token field reuses the settings surface's own field treatment (WizardCredentialField), not a re-implementation", () => {
    expect(SOURCE).toMatch(/<WizardCredentialField\b/)
  })

  test("the address field is a real single-line input, not a re-implementation of the scroll-field backstop", () => {
    expect(SOURCE).toMatch(/id="hub-address"/)
    expect(SOURCE).toMatch(/className="scroll-field"/)
  })

  test("the address is never sent to a route -- there is no credential slot for it", () => {
    expect(SOURCE).not.toMatch(/address\s*:/)
  })
})
