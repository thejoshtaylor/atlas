import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "ProviderSetStep.tsx"), "utf-8")

describe("ProviderSetStep -- a blank slot is named on the step, never a silently disabled control", () => {
  test("Continue is never disabled outright -- it re-checks and names what is still missing", () => {
    expect(SOURCE).not.toMatch(/<SubmitButton[^>]*disabled/)
  })

  test("the missing slots render as named text from the server's own detail.missing list", () => {
    expect(SOURCE).toMatch(/providerStep\?\.detail\?\.missing/)
    expect(SOURCE).toMatch(/Still needed/)
  })

  test("a slot's field reuses the settings surface's own treatment (WizardCredentialField)", () => {
    expect(SOURCE).toMatch(/<WizardCredentialField\b/)
  })

  test("covers exactly the three AI-provider slots, never the Home Assistant slot (the hub step's own concern)", () => {
    expect(SOURCE).toMatch(/"stt_api_key", "brain_api_key", "tts_api_key"/)
    expect(SOURCE).not.toMatch(/ha_token/)
  })

  test("the primary control reads the contract's word: Continue", () => {
    expect(SOURCE).toMatch(/>\s*Continue\s*<\/SubmitButton>/)
  })
})
