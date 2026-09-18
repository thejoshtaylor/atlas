import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "AudioSourceStep.tsx"), "utf-8")

describe("AudioSourceStep -- carries the Needs-restart badge, never implies a live effect it lacks", () => {
  test("the badge reads from the server's own applies_live field, not a hardcoded assumption", () => {
    expect(SOURCE).toMatch(/detail\?\.applies_live/)
    expect(SOURCE).toMatch(/appliesLive \? "Live" : "Needs restart"/)
  })

  test("the primary control reads the contract's word: Continue", () => {
    expect(SOURCE).toMatch(/>\s*Continue\s*<\/SubmitButton>/)
  })

  test("writes through PUT /api/wizard/audio-source via the shared lib, not a second path", () => {
    expect(SOURCE).toMatch(/setAudioSourceMutationOptions/)
  })
})
