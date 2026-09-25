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

describe("AudioSourceStep -- D-15: a second source, camera and edge", () => {
  test("offers both camera and edge as radio values", () => {
    expect(SOURCE).toMatch(/RadioGroupItem value=\{CAMERA_SOURCE\}/)
    expect(SOURCE).toMatch(/RadioGroupItem value=\{EDGE_SOURCE\}/)
    expect(SOURCE).toMatch(/CAMERA_SOURCE: AudioSource = "camera"/)
    expect(SOURCE).toMatch(/EDGE_SOURCE: AudioSource = "edge"/)
  })

  test("starts on the status query's own stored source, falling back to camera", () => {
    expect(SOURCE).toMatch(/step\.detail\?\.source/)
    expect(SOURCE).toMatch(/stored === EDGE_SOURCE \? EDGE_SOURCE : CAMERA_SOURCE/)
  })

  test("Continue sends the selected source, not a hardcoded one", () => {
    expect(SOURCE).toMatch(/setSource\.mutateAsync\(\{\s*source\s*\}\)/)
  })

  test("the edge option carries its own pairing note", () => {
    expect(SOURCE).toMatch(/Pair the Pi under Edge devices first\. The change applies after a restart\./)
  })
})
