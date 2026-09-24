// Source-text assertions, matching CalibrationRoute.test.ts's own pattern
// (03-06) -- this project has no rendered-component test infrastructure,
// and this page opens a real microphone and starts a real turn, so a
// rendered-DOM click simulation would not be honest here even if the
// infrastructure existed.
import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "DevMicRoute.tsx"), "utf-8")

describe("DevMicRoute -- the two things this plan decides are owed to a development-only page", () => {
  test("a denied microphone permission states what happened, keyed off NotAllowedError specifically", () => {
    expect(SOURCE).toMatch(/caught\.name === "NotAllowedError"/)
    expect(SOURCE).toMatch(/Microphone access was denied/)
  })

  test("a turn that returned no transcript states that too, keyed off the server's own turn_outcome", () => {
    expect(SOURCE).toMatch(/turn_outcome === "empty_transcript"/)
    expect(SOURCE).toMatch(/understood no speech/)
  })

  test("every one of the real nine stage timings renders -- none silently dropped to match a miscounted spec", () => {
    for (const stage of [
      "turn_started_at",
      "stt_socket_open_at",
      "first_partial_at",
      "speech_end_at",
      "stt_final_at",
      "brain_first_round_at",
      "tool_rounds_done_at",
      "first_audio_at",
      "answer_audio_at",
    ]) {
      expect(SOURCE).toContain(stage)
    }
  })

  test("the stage timings render as a stacked label/value list (justify-between rows), never a table", () => {
    expect(SOURCE).not.toMatch(/<table/i)
    expect(SOURCE).toMatch(/items-baseline justify-between/)
  })
})

describe("DevMicRoute -- no second network path", () => {
  test("the toggle goes through transports.ts's typed functions, never a bare fetch/WebSocket construction in this file", () => {
    expect(SOURCE).not.toMatch(/new WebSocket\(/)
    expect(SOURCE).not.toMatch(/(?<![a-zA-Z])fetch\(/)
  })
})
