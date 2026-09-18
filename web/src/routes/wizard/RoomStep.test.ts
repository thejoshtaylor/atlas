import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "RoomStep.tsx"), "utf-8")

describe("RoomStep -- mounts plan 03-06's calibration screen, never a second implementation", () => {
  test("mounts CalibrationRoute rather than re-implementing the three-result display", () => {
    expect(SOURCE).toMatch(/<CalibrationRoute\b/)
    expect(SOURCE).not.toMatch(/CalibrationResults/)
  })

  test("onFinish is CalibrationRoute's own gate -- there is no separate Finish control this step invents", () => {
    expect(SOURCE).toMatch(/onFinish=\{.*handleFinish/)
  })

  test("the finish refusal renders the server's detail unmodified -- every outstanding step name, not just the first", () => {
    expect(SOURCE).toMatch(/detail=\{finish\.error instanceof Error \? finish\.error\.message/)
  })

  test("finishes through finishWizardMutationOptions, the one lib function -- never a hand-rolled fetch", () => {
    expect(SOURCE).toMatch(/finishWizardMutationOptions/)
    expect(SOURCE).not.toMatch(/(?<![a-zA-Z])fetch\(/)
  })
})
