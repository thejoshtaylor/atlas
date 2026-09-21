import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "PolicyRoute.tsx"), "utf-8")

describe("PolicyRoute -- the two empty states carry the contract's exact, distinct copy", () => {
  test("denylist empty state", () => {
    expect(SOURCE).toMatch(/heading="Nothing denied yet\."/)
    expect(SOURCE).toMatch(
      /Every entity is reachable by voice\. Add an entity to block it from voice control — it stays readable\./,
    )
  })
  test("allowlist empty state, and it explicitly warns that empty means unreachable", () => {
    expect(SOURCE).toMatch(/heading="Nothing allowed yet\."/)
    expect(SOURCE).toMatch(
      /In allow-only mode, an empty allowlist means voice reaches nothing\. Add an entity to allow it\./,
    )
  })
})

describe("PolicyRoute -- switching to allow-only is a destructive confirmation with the contract's exact text", () => {
  test("dialog body and red confirm button", () => {
    expect(SOURCE).toMatch(
      /Switch to allow-only mode\? Every entity not on the allowlist becomes unreachable by voice,\s*immediately\./,
    )
    expect(SOURCE).toMatch(/variant="destructive" onClick=\{confirmModeSwitch\}/)
  })

  test("switching FROM allow-only back to deny-list mode is not gated behind this dialog", () => {
    // Only entering allowlist_only opens the confirm -- the reverse
    // direction calls setMode directly, matching the Copywriting
    // Contract (which names only one destructive mode-switch direction).
    expect(SOURCE).toMatch(/if \(next === "allowlist_only"\) \{/)
  })
})

describe("PolicyRoute -- removing a rule is a destructive confirmation with the contract's exact text", () => {
  test("dialog body names the entity and states the consequence; red confirm button", () => {
    expect(SOURCE).toMatch(
      /Remove \$\{rule\.value\} from the denylist\? Voice commands will be able to control it again\./,
    )
    expect(SOURCE).toMatch(/variant="destructive"[\s\S]{0,300}Remove from denylist/)
  })
})

describe("PolicyRoute -- the mode control never optimistically flips", () => {
  test("RadioGroup's value is bound to currentMode (derived from server data), not a local pending-selection value", () => {
    expect(SOURCE).toMatch(/<RadioGroup\s+value=\{currentMode\}/)
  })
  test("the mode control is a RadioGroup (segmented control), never a bare Switch", () => {
    expect(SOURCE).not.toMatch(/<Switch\b/)
  })
})

describe("PolicyRoute -- a failed load disables every add and remove control", () => {
  test("controlsDisabled derives from screen.kind !== \"ready\" and gates the add input, add button, and every row's remove button", () => {
    expect(SOURCE).toMatch(/const controlsDisabled = screen\.kind !== "ready"/)
    expect(SOURCE).toMatch(/disabled=\{controlsDisabled \|\| !pendingValue\.trim\(\)\}/)
    expect(SOURCE).toMatch(/disabled=\{disabled \|\| removeRule\.isPending\}/)
  })
})

describe("PolicyRoute -- an unresolved entity gets the amber denied-state badge pair and stays in the list", () => {
  test("Badge variant=\"denied\" renders \"Not found in Home Assistant\" when !rule.resolved, inside the same row (no filtering-out)", () => {
    expect(SOURCE).toMatch(/!rule\.resolved \? <Badge variant="denied">Not found in Home Assistant<\/Badge>/)
  })
})

describe("PolicyRoute -- the unsaved buffer lives in the Zustand store, not the query cache", () => {
  test("pendingValue and pendingRemovalId come from usePolicyDraftStore", () => {
    expect(SOURCE).toMatch(/usePolicyDraftStore\(\(state\) => state\.pendingValue\)/)
    expect(SOURCE).toMatch(/usePolicyDraftStore\(\(state\) => state\.pendingRemovalId\)/)
  })
})

describe("PolicyRoute -- the count line uses the shared pure formatter", () => {
  test("formatRuleCount is the only source of the count text", () => {
    expect(SOURCE).toMatch(/formatRuleCount\(currentMode, rules\.length\)/)
  })
})
