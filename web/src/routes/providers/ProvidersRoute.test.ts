import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

// `web/` has no rendered-DOM test infrastructure (no Playwright, no
// @testing-library/*) -- this file asserts the source text carries the
// wiring 07-UI-SPEC.md requires. It cannot prove the page actually
// renders correctly in a browser, the same limitation every prior
// phase's screen test in this repository carries (five phases now).
const SOURCE = readFileSync(join(import.meta.dir, "ProvidersRoute.tsx"), "utf-8")

describe("ProvidersRoute -- loading never renders as an empty page", () => {
  test("SkeletonList renders for screen.kind === \"loading\"", () => {
    expect(SOURCE).toMatch(/screen\.kind === "loading" \? <SkeletonList/)
  })
})

describe("ProvidersRoute -- a failed load", () => {
  test("ErrorState renders with a retry for screen.kind === \"error\"", () => {
    expect(SOURCE).toMatch(/screen\.kind === "error" \? \(/)
    expect(SOURCE).toMatch(/onRetry=\{\(\) => void query\.refetch\(\)\}/)
  })
})

describe("ProvidersRoute -- no empty state at all (07-UI-SPEC.md's Copywriting Contract)", () => {
  test("no EmptyState import or usage -- D-03's boot-time registry validation guarantees at least one option", () => {
    expect(SOURCE).not.toMatch(/EmptyState/)
  })
})

describe("ProvidersRoute -- the page title and the primary CTA copy", () => {
  test("Display-size page title reads Providers", () => {
    expect(SOURCE).toMatch(/<h1 className="text-display font-semibold">Providers<\/h1>/)
  })

  // "the save button reads the Copywriting Contract's exact text" and
  // "save success and failure copy match the Copywriting Contract
  // verbatim" retired (08-10-PLAN.md Task 1, D-17 backfill part B):
  // strictly superseded by ProvidersRoute.dom.test.tsx's
  // "choosing a different provider ... calls the mutation" /
  // "a successful save renders the restart-required message" / "a
  // failed save leaves the chosen value on screen" tests, which drive a
  // real save and assert the exact runtime-rendered button label and
  // success/failure copy, not just its presence in source text.
})

describe("ProvidersRoute -- one card per slot, driven entirely by the server response", () => {
  test("cards render from screen.slots.map, not a hardcoded slot list", () => {
    expect(SOURCE).toMatch(/screen\.slots\.map/)
  })

  // "each slot renders its own RadioGroup bound to the draft selection"
  // retired (08-10-PLAN.md Task 1): strictly superseded by
  // ProvidersRoute.dom.test.tsx's "each slot renders its currently
  // stored provider ..." and "choosing a different provider ..." tests,
  // which drive the RadioGroup by its accessible name and observe the
  // draft-bound selection and the save payload it produces.

  test("every option row carries touch-target", () => {
    expect(SOURCE).toMatch(/touch-target/)
  })
})

describe("ProvidersRoute -- badges are derived, never computed inline", () => {
  test("needs-restart, degraded, and wrapped badges all come from the derive module", () => {
    expect(SOURCE).toMatch(/needsRestartBadge\(slot\)/)
    expect(SOURCE).toMatch(/degradedBadge\(slot\)/)
    expect(SOURCE).toMatch(/wrappedBadge\(option\.wrapped, isActive \? slot\.measured_ms : null\)/)
  })
})

describe("ProvidersRoute -- the draft is never reverted on a failed save", () => {
  // "the draft state is only ever set from the loaded selection once per
  // slot, never reset on save failure" retired (08-10-PLAN.md Task 1):
  // strictly superseded by ProvidersRoute.dom.test.tsx's "a failed save
  // leaves the chosen value on screen rather than reverting it" test,
  // which drives a real failed save and asserts the chosen radio is
  // still checked afterward -- the runtime outcome the source-text
  // check could only approximate.

  test("the settings draft is also left untouched on a failed save", () => {
    const catchBlock = SOURCE.slice(SOURCE.indexOf("} catch {"), SOURCE.indexOf("Couldn't save your provider choices"))
    expect(catchBlock).not.toMatch(/setSettingsDraft/)
  })
})

describe("ProvidersRoute -- one save carries all three slots' settings in one request", () => {
  test("the save payload sends provider_name and settings per slot, not a per-slot save", () => {
    expect(SOURCE).toMatch(/\{ provider_name, settings: settingsDraft\[slot\] \?\? \{\} \}/)
  })
})

describe("ProvidersRoute -- the language-model slot's Server URL field is revealed by data, not a provider name", () => {
  test("the field is gated on selectedOption?.needs_server_url, never a literal provider name check", () => {
    expect(SOURCE).toMatch(/selectedOption\?\.needs_server_url/)
    expect(SOURCE).not.toMatch(/=== *"local"/)
  })

  test("the field uses the Copywriting Contract's exact label and helper text", () => {
    expect(SOURCE).toMatch(/<Label htmlFor=\{`provider-\$\{slot\.slot\}-server-url`\}>Server URL<\/Label>/)
    expect(SOURCE).toMatch(/The OpenAI-compatible endpoint serving your local model\./)
  })

  test("the field reuses the existing scroll-field font-mono treatment, not a new input style", () => {
    expect(SOURCE).toMatch(/id=\{`provider-\$\{slot\.slot\}-server-url`\}\s*className="scroll-field font-mono"/)
  })

  test("the field is disabled while a save is in flight, same as every RadioGroupItem", () => {
    const fieldBlock = SOURCE.slice(
      SOURCE.indexOf("selectedOption?.needs_server_url"),
      SOURCE.indexOf("selectedOption?.needs_server_url") + 600,
    )
    expect(fieldBlock).toMatch(/disabled=\{disabled\}/)
  })
})
