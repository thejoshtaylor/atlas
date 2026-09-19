import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

// `web/` has no rendered-DOM test infrastructure (no Playwright, no
// @testing-library/*) -- this asserts the source text carries the wiring
// 07-UI-SPEC.md requires, the same limitation `PluginsRoute.test.ts`
// states for itself. It cannot prove the page actually renders correctly
// in a browser.
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

  test("the save button reads the Copywriting Contract's exact text", () => {
    expect(SOURCE).toMatch(/Save provider choices/)
  })

  test("save success and failure copy match the Copywriting Contract verbatim", () => {
    expect(SOURCE).toMatch(/Saved\. Restart the assistant for this to take effect\./)
    expect(SOURCE).toMatch(/Couldn't save your provider choices\. Try again\./)
  })
})

describe("ProvidersRoute -- one card per slot, driven entirely by the server response", () => {
  test("cards render from screen.slots.map, not a hardcoded slot list", () => {
    expect(SOURCE).toMatch(/screen\.slots\.map/)
  })

  test("each slot renders its own RadioGroup bound to the draft selection", () => {
    expect(SOURCE).toMatch(/<RadioGroup value=\{selected\} onValueChange=\{onSelect\}/)
  })

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
  test("the draft state is only ever set from the loaded selection once per slot, never reset on save failure", () => {
    expect(SOURCE).not.toMatch(/setDraft\([^)]*\{\}\)/)
  })
})
