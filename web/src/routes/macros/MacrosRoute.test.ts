import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "MacrosRoute.tsx"), "utf-8")
const APP_SOURCE = readFileSync(join(import.meta.dir, "..", "..", "App.tsx"), "utf-8")
const SHELL_SOURCE = readFileSync(join(import.meta.dir, "..", "..", "components", "layout", "AppShell.tsx"), "utf-8")

describe("MacrosRoute -- the empty state carries the Copywriting Contract's exact copy and the create CTA", () => {
  test("heading and body", () => {
    expect(SOURCE).toMatch(/heading="No macros yet\."/)
    expect(SOURCE).toMatch(
      /A macro is a phrase that runs one or more actions without waiting on the model\./,
    )
  })

  test("the CTA is passed through EmptyState's own action prop, not a second bespoke button", () => {
    expect(SOURCE).toMatch(/<EmptyState[\s\S]{0,300}action=\{newMacroButton\}/)
  })

  test("the CTA reads \"New macro\" and links to /macros/new", () => {
    expect(SOURCE).toMatch(/<Link to="\/macros\/new">New macro<\/Link>/)
  })
})

describe("MacrosRoute -- loading never renders as an empty list", () => {
  test("SkeletonList renders for screen.kind === \"loading\"", () => {
    expect(SOURCE).toMatch(/screen\.kind === "loading" \? <SkeletonList/)
  })
})

describe("MacrosRoute -- a failed load disables create, edit and delete", () => {
  test("controlsDisabled derives from screen.kind !== \"ready\" and gates the new-macro button and every row's delete button", () => {
    expect(SOURCE).toMatch(/const controlsDisabled = screen\.kind !== "ready"/)
    expect(SOURCE).toMatch(/<Button asChild disabled=\{controlsDisabled\}>/)
    expect(SOURCE).toMatch(/disabled=\{disabled \|\| remove\.isPending\}/)
  })

  test("ErrorState renders with a retry for screen.kind === \"error\"", () => {
    expect(SOURCE).toMatch(/screen\.kind === "error" \? \(/)
    expect(SOURCE).toMatch(/onRetry=\{\(\) => void query\.refetch\(\)\}/)
  })
})

describe("MacrosRoute -- delete is a three-part destructive confirmation with the contract's exact text", () => {
  test("dialog body names the macro's phrase and states the consequence", () => {
    expect(SOURCE).toMatch(
      /Delete "\$\{macro\.phrase\}"\? This macro will no longer run, and its cached reply is discarded\./,
    )
  })

  test("the confirm button reads \"Delete macro\" and is destructive-styled", () => {
    expect(SOURCE).toMatch(/variant="destructive"[\s\S]{0,200}Delete macro/)
  })
})

describe("MacrosRoute -- each card shows a singular/plural action count via the shared formatter", () => {
  test("formatActionCount is the only source of the count text", () => {
    expect(SOURCE).toMatch(/formatActionCount\(macro\.actions\.length\)/)
  })
})

describe("MacrosRoute -- a macro with a denied/unresolved action shows the list-level summary badge", () => {
  test("the denied badge only renders when conflictedActionCount > 0, via the shared formatter", () => {
    expect(SOURCE).toMatch(/conflicted > 0 \? \(/)
    expect(SOURCE).toMatch(/<Badge variant="denied">\{formatConflictSummary\(conflicted\)\}<\/Badge>/)
  })
})

describe("MacrosRoute -- built from the enumerated inventory only, no data table", () => {
  test("cards are a hand-styled <li>, matching PolicyRuleRow's own precedent, not the Card primitive", () => {
    expect(SOURCE).toMatch(/<li className="flex items-center justify-between gap-3 px-4 py-3">/)
  })

  test("no @/components/ui/table import", () => {
    expect(SOURCE).not.toMatch(/@\/components\/ui\/table/)
  })
})

describe("App.tsx -- the macros list is mounted inside the authenticated shell at operator level", () => {
  test("MacrosRoute is imported and mounted at /macros beside PolicyRoute", () => {
    expect(APP_SOURCE).toMatch(/import \{ MacrosRoute \} from "@\/routes\/macros\/MacrosRoute"/)
    expect(APP_SOURCE).toMatch(/<Route path="\/macros" element=\{<MacrosRoute \/>\} \/>/)
  })
})

describe("AppShell.tsx -- a Macros navigation entry exists at operator level", () => {
  test("nav item points at /macros with minimumRole operator", () => {
    expect(SHELL_SOURCE).toMatch(/\{ to: "\/macros", label: "Macros", minimumRole: "operator"[,} ]/)
  })
})
