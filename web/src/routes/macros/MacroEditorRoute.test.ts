import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "MacroEditorRoute.tsx"), "utf-8")
const APP_SOURCE = readFileSync(join(import.meta.dir, "..", "..", "App.tsx"), "utf-8")

describe("MacroEditorRoute -- save is blocked, with the reason inline, before a doomed submit", () => {
  test("zero actions blocks save, with the exact zero-action copy shown inline under the action list", () => {
    expect(SOURCE).toMatch(/const zeroActionsReason = saveBlockedByEmptyActions\(actions\)/)
    expect(SOURCE).toMatch(/const saveDisabled = recordLoading \|\| actions\.length === 0 \|\| duplicate/)
    expect(SOURCE).toMatch(/\{zeroActionsReason \? <p className="text-body text-destructive">\{zeroActionsReason\}<\/p> : null\}/)
  })

  test("a duplicate phrase blocks save, with the exact Copywriting Contract text inline under the Phrase field", () => {
    expect(SOURCE).toMatch(/const duplicate = macrosListQuery\.data \? isDuplicatePhrase\(phrase, macrosListQuery\.data, macroId\) : false/)
    expect(SOURCE).toMatch(/\{duplicate \? <p className="text-body text-destructive">\{DUPLICATE_PHRASE_MESSAGE\}<\/p> : null\}/)
  })

  test("save stays blocked while the record is still loading", () => {
    expect(SOURCE).toMatch(/const recordLoading = screen\.kind !== "ready"/)
  })
})

describe("MacroEditorRoute -- the reply's three precache states, and no second progress indicator", () => {
  test("cached renders the exact secondary badge text", () => {
    expect(SOURCE).toMatch(/precache === "cached" \? <Badge variant="secondary">Reply cached · ready<\/Badge> : null/)
  })

  test("failed renders destructive text plus an outline Retry, never a badge", () => {
    expect(SOURCE).toMatch(/precache === "failed" \? \(/)
    expect(SOURCE).toMatch(/<p className="text-body text-destructive">\{synthesisFailedMessage\}<\/p>/)
    expect(SOURCE).toMatch(/variant="outline" onClick=\{\(\) => void handleSave\(\)\}>\s*Retry/)
  })

  test("SubmitButton alone covers the save-plus-resynthesis round trip -- no second spinner or progress bar", () => {
    expect(SOURCE).toMatch(/<SubmitButton onSubmit=\{handleSave\} disabled=\{saveDisabled\}>/)
    expect(SOURCE).not.toMatch(/role="progressbar"/)
  })
})

describe("MacroEditorRoute -- the action row's safety-conflict slot is never omitted", () => {
  test("denied and not_found both render the amber badge; unknown renders the muted could-not-check line, never nothing", () => {
    expect(SOURCE).toMatch(/conflict\.kind === "denied" \|\| conflict\.kind === "not_found" \? \(/)
    expect(SOURCE).toMatch(/<Badge variant="denied" className="w-fit">/)
    expect(SOURCE).toMatch(/conflict\.kind === "unknown" \? <p className="text-label text-muted-foreground">\{conflict\.text\}<\/p> : null/)
  })

  test("a newly added, unsaved action (no server annotation yet) defaults to ok -- no badge, not a false unknown", () => {
    expect(SOURCE).toMatch(/conflictByKey\.get\(action\.key\) \?\? "ok"/)
  })
})

describe("MacroEditorRoute -- the action row is two lines: summary+remove, then a right-aligned move pair", () => {
  test("Remove sits on the summary line", () => {
    expect(SOURCE).toMatch(/<span className="truncate text-body font-medium text-foreground">\s*\{`\$\{action\.domain\}\.\$\{action\.service\} → \$\{action\.entityId\}`\}/)
    expect(SOURCE).toMatch(/onClick=\{\(\) => removeAction\(action\.key\)\}\s*>\s*Remove/)
  })

  test("move up/down are icon-only with accessible names, disabled at the boundary via the store's own facts", () => {
    expect(SOURCE).toMatch(/aria-label="Move up"[\s\S]{0,80}disabled=\{isFirstAction\(actions, action\.key\)\}/)
    expect(SOURCE).toMatch(/aria-label="Move down"[\s\S]{0,80}disabled=\{isLastAction\(actions, action\.key\)\}/)
  })

  test("at least two aria-labels exist -- the reorder pair, icon-only, needs an accessible name", () => {
    const matches = SOURCE.match(/aria-label=/g) ?? []
    expect(matches.length).toBeGreaterThanOrEqual(2)
  })

  test("the action list is numbered -- an ordered list, and each row shows its own index", () => {
    expect(SOURCE).toMatch(/<ol className="flex flex-col gap-2">/)
    expect(SOURCE).toMatch(/\{index \+ 1\}\./)
  })
})

describe("MacroEditorRoute -- removing an action or an alias needs no confirmation; only deleting the macro does", () => {
  test("removeAction and removeAlias are called directly on click, no AlertDialog wraps them", () => {
    expect(SOURCE).toMatch(/onClick=\{\(\) => removeAction\(action\.key\)\}/)
    expect(SOURCE).toMatch(/onClick=\{\(\) => removeAlias\(alias\)\}/)
  })

  test("deleting the macro is a three-part destructive AlertDialog, separated from Save at the bottom of the screen", () => {
    expect(SOURCE).toMatch(
      /Delete "\$\{phrase\}"\? This macro will no longer run, and its cached reply is discarded\./,
    )
    expect(SOURCE).toMatch(/border-t border-border pt-4/)
  })
})

describe("MacroEditorRoute -- aliases are an unordered add/remove list, no numbering, no move controls", () => {
  test("aliases render in a <ul>, not the numbered <ol> the actions use", () => {
    expect(SOURCE).toMatch(/<ul className="flex flex-col gap-2">\s*\{aliases\.map/)
  })

  test("the alias row's own <li> carries no reorder button, only Remove", () => {
    const aliasLi = SOURCE.match(/<li key=\{alias\}[\s\S]*?<\/li>/)?.[0] ?? ""
    expect(aliasLi).not.toBe("")
    expect(aliasLi).not.toMatch(/aria-label="Move/)
  })
})

describe("MacroEditorRoute -- structured fields only, never a free-text configuration parser", () => {
  test("no textarea, yaml, or JSON.parse in this file", () => {
    expect(SOURCE).not.toMatch(/textarea/i)
    expect(SOURCE).not.toMatch(/yaml/i)
    expect(SOURCE).not.toMatch(/JSON\.parse/)
  })

  test("the reply field reuses the existing scroll-field treatment, not a new component", () => {
    expect(SOURCE).toMatch(/id="macro-reply" className="scroll-field"/)
  })
})

describe("MacroEditorRoute -- a single macro load failure offers a retry and a way back to the list", () => {
  test("ErrorState plus a plain link back to /macros", () => {
    expect(SOURCE).toMatch(/<ErrorState message=\{screen\.message\} onRetry=\{\(\) => void macroQuery\.refetch\(\)\}/)
    expect(SOURCE).toMatch(/<Link to="\/macros"[\s\S]{0,200}Back to macros/)
  })
})

describe("MacroEditorRoute -- the title is the macro's own phrase, or \"New macro\" for an unsaved one", () => {
  test("title derives from screen.macro.phrase when present, else the new-macro title", () => {
    expect(SOURCE).toMatch(/const title = screen\.kind === "ready" && screen\.macro \? screen\.macro\.phrase : "New macro"/)
  })
})

describe("App.tsx -- the editor is mounted under the list's own path, at operator level", () => {
  test("MacroEditorRoute is imported and mounted at /macros/new and /macros/:id", () => {
    expect(APP_SOURCE).toMatch(/import \{ MacroEditorRoute \} from "@\/routes\/macros\/MacroEditorRoute"/)
    expect(APP_SOURCE).toMatch(/<Route path="\/macros\/new" element=\{<MacroEditorRoute \/>\} \/>/)
    expect(APP_SOURCE).toMatch(/<Route path="\/macros\/:id" element=\{<MacroEditorRoute \/>\} \/>/)
  })
})
