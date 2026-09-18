import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { Macro } from "@/lib/macros"
import {
  actionConflictDisplay,
  deriveMacroEditorState,
  deriveReplyPrecacheState,
  DUPLICATE_PHRASE_MESSAGE,
  isDuplicatePhrase,
  type QueryLike,
} from "./deriveMacroEditorState"

function pending(): QueryLike<Macro> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored(error: unknown): QueryLike<Macro> {
  return { status: "error", data: undefined, error }
}
function success(data: Macro): QueryLike<Macro> {
  return { status: "success", data, error: undefined }
}

function macro(overrides: Partial<Macro> = {}): Macro {
  return {
    id: 1,
    phrase: "goodnight",
    aliases: ["night night"],
    reply: "Good night.",
    actions: [],
    created_at: "2026-09-18T00:00:00Z",
    updated_at: "2026-09-18T00:00:00Z",
    created_by_user_id: 1,
    reply_cached: true,
    reply_synthesis_degraded: false,
    reply_synthesis_message: null,
    ...overrides,
  }
}

describe("deriveMacroEditorState -- new macro (no query at all)", () => {
  test("query === null is always ready with macro: null -- there is nothing to fetch", () => {
    expect(deriveMacroEditorState(null)).toEqual({ kind: "ready", macro: null })
  })
})

describe("deriveMacroEditorState -- editing an existing macro (opened by id)", () => {
  test("pending renders loading -- save stays blocked until the record resolves", () => {
    expect(deriveMacroEditorState(pending())).toEqual({ kind: "loading" })
  })

  test("a failed load carries the Copywriting Contract's exact fallback text", () => {
    expect(deriveMacroEditorState(errored(new Error("network down")))).toEqual({
      kind: "error",
      message: "Couldn't load this macro. It may have been deleted.",
    })
  })

  test("a failed load with a named server reason surfaces it unmodified", () => {
    expect(deriveMacroEditorState(errored(new ApiError(404, "no macro with id 5")))).toEqual({
      kind: "error",
      message: "no macro with id 5",
    })
  })

  test("a successful load renders ready with the macro", () => {
    const m = macro()
    expect(deriveMacroEditorState(success(m))).toEqual({ kind: "ready", macro: m })
  })
})

describe("isDuplicatePhrase -- a macro keeping its own phrase is never reported as a duplicate", () => {
  test("the macro being edited is excluded from the comparison set", () => {
    const editing = macro({ id: 1, phrase: "goodnight" })
    const others = [editing, macro({ id: 2, phrase: "lights out" })]
    expect(isDuplicatePhrase("goodnight", others, 1)).toBe(false)
  })

  test("another macro's phrase, normalized identically, is a duplicate", () => {
    const others = [macro({ id: 1, phrase: "Good Night!" }), macro({ id: 2, phrase: "lights out" })]
    expect(isDuplicatePhrase("good night", others, null)).toBe(true)
  })

  test("another macro's alias, normalized identically, is also a duplicate", () => {
    const others = [macro({ id: 1, phrase: "sleep mode", aliases: ["good night"] })]
    expect(isDuplicatePhrase("Good Night", others, null)).toBe(true)
  })

  test("a genuinely new phrase is not a duplicate", () => {
    const others = [macro({ id: 1, phrase: "goodnight" })]
    expect(isDuplicatePhrase("good morning", others, null)).toBe(false)
  })

  test("a blank phrase is never reported as a duplicate", () => {
    const others = [macro({ id: 1, phrase: "goodnight" })]
    expect(isDuplicatePhrase("   ", others, null)).toBe(false)
  })

  test("DUPLICATE_PHRASE_MESSAGE is the Copywriting Contract's exact inline text", () => {
    expect(DUPLICATE_PHRASE_MESSAGE).toBe("This phrase is already used by another macro.")
  })
})

describe("deriveReplyPrecacheState -- cached, unsaved-and-therefore-absent, and failed", () => {
  test("a new, never-saved macro (macro: null) is absent, never a placeholder badge", () => {
    expect(deriveReplyPrecacheState(null, null)).toBe("absent")
  })

  test("a saved macro whose reply is in the cache is cached", () => {
    expect(deriveReplyPrecacheState(macro({ reply_cached: true }), null)).toBe("cached")
  })

  test("a saved macro whose reply is not yet in the cache is absent", () => {
    expect(deriveReplyPrecacheState(macro({ reply_cached: false }), null)).toBe("absent")
  })

  test("a synthesis failure message takes precedence over the macro's own cached flag", () => {
    expect(deriveReplyPrecacheState(macro({ reply_cached: true }), "couldn't prepare it")).toBe("failed")
  })
})

describe("actionConflictDisplay -- unknown is a distinct state from ok (no conflict), never rendered as clean", () => {
  test("ok renders no badge slot at all", () => {
    expect(actionConflictDisplay("ok")).toEqual({ kind: "ok" })
  })

  test("unknown is its own kind, with its own muted-line text, never equal to ok's shape", () => {
    const unknown = actionConflictDisplay("unknown")
    const ok = actionConflictDisplay("ok")
    expect(unknown.kind).not.toBe(ok.kind)
    expect(unknown).toEqual({ kind: "unknown", text: "Couldn't check this against the safety policy." })
  })

  test("denied carries the editor's own trailing clause, not the policy screen's", () => {
    expect(actionConflictDisplay("denied")).toEqual({
      kind: "denied",
      text: "Denied for control — would be refused when this macro runs.",
    })
  })

  test("not_found reuses Phase 3's exact string verbatim", () => {
    expect(actionConflictDisplay("not_found")).toEqual({ kind: "not_found", text: "Not found in Home Assistant" })
  })
})
