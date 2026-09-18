import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { Macro } from "@/lib/macros"
import {
  conflictedActionCount,
  deriveMacrosScreenState,
  formatActionCount,
  formatConflictSummary,
  type QueryLike,
} from "./deriveMacrosScreenState"

function pending(): QueryLike<Macro[]> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored(error: unknown): QueryLike<Macro[]> {
  return { status: "error", data: undefined, error }
}
function success(data: Macro[]): QueryLike<Macro[]> {
  return { status: "success", data, error: undefined }
}

function macro(overrides: Partial<Macro> = {}): Macro {
  return {
    id: 1,
    phrase: "goodnight",
    aliases: [],
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

describe("deriveMacrosScreenState", () => {
  test("pending renders loading -- the list must show skeleton rows, never an empty flash", () => {
    expect(deriveMacrosScreenState(pending())).toEqual({ kind: "loading" })
  })

  test("a failed load carries the Copywriting Contract's exact fallback text when the server gives none more specific", () => {
    expect(deriveMacrosScreenState(errored(new Error("network down")))).toEqual({
      kind: "error",
      message: "Couldn't load macros. Try again.",
    })
  })

  test("a failed load with a named server reason surfaces it unmodified", () => {
    expect(deriveMacrosScreenState(errored(new ApiError(500, "database unreachable")))).toEqual({
      kind: "error",
      message: "database unreachable",
    })
  })

  test("a successful load renders ready with every macro", () => {
    const macros = [macro({ id: 1 }), macro({ id: 2 })]
    expect(deriveMacrosScreenState(success(macros))).toEqual({ kind: "ready", macros })
  })
})

describe("formatActionCount -- zero-one-many", () => {
  test("1 reads as singular", () => {
    expect(formatActionCount(1)).toBe("1 action")
  })
  test("0 and >1 read as plural", () => {
    expect(formatActionCount(0)).toBe("0 actions")
    expect(formatActionCount(3)).toBe("3 actions")
  })
})

describe("conflictedActionCount -- denied and not_found count as a conflict, ok and unknown do not", () => {
  test("counts only denied and not_found actions", () => {
    const m = macro({
      actions: [
        { id: 1, position: 0, tool: "t", arguments: {}, conflict: "ok" },
        { id: 2, position: 1, tool: "t", arguments: {}, conflict: "denied" },
        { id: 3, position: 2, tool: "t", arguments: {}, conflict: "not_found" },
        { id: 4, position: 3, tool: "t", arguments: {}, conflict: "unknown" },
      ],
    })
    expect(conflictedActionCount(m)).toBe(2)
  })

  test("a macro with no conflicted actions counts zero", () => {
    const m = macro({ actions: [{ id: 1, position: 0, tool: "t", arguments: {}, conflict: "ok" }] })
    expect(conflictedActionCount(m)).toBe(0)
  })
})

describe("formatConflictSummary -- zero-one-many, verbatim Copywriting Contract text", () => {
  test("1 reads as singular", () => {
    expect(formatConflictSummary(1)).toBe("1 action denied by policy")
  })
  test("many reads as plural", () => {
    expect(formatConflictSummary(3)).toBe("3 actions denied by policy")
  })
})
