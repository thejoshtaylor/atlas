// MacroEditorRoute's own logic, kept apart from its JSX -- copied
// verbatim in shape from `routes/policy/derivePolicyScreenState.ts`'s
// `QueryLike` interface and discriminated union. Imports nothing from
// React, for the same stated reason.
import { ApiError } from "@/lib/api"
import type { ConflictAnnotation, Macro } from "@/lib/macros"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

// `macro: null` is the "new, unsaved macro" ready state -- there is no
// id to fetch yet, so `MacroEditorRoute` never runs a query for it at
// all (see `query: null` below) rather than deriving a fake loading
// state for a fetch that would never happen.
export type MacroEditorScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; macro: Macro | null }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load this macro. It may have been deleted."
}

/**
 * `query` is `null` for a new macro (`/macros/new`, no id to fetch) --
 * that route never contacts the server at all, so it is always ready
 * with `macro: null` rather than a query that stays pending forever.
 */
export function deriveMacroEditorState(query: QueryLike<Macro> | null): MacroEditorScreenState {
  if (query === null) return { kind: "ready", macro: null }
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  if (!query.data) return { kind: "loading" }
  return { kind: "ready", macro: query.data }
}

// A deliberately advisory mirror of `atlas.turn.macros.normalize`
// (NFKC fold, case fold, the same punctuation class, whitespace
// collapse) -- not byte-identical (JS has no `casefold()`, so
// `toLowerCase()` stands in for it), and it does not need to be: this
// check exists for the operator's benefit before submit, never as the
// boundary itself. `_check_macros_do_not_collide` (routes/macros.py) is
// the real check and runs again on save regardless of what this reports
// (T-04-37).
const PUNCT_RE = /[.,!?;:'"()[\]{}\-_/\\]/g

function normalizePhrase(text: string): string {
  return text
    .normalize("NFKC")
    .toLowerCase()
    .replace(PUNCT_RE, "")
    .split(/\s+/)
    .filter(Boolean)
    .join(" ")
}

/** Every other macro's normalized phrase and aliases, excluding the
 * macro currently being edited -- mirrors the collision surface
 * `_check_macros_do_not_collide` actually checks against
 * (`Macro.normalized_keys` is phrase + every alias), so a duplicate this
 * function misses because it only compared phrases would still be
 * caught by the save, just later than this check exists to catch it. */
function otherMacroNormalizedKeys(macros: Macro[], excludingMacroId: number | null): Set<string> {
  const keys = new Set<string>()
  for (const macro of macros) {
    if (macro.id === excludingMacroId) continue
    keys.add(normalizePhrase(macro.phrase))
    for (const alias of macro.aliases) keys.add(normalizePhrase(alias))
  }
  return keys
}

/**
 * A macro keeping its own phrase is not reported as a duplicate: the
 * macro being edited is excluded from the comparison set before the
 * check runs, the same self-collision-is-legal rule
 * `_check_no_collision` (routes/macros.py) applies server-side.
 */
export function isDuplicatePhrase(phrase: string, macros: Macro[], excludingMacroId: number | null): boolean {
  const key = normalizePhrase(phrase)
  if (!key) return false
  return otherMacroNormalizedKeys(macros, excludingMacroId).has(key)
}

/** UI-SPEC's exact "Error state -- duplicate phrase on save" copy. */
export const DUPLICATE_PHRASE_MESSAGE = "This phrase is already used by another macro."

// MED-01 fix (phase 4 code review): `MacroConfig.from_config` (the file
// parser) has always refused a blank phrase or blank reply
// (`_validate_actions`, `routes/macros.py`, mirrors that check on the
// route since this same fix). Neither `Input` below had a client-side
// guard before this fix, so an operator clearing the Phrase field while
// editing, or leaving Reply empty, could hit Save before ever learning
// the macro would be permanently dead (a blank phrase normalizes to a
// key `turn/macros.py::match()` never matches a transcript against) or
// would fail to precache (a blank reply handed to the TTS provider).
export const BLANK_PHRASE_SAVE_BLOCKED_REASON = "Add a phrase before saving."
export const BLANK_REPLY_SAVE_BLOCKED_REASON = "Add a reply before saving."

/** `.trim()`, not a bare emptiness check -- a phrase of only whitespace
 * normalizes to the same empty key `normalize("")` does server-side, so
 * treating it as non-blank here would let it slip past this guard only
 * to be caught later by the same route this guard exists to get ahead
 * of. */
export function saveBlockedByBlankPhrase(phrase: string): boolean {
  return phrase.trim() === ""
}

export function saveBlockedByBlankReply(reply: string): boolean {
  return reply.trim() === ""
}

export type ReplyPrecacheState = "cached" | "absent" | "failed"

/**
 * The three reply precache states (Task 1's own behaviour):
 * - `failed`   -- the most recent save persisted the macro but could not
 *                 prepare its reply (`synthesisFailedMessage` is the
 *                 server's own degraded-save message, carried by the
 *                 caller from the mutation response -- a GET never
 *                 reports this, only a create/update response does).
 * - `cached`   -- the server's own `reply_cached` flag is true.
 * - `absent`   -- everything else, including "this macro has never been
 *                 saved" (`macro` is `null`) and "saved, but its reply
 *                 is not yet in the cache."
 */
export function deriveReplyPrecacheState(macro: Macro | null, synthesisFailedMessage: string | null): ReplyPrecacheState {
  if (synthesisFailedMessage) return "failed"
  if (macro?.reply_cached) return "cached"
  return "absent"
}

export type ActionConflictDisplay =
  | { kind: "ok" }
  | { kind: "denied"; text: string }
  | { kind: "not_found"; text: string }
  | { kind: "unknown"; text: string }

/**
 * Every string here is 04-UI-SPEC.md's Copywriting Contract, verbatim.
 * `unknown` is a distinct member from `ok` on purpose -- a policy or
 * catalog read that failed while the editor was open must never render
 * as "no conflict" (T-04-36); this function is what a test asserts that
 * distinction against directly, without a rendered DOM.
 */
export function actionConflictDisplay(conflict: ConflictAnnotation): ActionConflictDisplay {
  switch (conflict) {
    case "ok":
      return { kind: "ok" }
    case "denied":
      return { kind: "denied", text: "Denied for control — would be refused when this macro runs." }
    case "not_found":
      return { kind: "not_found", text: "Not found in Home Assistant" }
    case "unknown":
      return { kind: "unknown", text: "Couldn't check this against the safety policy." }
  }
}
