// SessionDetailRoute's own logic, kept apart from its JSX -- follows
// `deriveCalibrationScreenState.ts`'s multi-branch shape. Imports nothing
// from React: what belongs here is only ever a fact about data, never a
// fact about a render.
import { ApiError } from "@/lib/api"
import type { SessionDetail, TimelineEntry } from "@/lib/sessions"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type SessionDetailScreenState =
  | { kind: "loading" }
  | { kind: "ready"; session: SessionDetail }
  | { kind: "removed" }
  | { kind: "not_found" }

// The server's own refusal wording (`routes/sessions.py::_removed_by_
// retention_error`), matched case-insensitively so a paraphrase never
// silently drifts this classification -- `apiFetch` carries the server's
// `detail` string through verbatim (`lib/api.ts`'s own contract), so this
// is comparing against a string this module does not own but must not
// paraphrase either.
const REMOVED_BY_RETENTION_MARKER = "removed by the retention sweep"

function isRemovedByRetention(error: unknown): boolean {
  return error instanceof ApiError && error.message.toLowerCase().includes(REMOVED_BY_RETENTION_MARKER)
}

/**
 * `hasEverLoaded` is the one piece of state this pure function cannot
 * derive from the query object alone: D-04's client-side case (08-UI-
 * SPEC.md's UI Considerations table) is "the client already holds real,
 * previously-loaded data for this session; a subsequent poll or a failed
 * follow-up fetch ... is read as a loaded-then-gone transition" --
 * regardless of what the second failure's own error text says. The
 * caller (`SessionDetailRoute.tsx`) tracks whether the `ready` branch was
 * ever reached for this session id and passes that fact in; this module
 * still owns the decision of what it means.
 */
export function deriveSessionDetailScreenState(input: {
  query: QueryLike<SessionDetail>
  hasEverLoaded: boolean
}): SessionDetailScreenState {
  const { query, hasEverLoaded } = input

  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "success" && query.data) return { kind: "ready", session: query.data }
  if (query.status === "error") {
    if (hasEverLoaded) return { kind: "removed" }
    return isRemovedByRetention(query.error) ? { kind: "removed" } : { kind: "not_found" }
  }
  return { kind: "loading" }
}

/**
 * DBG-03/D-09's other pure half: which timeline row is active at a given
 * `<audio>` `currentTime`. The last entry whose `offset_s` is at or before
 * `currentTime` -- inclusive at the entry's own offset, because a row
 * becomes active at the instant its event happened, not a tick later. A
 * tie (two entries at the same offset) resolves to the later of the two,
 * matching the order `session/recorder.py` appended them and
 * `render_timeline`'s own stable sort preserves. `null` before the first
 * entry's offset, or for an empty timeline -- there is nothing active yet.
 * An entry with `offset_s: null` never becomes active (no offset to
 * compare against a clock).
 */
export function activeTimelineIndexAt(entries: TimelineEntry[], currentTime: number): number | null {
  let activeIndex: number | null = null
  for (let index = 0; index < entries.length; index += 1) {
    const offset = entries[index].offset_s
    if (offset !== null && offset <= currentTime) {
      activeIndex = index
    }
  }
  return activeIndex
}
