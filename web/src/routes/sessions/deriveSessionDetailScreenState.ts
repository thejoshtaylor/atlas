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
  /** `stale` is true when this session's data is real and previously
   * fetched but the most recent fetch failed. The screen keeps showing it
   * and says the refresh failed, rather than discarding data it is still
   * holding. */
  | { kind: "ready"; session: SessionDetail; stale: boolean }
  | { kind: "removed" }
  | { kind: "not_found" }
  /** A failure this module has not classified -- shown with the server's
   * own reason. `routes/sessions.py` answers 409 with a named refusal for
   * a recording that was never finished being written and for one this
   * deployment cannot play back; neither is a missing session and neither
   * should read as one. */
  | { kind: "failed"; message: string }

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

/** A 404 specifically. The loaded-then-gone reading below turns on this:
 * only the server saying the session is not there is evidence that it
 * went away. A 500, a dropped connection or an expired cookie says
 * nothing about whether the recording still exists. */
function isNotFound(error: unknown): boolean {
  return error instanceof ApiError && error.status === 404
}

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load this session. Try again."
}

/**
 * `hasEverLoaded` is the one piece of state this pure function cannot
 * derive from the query object alone: D-04's client-side case (08-UI-
 * SPEC.md's UI Considerations table) is "the client already holds real,
 * previously-loaded data for this session; a subsequent poll or a failed
 * follow-up fetch ... is read as a loaded-then-gone transition". The
 * caller (`SessionDetailRoute.tsx`) tracks whether the `ready` branch was
 * ever reached for this session id and passes that fact in; this module
 * still owns the decision of what it means.
 *
 * What it does *not* mean, as of WR-06's review: any error whatsoever.
 * `hasEverLoaded && error -> removed` read a 500, a dropped connection,
 * an expired cookie or the audio route's own 409 as "This session was
 * removed. ... This one is gone, and that's expected." -- for a session
 * that is neither gone nor expected to be. It is reachable without any
 * exotic setup: `refetchOnMount` is on, so navigating away from
 * `/sessions/:id` and back serves the cached entry (flipping the flag on
 * the first commit) and then refetches in the background. TanStack Query
 * v5 moves `status` to `"error"` while still holding `data`, so the screen
 * both discarded real data it was still holding and stated a reason it had
 * not established -- the inverse of this project's own "a check that could
 * not be performed is reported as such" discipline.
 *
 * So the loaded-then-gone transition is narrowed to a 404, which is the
 * only answer that is evidence of going away, and cached data outranks an
 * unclassified failure.
 */
export function deriveSessionDetailScreenState(input: {
  query: QueryLike<SessionDetail>
  hasEverLoaded: boolean
}): SessionDetailScreenState {
  const { query, hasEverLoaded } = input

  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "success" && query.data) return { kind: "ready", session: query.data, stale: false }
  if (query.status === "error") {
    // The server's own named refusal outranks everything, held or not:
    // it says what happened.
    if (isRemovedByRetention(query.error)) return { kind: "removed" }
    // Real data still in hand beats an unexplained failure. Stale, and
    // said to be.
    if (query.data) return { kind: "ready", session: query.data, stale: true }
    if (isNotFound(query.error)) return hasEverLoaded ? { kind: "removed" } : { kind: "not_found" }
    return { kind: "failed", message: messageFor(query.error) }
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
