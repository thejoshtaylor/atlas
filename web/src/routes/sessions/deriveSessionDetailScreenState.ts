// SessionDetailRoute's own logic, kept apart from its JSX -- follows
// `deriveCalibrationScreenState.ts`'s multi-branch shape. Imports nothing
// from React: what belongs here is only ever a fact about data, never a
// fact about a render.
import { ApiError } from "@/lib/api"
import type { SessionDetail } from "@/lib/sessions"

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
