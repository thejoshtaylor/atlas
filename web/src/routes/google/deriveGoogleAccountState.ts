// GoogleAccountRoute's own logic, kept apart from its JSX -- copied in
// shape from `routes/plugins/derivePluginEditorState.ts`. Imports nothing
// from React.
import { ApiError } from "@/lib/api"
import type { GoogleAccount, GoogleAccountStatus, GoogleCalendar, GoogleCalendarAccess } from "@/lib/google"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type GoogleAccountScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "not_found" }
  | { kind: "ready"; account: GoogleAccount }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load this account. Try again."
}

/** An unknown id (`GET /api/google/accounts/{id}` answering 404) is its
 * own state, distinct from a transient load failure -- the not-found
 * state renders with no Retry control, since retrying a request for an
 * id that does not exist can never succeed. */
export function deriveGoogleAccountState(query: QueryLike<GoogleAccount>): GoogleAccountScreenState {
  if (query.status === "error") {
    if (query.error instanceof ApiError && query.error.status === 404) return { kind: "not_found" }
    return { kind: "error", message: messageFor(query.error) }
  }
  if (query.status === "pending") return { kind: "loading" }
  if (!query.data) return { kind: "loading" }
  return { kind: "ready", account: query.data }
}

export interface CalendarAccessOption {
  value: GoogleCalendarAccess
  label: string
  disabled: boolean
}

/** D-05: named here rather than written inline, like every other
 * copywriting line on these screens. */
export const READ_ONLY_CALENDAR_NOTE = "Google shares this calendar read-only"

/** Off, Read only, and Read and write, in that order -- Read and write is
 * `disabled` exactly when Google itself reports this calendar read-only
 * (`can_write: false`). A new calendar arrives with `access: "off"`
 * server-side (09-03-PLAN.md); this function only describes the three
 * options, never the stored value -- the caller reads that straight off
 * `calendar.access`. */
export function calendarAccessOptions(calendar: Pick<GoogleCalendar, "can_write">): CalendarAccessOption[] {
  return [
    { value: "off", label: "Off", disabled: false },
    { value: "read_only", label: "Read only", disabled: false },
    { value: "read_write", label: "Read and write", disabled: !calendar.can_write },
  ]
}

export interface AccountStatusDisplay {
  badgeVariant: "secondary" | "denied"
  badgeText: string
}

/** `GoogleAccountStatus`'s exact three values, one badge each -- the
 * same "one status maps to exactly one badge" rule
 * `runtimeStatusDisplay` (`derivePluginsScreenState.ts`) already
 * establishes for plugin state. */
export function accountStatusDisplay(status: GoogleAccountStatus): AccountStatusDisplay {
  switch (status) {
    case "ok":
      return { badgeVariant: "secondary", badgeText: "Linked" }
    case "needs_relink":
      return { badgeVariant: "denied", badgeText: "Needs re-link" }
    case "unreachable":
      return { badgeVariant: "denied", badgeText: "Unreachable" }
  }
}
