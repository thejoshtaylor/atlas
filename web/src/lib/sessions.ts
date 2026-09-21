// The sessions review surface's fetch layer (WEB-07, 08-04-PLAN.md). Every
// shape here matches `src/spire_voice/routes/sessions.py`'s own response
// models field for field, the same discipline `lib/plugins.ts` states at
// its own top of file -- a renamed or reshaped field here is a silent
// drift from what the server actually sends.
import { apiFetch } from "./api"

/** `SessionSummaryResponse`'s exact shape. No source field -- deliberate,
 * see `routes/sessions.py`'s own module docstring: which source heard a
 * turn is not among the artifacts `session/recorder.py` writes. */
export interface SessionSummary {
  id: string
  started_at: string
  turn_outcome: string
  reply_text: string | null
  duration_ms: number | null
  has_audio: boolean
}

interface SessionsListResponse {
  sessions: SessionSummary[]
}

export const SESSIONS_QUERY_KEY = ["sessions"] as const

export function fetchSessions(): Promise<SessionSummary[]> {
  return apiFetch<SessionsListResponse>("/api/sessions").then((response) => response.sessions)
}
