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

/** `TimelineEntryResponse`'s exact shape. `stage`/`type`/`text`/... are
 * whichever extra fields `render_timeline` wrote for that entry's own
 * kind -- carried through untyped (`[key: string]: unknown`) rather than
 * re-declared here, the same "derived, never a second source of truth"
 * discipline `session/timeline.py` states for the merge itself. */
export interface TimelineEntry {
  ts: number
  kind: "event" | "stage"
  offset_s: number | null
  [key: string]: unknown
}

/** `SessionDetailResponse`'s exact shape. */
export interface SessionDetail {
  id: string
  started_at: string
  turn_outcome: string
  transcript: string | null
  reply_text: string | null
  stage_durations_ms: Record<string, number | null>
  end_of_speech_to_first_audio_ms: number | null
  end_of_speech_to_answer_audio_ms: number | null
  audio_format: { encoding: string; sample_rate: number } | null
  has_audio: boolean
  /** Seconds at the head of the recording captured before the wake word.
   * `0` when the session built no pre-roll buffer. The one place this is
   * derived server-side is `session/timeline.py::preroll_offset_s`. */
  preroll_s: number
  timeline: TimelineEntry[]
}

export const SESSIONS_QUERY_KEY = ["sessions"] as const

export function sessionQueryKey(id: string) {
  return ["sessions", id] as const
}

export function fetchSessions(): Promise<SessionSummary[]> {
  return apiFetch<SessionsListResponse>("/api/sessions").then((response) => response.sessions)
}

/** `encodeURIComponent` on the id, unlike `lib/plugins.ts`/`lib/macros.ts`/
 * `lib/workflows.ts`, which interpolate theirs raw. The difference is
 * where the id comes from: theirs come back from a server response, this
 * one comes from `useParams` -- i.e. from the address bar. An id holding a
 * `?` or a `#` silently retargets the request (`/api/sessions/a?b` asks for
 * `/api/sessions/a` with a query string), so D-04's named not-found copy is
 * replaced by whatever that different request answers. */
export function fetchSession(id: string): Promise<SessionDetail> {
  return apiFetch<SessionDetail>(`/api/sessions/${encodeURIComponent(id)}`)
}

/** The audio path for a session id -- a plain string, never a fetch. The
 * `<audio>` element's own `src` attribute issues its own browser-native
 * request and carries the session cookie automatically for a same-origin
 * address, so this one path deliberately does not go through `apiFetch`
 * (`lib/api.ts`'s own docstring: "No screen may call `fetch` directly").
 * This is the named exception to that rule, not a lapse -- `apiFetch`'s
 * JSON-shaped seam is the wrong fit for routing binary audio bytes. */
export function sessionAudioUrl(id: string): string {
  return `/api/sessions/${encodeURIComponent(id)}/audio`
}
