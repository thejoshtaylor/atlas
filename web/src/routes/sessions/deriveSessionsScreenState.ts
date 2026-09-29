// SessionsRoute's own logic, kept apart from its JSX -- copied in shape
// from `routes/plugins/derivePluginsScreenState.ts`. Imports nothing from
// React, for the same stated reason: what belongs here is only ever a
// fact about data, never a fact about a render.
import { ApiError } from "@/lib/api"
import { formatMs } from "@/lib/format"
import type { SessionSpeaker, SessionSummary } from "@/lib/sessions"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

// `derivePluginsScreenState.ts` has no `empty` member because the plugin
// list is never empty (Home Assistant and weather are always-seeded
// builtin rows). The sessions list genuinely can be -- a fresh install
// before the first turn -- so this diverges deliberately, added here in
// Task 3 rather than Task 1, so the divergence reads as a decision, not a
// copy that drifted.
export type SessionsScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "empty" }
  | { kind: "ready"; sessions: SessionSummary[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load sessions. Try again."
}

export function deriveSessionsScreenState(query: QueryLike<SessionSummary[]>): SessionsScreenState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  const sessions = query.data
  if (!sessions) return { kind: "loading" }
  if (sessions.length === 0) return { kind: "empty" }
  return { kind: "ready", sessions }
}

// 08-UI-SPEC.md's Copywriting Contract, verbatim, for the two named
// outcomes -- a closed map with an explicit fallback that returns the raw
// value. The fallback is the point: an unrecognised `turn_outcome` is
// shown verbatim, never a fabricated friendlier label for a state this
// contract has not confirmed exists. `turn/controller.py` sets at least
// nine distinct outcome values today (this plan's own action text); a
// tenth added later shows up here as a raw string, never a wrong label.
const OUTCOME_SUMMARY: Record<string, string> = {
  empty_transcript: "Understood no speech.",
  empty_reply: "No reply was given.",
  // Plan 11-10 (D-09, D-13): the gate's own silent stop for a speaker this
  // deployment does not recognise -- a raw `unknown_speaker` string would
  // read as an error rather than the deliberate refusal it is.
  unknown_speaker: "Stopped: the voice is not enrolled.",
}

/** A row's summary text: the reply, truncated to one line by CSS at the
 * call site (this function returns the full string), or -- when there is
 * no reply -- the outcome mapping above, falling back to the raw
 * `turn_outcome` value verbatim. */
export function summarizeSessionOutcome(session: Pick<SessionSummary, "reply_text" | "turn_outcome">): string {
  if (session.reply_text) return session.reply_text
  return OUTCOME_SUMMARY[session.turn_outcome] ?? session.turn_outcome
}

/** "{N} ms end to end" (08-UI-SPEC.md Copywriting Contract), reusing
 * `lib/format.ts`'s shared millisecond formatter rather than a second one
 * -- `formatMs`'s own "not reached" branch composes into "not reached end
 * to end" for a session whose duration could not be measured. */
export function formatSessionDuration(durationMs: number | null): string {
  return `${formatMs(durationMs)} end to end`
}

/** Who spoke this turn, in the four states this plan (11-10) names.
 * `null` -- a session recorded before this phase, or one built with no
 * speaker context at all -- reads "Speaker not recorded", never a blank.
 * `detail` "speaker_id_off"/"not_edge_source" both mean speaker id took no
 * measurement at all (mode off, or a camera/browser turn -- D-07), so both
 * read the same "Speaker ID off" rather than the generic unknown-voice
 * copy, which would wrongly suggest a real, failed match attempt. */
export function formatSessionSpeaker(speaker: SessionSpeaker | null): string {
  if (speaker === null) return "Speaker not recorded"
  if (speaker.status === "identified") return speaker.speaker_name ?? "Unknown voice"
  if (speaker.detail === "speaker_id_off" || speaker.detail === "not_edge_source") return "Speaker ID off"
  return "Unknown voice"
}
