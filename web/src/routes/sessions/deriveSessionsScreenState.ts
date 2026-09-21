// SessionsRoute's own logic, kept apart from its JSX -- copied in shape
// from `routes/plugins/derivePluginsScreenState.ts`. Imports nothing from
// React, for the same stated reason: what belongs here is only ever a
// fact about data, never a fact about a render.
import { ApiError } from "@/lib/api"
import type { SessionSummary } from "@/lib/sessions"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

// Task 1's own scope: loading, error, ready. `derivePluginsScreenState.ts`
// has no empty member because the plugin list is never empty; the
// sessions list genuinely can be (a fresh install before the first turn)
// -- that branch is added by Task 3, deliberately after this one, so the
// divergence from the plugins analog reads as a decision, not a copy that
// drifted.
export type SessionsScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
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
  return { kind: "ready", sessions }
}
