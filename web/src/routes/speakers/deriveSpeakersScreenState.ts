// SpeakersRoute's own logic, kept apart from its JSX -- copies
// `deriveEdgeDevicesScreenState.ts`'s established pattern. Imports
// nothing from React: what belongs here is only ever a fact about data,
// never a fact about a render.
import { ApiError } from "@/lib/api"
import type { Speaker } from "@/lib/speakers"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type SpeakersScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; speakers: Speaker[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Can't reach the server. Check your connection and try again."
}

export function deriveSpeakersScreenState(input: { speakers: QueryLike<Speaker[]> }): SpeakersScreenState {
  const { speakers } = input

  if (speakers.status === "pending") return { kind: "loading" }
  if (speakers.status === "error") return { kind: "error", message: messageFor(speakers.error) }

  return { kind: "ready", speakers: speakers.data ?? [] }
}

/**
 * "3 of 5 phrases" while enrollment is short of `required_phrases`,
 * "Enrolled" once it reaches it, or "Speaker ID model not set" when
 * `model_id` is null -- a count against no model would be spurious.
 */
export function formatEnrollmentProgress(speaker: Speaker): string {
  if (speaker.model_id === null) return "Speaker ID model not set"
  if (speaker.enrolled_phrases >= speaker.required_phrases) return "Enrolled"
  return `${speaker.enrolled_phrases} of ${speaker.required_phrases} phrases`
}
