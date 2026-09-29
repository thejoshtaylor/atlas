// The voice inbox and retroactive clips (quick task 260929-j08). Field names
// are copied verbatim from `src/atlas/routes/speaker_inbox.py`. This is a
// module of its own so that `lib/speakers.ts` keeps its export list.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"
import { SPEAKERS_QUERY_KEY } from "./speakers"

/** `VoiceInboxItemResponse`'s exact shape. The transcript is untrusted text. */
export interface VoiceInboxItem {
  session_id: string
  started_at: string
  transcript: string | null
  speech_ms: number | null
  score: number | null
  blocked: boolean
}

/** `VoiceInboxResponse`'s exact shape. */
export interface VoiceInbox {
  items: VoiceInboxItem[]
}

export const VOICE_INBOX_QUERY_KEY = ["speakers", "voice-inbox"] as const

export function fetchVoiceInbox(): Promise<VoiceInbox> {
  return apiFetch<VoiceInbox>("/api/speakers/voice-inbox")
}

export const voiceInboxQueryOptions = {
  queryKey: VOICE_INBOX_QUERY_KEY,
  queryFn: fetchVoiceInbox,
}

// The audio URL helpers are plain paths for an `<audio src>`. The element
// makes its own request and carries the session cookie. This is the same
// named exception as `sessionAudioUrl` in `lib/sessions.ts`.
export function voiceInboxAudioUrl(sessionId: string): string {
  return `/api/speakers/voice-inbox/${encodeURIComponent(sessionId)}/audio`
}

export type AssignTarget = { kind: "existing"; speakerId: number } | { kind: "new"; displayName: string }

export interface AssignVoiceInput {
  sessionId: string
  target: AssignTarget
}

/** `VoiceAssignResponse`'s exact shape. */
export interface AssignVoiceResult {
  speaker_id: number
  phrase_index: number
  speech_ms: number
  dropped_phrase_indices: number[]
}

// Both mutations invalidate `["speakers"]`. The prefix match refreshes the
// member list, the inbox, and every clip list together.
export const assignVoiceMutationOptions: UseMutationOptions<AssignVoiceResult, unknown, AssignVoiceInput> = {
  mutationFn: ({ sessionId, target }) =>
    apiFetch<AssignVoiceResult>(`/api/speakers/voice-inbox/${encodeURIComponent(sessionId)}/assign`, {
      method: "POST",
      body: target.kind === "existing" ? { speaker_id: target.speakerId } : { new_speaker_name: target.displayName },
    }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: SPEAKERS_QUERY_KEY })
  },
}

/** `RetroactiveClipResponse`'s exact shape. */
export interface RetroactiveClip {
  phrase_index: number
  session_id: string | null
  speech_ms: number | null
  created_at: string | null
}

/** `RetroactiveClipsResponse`'s exact shape. */
export interface RetroactiveClipList {
  clips: RetroactiveClip[]
}

export function retroactiveClipsQueryKey(speakerId: number) {
  return ["speakers", speakerId, "retroactive-clips"] as const
}

export function retroactiveClipsQueryOptions(speakerId: number) {
  return {
    queryKey: retroactiveClipsQueryKey(speakerId),
    queryFn: () => apiFetch<RetroactiveClipList>(`/api/speakers/${speakerId}/retroactive-clips`),
  }
}

export function retroactiveClipAudioUrl(speakerId: number, phraseIndex: number): string {
  return `/api/speakers/${speakerId}/retroactive-clips/${phraseIndex}/audio`
}

export interface DeleteRetroactiveClipInput {
  speakerId: number
  phraseIndex: number
}

export const deleteRetroactiveClipMutationOptions: UseMutationOptions<void, unknown, DeleteRetroactiveClipInput> = {
  mutationFn: ({ speakerId, phraseIndex }) =>
    apiFetch<void>(`/api/speakers/${speakerId}/retroactive-clips/${phraseIndex}`, { method: "DELETE" }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: SPEAKERS_QUERY_KEY })
  },
}
