// The household-members surface: list, add, enroll and delete a speaker
// (D-01 through D-04). Field names copied verbatim from
// `src/atlas/routes/speakers.py`'s response models, matching
// `lib/edgeDevices.ts`'s own established convention -- never renamed to a
// convention of this module's own invention.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `SpeakerResponse`'s exact shape. */
export interface Speaker {
  id: number
  display_name: string
  linked_user_id: number | null
  created_at: string
  enrolled_phrases: number
  required_phrases: number
  model_id: string | null
}

export const SPEAKERS_QUERY_KEY = ["speakers"] as const

export function fetchSpeakers(): Promise<Speaker[]> {
  return apiFetch<Speaker[]>("/api/speakers")
}

export const speakersQueryOptions = {
  queryKey: SPEAKERS_QUERY_KEY,
  queryFn: fetchSpeakers,
}

export interface CreateSpeakerInput {
  displayName: string
  linkedUserId: number | null
}

export const createSpeakerMutationOptions: UseMutationOptions<Speaker, unknown, CreateSpeakerInput> = {
  mutationFn: (input) =>
    apiFetch<Speaker>("/api/speakers", {
      method: "POST",
      body: { display_name: input.displayName, linked_user_id: input.linkedUserId },
    }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: SPEAKERS_QUERY_KEY })
  },
}

export interface DeleteSpeakerInput {
  speakerId: number
}

export const deleteSpeakerMutationOptions: UseMutationOptions<void, unknown, DeleteSpeakerInput> = {
  mutationFn: ({ speakerId }) => apiFetch<void>(`/api/speakers/${speakerId}`, { method: "DELETE" }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: SPEAKERS_QUERY_KEY })
  },
}

/** `EnrollmentPhrasesResponse`'s exact shape. */
export interface EnrollmentPhrases {
  phrases: string[]
}

export const ENROLLMENT_PHRASES_QUERY_KEY = ["speakers", "enrollment-phrases"] as const

export function fetchEnrollmentPhrases(): Promise<EnrollmentPhrases> {
  return apiFetch<EnrollmentPhrases>("/api/speakers/enrollment-phrases")
}

export const enrollmentPhrasesQueryOptions = {
  queryKey: ENROLLMENT_PHRASES_QUERY_KEY,
  queryFn: fetchEnrollmentPhrases,
}

/** `EnrollmentResponse`'s exact shape -- used by `EnrollmentPanel.tsx` (Task 2). */
export interface EnrollmentResult {
  speaker_id: number
  phrase_index: number
  speech_ms: number
  enrolled_phrases: number
  required_phrases: number
}

export interface EnrollPhraseInput {
  speakerId: number
  phraseIndex: number
  deviceId: number
}

/**
 * Captures one prompted phrase through the real edge microphone (D-02) --
 * used by `EnrollmentPanel.tsx` (Task 2). Every write here invalidates the
 * speakers list, same as create/delete above, so the enrollment progress
 * shown on `SpeakersRoute.tsx` updates without a manual refetch.
 */
export const enrollPhraseMutationOptions: UseMutationOptions<EnrollmentResult, unknown, EnrollPhraseInput> = {
  mutationFn: ({ speakerId, phraseIndex, deviceId }) =>
    apiFetch<EnrollmentResult>(`/api/speakers/${speakerId}/enrollment/${phraseIndex}`, {
      method: "POST",
      body: { device_id: deviceId },
    }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: SPEAKERS_QUERY_KEY })
  },
}
