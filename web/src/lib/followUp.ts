// The follow-up window's own admin-editable length (D-09, plan 09-07):
// `GET`/`PUT /api/settings/follow-up-window` (`routes/follow_up_settings.py`).
// The same one-field GET/PUT shape `lib/wizard.ts`'s own
// `timezoneQueryOptions`/`setTimezoneMutationOptions` establish for the
// house time zone -- this module is that shape's sibling, not a second
// pattern.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `FollowUpWindowResponse`'s exact shape (`routes/follow_up_settings.py`). */
export interface FollowUpWindowStatus {
  window_s: number
  default_s: number
  resolved_from: "database" | "config"
}

export const FOLLOW_UP_WINDOW_QUERY_KEY = ["settings", "follow-up-window"] as const

export function fetchFollowUpWindow(): Promise<FollowUpWindowStatus> {
  return apiFetch<FollowUpWindowStatus>("/api/settings/follow-up-window")
}

export const followUpWindowQueryOptions = {
  queryKey: FOLLOW_UP_WINDOW_QUERY_KEY,
  queryFn: fetchFollowUpWindow,
}

export interface SaveFollowUpWindowInput {
  window_s: number
}

function saveFollowUpWindow(input: SaveFollowUpWindowInput): Promise<FollowUpWindowStatus> {
  return apiFetch<FollowUpWindowStatus>("/api/settings/follow-up-window", { method: "PUT", body: input })
}

/**
 * `PUT /api/settings/follow-up-window` -- `onSuccess` writes the returned
 * status straight into `FOLLOW_UP_WINDOW_QUERY_KEY` (the response already
 * carries everything a fresh `GET` would, per that route's own contract),
 * rather than invalidating and waiting on a second round trip -- the same
 * shape `lib/wizard.ts`'s own `setTimezoneMutationOptions` already uses.
 */
export const saveFollowUpWindowMutationOptions: UseMutationOptions<
  FollowUpWindowStatus,
  unknown,
  SaveFollowUpWindowInput
> = {
  mutationFn: saveFollowUpWindow,
  onSuccess: (status) => {
    queryClient.setQueryData(FOLLOW_UP_WINDOW_QUERY_KEY, status)
  },
}
