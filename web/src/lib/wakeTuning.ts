// The wake-tuning surface's fetch layer (DBG-05, D-13, D-14a, D-15, D-16).
// Every shape here matches `src/atlas/routes/wake.py`'s own response
// models field for field, the same discipline `lib/plugins.ts` states at
// its own top of file -- a renamed or reshaped field here is a silent
// drift from what the server actually sends.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `GateDecision`'s own three named block reasons (`wake/gate.py`) as of
 * this phase. The three this screen recognises -- not a closed statement
 * about what the column can hold: `block_reason` below is `string | null`
 * on purpose, because the server sends whatever `BlockReason` holds at
 * the time and the column is nullable. A fourth reason added to the gate
 * later arrives here as a string this union does not name, and
 * `deriveWakeTuningScreenState.ts` is where that becomes a first-class
 * unknown rather than a wrong label. */
export type WakeEventBlockReason = "below_threshold" | "refractory" | "media_playing"

/** `WakeEventResponse`'s exact shape (`routes/wake.py`). `score` is
 * nullable for pre-migration data (D-16); for every row this phase's own
 * write path adds going forward it is always populated. */
export interface WakeEvent {
  id: number
  source: string
  engine: string
  score: number | null
  allowed: boolean
  block_reason: string | null
  recorded_at: string
}

/** `WakeEventsResponse`'s exact shape. `engine_grades`, `threshold` and
 * `not_scored_session_count` are read fresh at response-build time on the
 * server, never a stored copy -- this module carries them through
 * unmodified. */
export interface WakeEventsResponse {
  events: WakeEvent[]
  engine: string
  engine_grades: boolean
  threshold: number
  not_scored_session_count: number
  /** True when older wake attempts exist that this response does not
   * carry (`routes/wake.py::MAX_WAKE_EVENTS_IN_RESPONSE`). The screen
   * states it rather than presenting a partial history as a whole one. */
  capped: boolean
}

export const WAKE_EVENTS_QUERY_KEY = ["wake-events"] as const

export function fetchWakeEvents(): Promise<WakeEventsResponse> {
  return apiFetch<WakeEventsResponse>("/api/wake-events")
}

export interface SetWakeThresholdInput {
  threshold: number
}

/** `SetWakeThresholdResponse`'s exact shape. `applied_to_sources` is how
 * many running wake sources the change actually reached -- `0` means the
 * value is stored and will be picked up at the next boot, but nothing is
 * listening for it to take effect on right now. D-15 promises "takes
 * effect live", so the screen states when it did not. */
export interface SetWakeThresholdResult {
  threshold: number
  applied_to_sources: number
}

/**
 * `PUT /api/wake-threshold` -- the one write this screen makes (D-15: no
 * restart, applies live). `onSuccess` invalidates the events query so the
 * server's own live `threshold` field (and any event recorded in the
 * meantime) is re-fetched, matching `lib/plugins.ts`'s own
 * invalidate-on-write convention.
 */
export const setWakeThresholdMutationOptions: UseMutationOptions<
  SetWakeThresholdResult,
  unknown,
  SetWakeThresholdInput
> = {
  mutationFn: ({ threshold }) =>
    apiFetch<SetWakeThresholdResult>("/api/wake-threshold", { method: "PUT", body: { threshold } }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: WAKE_EVENTS_QUERY_KEY })
  },
}
