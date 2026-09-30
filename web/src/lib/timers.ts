// The timers and alarms surface. Field names are copied verbatim from
// `src/atlas/routes/timers.py`'s models, matching `lib/edgeDevices.ts`'s own
// convention -- never renamed to a convention of this module's own invention.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

export type Weekday = "mon" | "tue" | "wed" | "thu" | "fri" | "sat" | "sun"

export const WEEKDAYS: readonly Weekday[] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]

/** `TimerResponse`'s exact shape. */
export interface Timer {
  id: number
  kind: "timer" | "alarm"
  label: string
  enabled: boolean
  paused: boolean
  duration_seconds: number | null
  remaining_seconds: number | null
  time: string | null
  days: Weekday[]
  next_fire_at: string | null
  created_at: string
}

export interface NewTimer {
  kind: "timer"
  label: string
  duration_seconds: number
}

export interface NewAlarm {
  kind: "alarm"
  label: string
  time: string
  days: Weekday[]
}

/** `TimerChanges`: give at least one field. */
export interface TimerChanges {
  label?: string
  remaining_seconds?: number
  add_seconds?: number
  paused?: boolean
  time?: string
  days?: Weekday[]
  enabled?: boolean
}

export const TIMERS_QUERY_KEY = ["timers"] as const

export function fetchTimers(): Promise<Timer[]> {
  return apiFetch<Timer[]>("/api/timers")
}

// A 5 second refetch so a timer that rang leaves the list on its own.
export const timersQueryOptions = {
  queryKey: TIMERS_QUERY_KEY,
  queryFn: fetchTimers,
  refetchInterval: 5000,
}

function invalidate() {
  void queryClient.invalidateQueries({ queryKey: TIMERS_QUERY_KEY })
}

export const createTimerMutationOptions: UseMutationOptions<Timer, unknown, NewTimer | NewAlarm> = {
  mutationFn: (input) => apiFetch<Timer>("/api/timers", { method: "POST", body: input }),
  onSuccess: invalidate,
}

export interface UpdateTimerInput {
  id: number
  changes: TimerChanges
}

export const updateTimerMutationOptions: UseMutationOptions<Timer, unknown, UpdateTimerInput> = {
  mutationFn: ({ id, changes }) => apiFetch<Timer>(`/api/timers/${id}`, { method: "PATCH", body: changes }),
  onSuccess: invalidate,
}

export interface DeleteTimerInput {
  id: number
}

export const deleteTimerMutationOptions: UseMutationOptions<void, unknown, DeleteTimerInput> = {
  mutationFn: ({ id }) => apiFetch<void>(`/api/timers/${id}`, { method: "DELETE" }),
  onSuccess: invalidate,
}
