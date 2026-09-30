// TimersRoute's own logic, kept apart from its JSX (the pattern of
// `deriveEdgeDevicesScreenState.ts`). Imports nothing from React.
import { ApiError } from "@/lib/api"
import type { Timer, Weekday } from "@/lib/timers"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type TimersScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; timers: Timer[]; alarms: Timer[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Can't reach the server. Check your connection and try again."
}

function byFireTime(a: Timer, b: Timer): number {
  return Date.parse(a.next_fire_at ?? "") - Date.parse(b.next_fire_at ?? "")
}

/** Running timers first, soonest first, then paused ones. */
function sortTimers(timers: Timer[]): Timer[] {
  const running = timers.filter((timer) => !timer.paused).sort(byFireTime)
  const paused = timers.filter((timer) => timer.paused)
  return [...running, ...paused]
}

/** Alarms by their clock time, then by id. */
function sortAlarms(alarms: Timer[]): Timer[] {
  return [...alarms].sort((a, b) => (a.time ?? "").localeCompare(b.time ?? "") || a.id - b.id)
}

export function deriveTimersScreenState(input: { timers: QueryLike<Timer[]> }): TimersScreenState {
  const { timers } = input
  if (timers.status === "pending") return { kind: "loading" }
  if (timers.status === "error") return { kind: "error", message: messageFor(timers.error) }
  const all = timers.data ?? []
  return {
    kind: "ready",
    timers: sortTimers(all.filter((entry) => entry.kind === "timer")),
    alarms: sortAlarms(all.filter((entry) => entry.kind === "alarm")),
  }
}

/** Whole seconds left. A paused timer keeps its stored remainder; a running
 * one counts down to `next_fire_at` against `nowMs` and stops at 0. */
export function secondsLeft(timer: Timer, nowMs: number): number {
  if (timer.paused || timer.next_fire_at === null) return timer.remaining_seconds ?? 0
  return Math.max(0, Math.ceil((Date.parse(timer.next_fire_at) - nowMs) / 1000))
}

/** 432 gives "7:12" and 3723 gives "1:02:03". */
export function formatCountdown(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds))
  const hours = Math.floor(seconds / 3600)
  const minutes = Math.floor((seconds % 3600) / 60)
  const rest = String(seconds % 60).padStart(2, "0")
  if (hours > 0) return `${hours}:${String(minutes).padStart(2, "0")}:${rest}`
  return `${minutes}:${rest}`
}

const WEEK: readonly Weekday[] = ["mon", "tue", "wed", "thu", "fri", "sat", "sun"]
const DAY_NAMES: Record<Weekday, string> = {
  mon: "Mon",
  tue: "Tue",
  wed: "Wed",
  thu: "Thu",
  fri: "Fri",
  sat: "Sat",
  sun: "Sun",
}

export function describeDays(days: readonly Weekday[]): string {
  const set = new Set(days)
  if (set.size === 0) return "Once"
  if (set.size === 7) return "Every day"
  if (set.size === 5 && WEEK.slice(0, 5).every((day) => set.has(day))) return "Weekdays"
  if (set.size === 2 && set.has("sat") && set.has("sun")) return "Weekends"
  return WEEK.filter((day) => set.has(day))
    .map((day) => DAY_NAMES[day])
    .join(", ")
}

/** "07:00" gives "7:00 AM". */
export function formatClock(time: string): string {
  const [hourText, minuteText] = time.split(":")
  const hour = Number(hourText)
  const suffix = hour < 12 ? "AM" : "PM"
  return `${hour % 12 || 12}:${minuteText} ${suffix}`
}
