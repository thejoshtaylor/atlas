import { describe, expect, test } from "bun:test"
import type { Timer } from "@/lib/timers"
import {
  deriveTimersScreenState,
  describeDays,
  formatClock,
  formatCountdown,
  secondsLeft,
} from "./deriveTimersScreenState"

function entry(overrides: Partial<Timer>): Timer {
  return {
    id: 1,
    kind: "timer",
    label: "",
    enabled: true,
    paused: false,
    duration_seconds: 600,
    remaining_seconds: 600,
    time: null,
    days: [],
    next_fire_at: "2027-01-01T12:10:00Z",
    created_at: "2027-01-01T12:00:00Z",
    ...overrides,
  }
}

describe("deriveTimersScreenState", () => {
  test("loading and error states", () => {
    expect(deriveTimersScreenState({ timers: { status: "pending", data: undefined, error: null } })).toEqual({
      kind: "loading",
    })
    const error = deriveTimersScreenState({ timers: { status: "error", data: undefined, error: new Error("x") } })
    expect(error.kind).toBe("error")
  })

  test("running timers sort by next_fire_at, paused ones come after, alarms sort by time", () => {
    const state = deriveTimersScreenState({
      timers: {
        status: "success",
        error: null,
        data: [
          entry({ id: 1, paused: true, next_fire_at: null, remaining_seconds: 30 }),
          entry({ id: 2, next_fire_at: "2027-01-01T12:20:00Z" }),
          entry({ id: 3, next_fire_at: "2027-01-01T12:05:00Z" }),
          entry({ id: 4, kind: "alarm", time: "09:00", next_fire_at: "2027-01-02T09:00:00Z" }),
          entry({ id: 5, kind: "alarm", time: "06:30", next_fire_at: "2027-01-02T06:30:00Z" }),
        ],
      },
    })
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.timers.map((timer) => timer.id)).toEqual([3, 2, 1])
    expect(state.alarms.map((alarm) => alarm.id)).toEqual([5, 4])
  })
})

describe("helpers", () => {
  test("secondsLeft counts a running timer down and keeps a paused remainder", () => {
    const now = Date.parse("2027-01-01T12:00:00Z")
    expect(secondsLeft(entry({}), now)).toBe(600)
    expect(secondsLeft(entry({}), now + 599_500)).toBe(1)
    expect(secondsLeft(entry({}), now + 700_000)).toBe(0)
    expect(secondsLeft(entry({ paused: true, next_fire_at: null, remaining_seconds: 42 }), now)).toBe(42)
  })

  test("formatCountdown", () => {
    expect(formatCountdown(432)).toBe("7:12")
    expect(formatCountdown(3723)).toBe("1:02:03")
    expect(formatCountdown(0)).toBe("0:00")
  })

  test("describeDays", () => {
    expect(describeDays([])).toBe("Once")
    expect(describeDays(["mon", "tue", "wed", "thu", "fri", "sat", "sun"])).toBe("Every day")
    expect(describeDays(["mon", "tue", "wed", "thu", "fri"])).toBe("Weekdays")
    expect(describeDays(["sun", "sat"])).toBe("Weekends")
    expect(describeDays(["wed", "mon"])).toBe("Mon, Wed")
  })

  test("formatClock", () => {
    expect(formatClock("07:00")).toBe("7:00 AM")
    expect(formatClock("00:30")).toBe("12:30 AM")
    expect(formatClock("12:00")).toBe("12:00 PM")
    expect(formatClock("18:45")).toBe("6:45 PM")
  })
})
