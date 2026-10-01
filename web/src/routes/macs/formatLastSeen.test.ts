import { describe, expect, test } from "bun:test"
import { formatRelative, lastSeenLine } from "./formatLastSeen"

const NOW = new Date("2026-10-01T12:00:00Z")

function ago(ms: number): string {
  return new Date(NOW.getTime() - ms).toISOString()
}

const MIN = 60_000
const HOUR = 60 * MIN

describe("formatRelative", () => {
  test("under one minute is just now", () => {
    expect(formatRelative(ago(30_000), NOW)).toBe("just now")
  })
  test("one minute and n minutes", () => {
    expect(formatRelative(ago(MIN), NOW)).toBe("1 minute ago")
    expect(formatRelative(ago(5 * MIN), NOW)).toBe("5 minutes ago")
    expect(formatRelative(ago(59 * MIN), NOW)).toBe("59 minutes ago")
  })
  test("one hour and n hours", () => {
    expect(formatRelative(ago(HOUR), NOW)).toBe("1 hour ago")
    expect(formatRelative(ago(23 * HOUR), NOW)).toBe("23 hours ago")
  })
  test("24 hours or more is the local date", () => {
    const iso = ago(24 * HOUR)
    expect(formatRelative(iso, NOW)).toBe(new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(new Date(iso)))
  })
  test("a time in the future reads as just now", () => {
    expect(formatRelative(ago(-5 * MIN), NOW)).toBe("just now")
  })
})

describe("lastSeenLine", () => {
  test("connected shows Connected now with no title", () => {
    expect(lastSeenLine({ connected: true, last_seen_at: ago(HOUR) }, NOW)).toEqual({
      text: "Connected now",
      title: null,
    })
  })
  test("never seen shows Never connected", () => {
    expect(lastSeenLine({ connected: false, last_seen_at: null }, NOW)).toEqual({
      text: "Never connected",
      title: null,
    })
  })
  test("seen before shows the relative text and a full date title", () => {
    const iso = ago(5 * MIN)
    const line = lastSeenLine({ connected: false, last_seen_at: iso }, NOW)
    expect(line.text).toBe("Last seen 5 minutes ago")
    expect(line.title).toBe(
      new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(new Date(iso)),
    )
  })
})
