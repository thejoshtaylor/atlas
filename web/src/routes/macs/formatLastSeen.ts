// Last-seen text for a Mac row (D-17). Pure: the caller passes `now`.
import type { DesktopDevice } from "@/lib/desktopDevices"

const MINUTE_MS = 60_000
const HOUR_MS = 60 * MINUTE_MS
const DAY_MS = 24 * HOUR_MS

/** "just now", "n minutes ago", "n hours ago", then the local date. */
export function formatRelative(iso: string, now: Date): string {
  const then = new Date(iso)
  const elapsed = Math.max(0, now.getTime() - then.getTime())
  if (elapsed < MINUTE_MS) return "just now"
  if (elapsed < HOUR_MS) {
    const minutes = Math.floor(elapsed / MINUTE_MS)
    return minutes === 1 ? "1 minute ago" : `${minutes} minutes ago`
  }
  if (elapsed < DAY_MS) {
    const hours = Math.floor(elapsed / HOUR_MS)
    return hours === 1 ? "1 hour ago" : `${hours} hours ago`
  }
  return new Intl.DateTimeFormat(undefined, { dateStyle: "medium" }).format(then)
}

export function lastSeenLine(
  device: Pick<DesktopDevice, "connected" | "last_seen_at">,
  now: Date,
): { text: string; title: string | null } {
  if (device.connected) return { text: "Connected now", title: null }
  if (device.last_seen_at === null) return { text: "Never connected", title: null }
  return {
    text: `Last seen ${formatRelative(device.last_seen_at, now)}`,
    title: new Intl.DateTimeFormat(undefined, { dateStyle: "medium", timeStyle: "short" }).format(
      new Date(device.last_seen_at),
    ),
  }
}
