// The Macs page logic, apart from its JSX. This file imports nothing from
// React. It copies `deriveEdgeDevicesScreenState`.
import { ApiError } from "@/lib/api"
import type { DesktopDevice } from "@/lib/desktopDevices"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type DesktopDevicesScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; devices: DesktopDevice[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Cannot reach the server. Check your connection and try again."
}

/**
 * Active before revoked, online before offline, then by name. An equal
 * name falls back to the id, so the order never changes between renders.
 */
function sortDevices(devices: DesktopDevice[]): DesktopDevice[] {
  return [...devices].sort((a, b) => {
    if (a.revoked !== b.revoked) return a.revoked ? 1 : -1
    if (a.connected !== b.connected) return a.connected ? -1 : 1
    const byName = a.name.localeCompare(b.name)
    if (byName !== 0) return byName
    return a.id - b.id
  })
}

export function deriveDesktopDevicesScreenState(input: {
  devices: QueryLike<DesktopDevice[]>
}): DesktopDevicesScreenState {
  const { devices } = input

  if (devices.status === "pending") return { kind: "loading" }
  if (devices.status === "error") return { kind: "error", message: messageFor(devices.error) }

  return { kind: "ready", devices: sortDevices(devices.data ?? []) }
}
