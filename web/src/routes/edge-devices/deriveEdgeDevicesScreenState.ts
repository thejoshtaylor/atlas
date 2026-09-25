// EdgeDevicesRoute's own logic, kept apart from its JSX -- this project
// has no rendered-component test infrastructure for pure state
// derivation (see CalibrationRoute's own derive module, 03-06, and
// AccountsRoute's, 03-08, for the established reason and pattern this
// file copies). Imports nothing from React: what belongs here is only
// ever a fact about data, never a fact about a render.
import { ApiError } from "@/lib/api"
import type { EdgeDevice } from "@/lib/edgeDevices"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type EdgeDevicesScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; devices: EdgeDevice[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Can't reach the server. Check your connection and try again."
}

/**
 * Active devices first, connected ones first among those; a revoked
 * device sinks to the bottom regardless of whether it was connected the
 * moment before revocation -- it no longer competes with the devices
 * that still matter today.
 */
function sortDevices(devices: EdgeDevice[]): EdgeDevice[] {
  return [...devices].sort((a, b) => {
    if (a.revoked !== b.revoked) return a.revoked ? 1 : -1
    if (a.connected !== b.connected) return a.connected ? -1 : 1
    return 0
  })
}

export function deriveEdgeDevicesScreenState(input: { devices: QueryLike<EdgeDevice[]> }): EdgeDevicesScreenState {
  const { devices } = input

  if (devices.status === "pending") return { kind: "loading" }
  if (devices.status === "error") return { kind: "error", message: messageFor(devices.error) }

  return { kind: "ready", devices: sortDevices(devices.data ?? []) }
}
