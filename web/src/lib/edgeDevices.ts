// The edge devices surface: pair, watch and revoke a Raspberry Pi
// microphone (D-03, D-15). Field names copied verbatim from
// `src/atlas/routes/edge_devices.py`'s response models, matching
// `lib/accounts.ts`'s own established convention -- never renamed to a
// convention of this module's own invention.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `EdgeDeviceResponse`'s exact shape -- never a token or a hash. */
export interface EdgeDevice {
  id: number
  name: string
  created_at: string
  revoked: boolean
  last_connected_at: string | null
  connected: boolean
}

/** `EdgeDeviceCreatedResponse`'s exact shape -- the one response that
 * ever carries the plaintext device token. */
export interface EdgeDeviceCreated {
  id: number
  name: string
  token: string
  created_at: string
}

export const EDGE_DEVICES_QUERY_KEY = ["edge-devices"] as const

export function fetchEdgeDevices(): Promise<EdgeDevice[]> {
  return apiFetch<EdgeDevice[]>("/api/edge-devices")
}

export const edgeDevicesQueryOptions = {
  queryKey: EDGE_DEVICES_QUERY_KEY,
  queryFn: fetchEdgeDevices,
}

export interface CreateEdgeDeviceInput {
  name: string
}

/**
 * Creates a device and returns the plaintext token exactly once
 * (`EdgeDeviceCreatedResponse`) -- the caller (`EdgeDevicesRoute`) is
 * responsible for presenting it as something to copy and never fetching
 * it again, since no route will ever return it a second time. The
 * created token lives only in the caller's own local state: this
 * mutation's `onSuccess` invalidates the list query but never writes the
 * token into it, and no store holds it either.
 */
export const createEdgeDeviceMutationOptions: UseMutationOptions<
  EdgeDeviceCreated,
  unknown,
  CreateEdgeDeviceInput
> = {
  mutationFn: (input) => apiFetch<EdgeDeviceCreated>("/api/edge-devices", { method: "POST", body: input }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: EDGE_DEVICES_QUERY_KEY })
  },
}

export interface RevokeEdgeDeviceInput {
  deviceId: number
}

export const revokeEdgeDeviceMutationOptions: UseMutationOptions<void, unknown, RevokeEdgeDeviceInput> = {
  mutationFn: ({ deviceId }) => apiFetch<void>(`/api/edge-devices/${deviceId}`, { method: "DELETE" }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: EDGE_DEVICES_QUERY_KEY })
  },
}
