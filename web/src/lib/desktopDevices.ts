// The Macs surface: pair and watch a Mac that runs the ATLAS menu bar app
// (PAIR-01, PAIR-04). Field names are copied verbatim from the response
// models in `src/atlas/routes/desktop_devices.py`.
//
// The plaintext token never enters the query cache. The create mutation
// only invalidates the list. The route keeps the token in its own state.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `DesktopDeviceResponse`'s exact shape. It never carries a token. */
export interface DesktopDevice {
  id: number
  name: string
  created_at: string
  revoked: boolean
  last_seen_at: string | null
  connected: boolean
  edge_device_id: number | null
  is_default: boolean
}

/** The one response that carries the plaintext token. */
export interface DesktopDeviceCreated {
  id: number
  name: string
  token: string
  created_at: string
}

export const DESKTOP_DEVICES_QUERY_KEY = ["desktop-devices"] as const

export function fetchDesktopDevices(): Promise<DesktopDevice[]> {
  return apiFetch<DesktopDevice[]>("/api/desktop-devices")
}

// The page refetches every 5 seconds, so an online change shows within
// about 5 seconds (D-17). `timersQueryOptions` does the same.
export const desktopDevicesQueryOptions = {
  queryKey: DESKTOP_DEVICES_QUERY_KEY,
  queryFn: fetchDesktopDevices,
  refetchInterval: 5000,
}

export interface CreateDesktopDeviceInput {
  name: string
}

/**
 * Creates a Mac and returns its plaintext token once. The caller keeps the
 * token in local state. `onSuccess` only invalidates the list.
 */
export const createDesktopDeviceMutationOptions: UseMutationOptions<
  DesktopDeviceCreated,
  unknown,
  CreateDesktopDeviceInput
> = {
  mutationFn: (input) => apiFetch<DesktopDeviceCreated>("/api/desktop-devices", { method: "POST", body: input }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: DESKTOP_DEVICES_QUERY_KEY })
  },
}

/** The pair link the Mac app opens (D-03). Both values are encoded. */
export function buildPairLink(host: string, token: string): string {
  return `atlas://pair?server=${encodeURIComponent(host)}&token=${encodeURIComponent(token)}`
}

/** The browser origin the Mac must reach (D-07). A test replaces this function. */
export function pageLocation(): { host: string; protocol: string } {
  return { host: window.location.host, protocol: window.location.protocol }
}

export interface UpdateDesktopDeviceInput {
  deviceId: number
  /** Only the keys given are sent, so one control never overwrites another. */
  changes: { name?: string; edge_device_id?: number | null; is_default?: boolean }
}

/** Renames a Mac, maps it to a room, or sets it as the default (PAIR-03). */
export const updateDesktopDeviceMutationOptions: UseMutationOptions<
  DesktopDevice,
  unknown,
  UpdateDesktopDeviceInput
> = {
  mutationFn: ({ deviceId, changes }) =>
    apiFetch<DesktopDevice>(`/api/desktop-devices/${deviceId}`, { method: "PATCH", body: changes }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: DESKTOP_DEVICES_QUERY_KEY })
  },
}

export interface RevokeDesktopDeviceInput {
  deviceId: number
}

export const revokeDesktopDeviceMutationOptions: UseMutationOptions<void, unknown, RevokeDesktopDeviceInput> = {
  mutationFn: ({ deviceId }) => apiFetch<void>(`/api/desktop-devices/${deviceId}`, { method: "DELETE" }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: DESKTOP_DEVICES_QUERY_KEY })
  },
}

export interface TestDesktopDeviceInput {
  deviceId: number
}

/** `answered` is false when the Mac gave no answer inside the server's 5 second wait. */
export interface TestDesktopDeviceResult {
  answered: boolean
  rtt_ms: number | null
}

// A test changes nothing the list shows, so it does not invalidate (D-15).
export const testDesktopDeviceMutationOptions: UseMutationOptions<
  TestDesktopDeviceResult,
  unknown,
  TestDesktopDeviceInput
> = {
  mutationFn: ({ deviceId }) =>
    apiFetch<TestDesktopDeviceResult>(`/api/desktop-devices/${deviceId}/test`, { method: "POST" }),
}
