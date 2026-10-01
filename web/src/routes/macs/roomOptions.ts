// Pure rules for the Room dropdown on a Mac row (D-16). No React here, so
// the choice logic is proven in `roomOptions.test.ts` without driving a
// Radix Select under happy-dom (14-RESEARCH.md Pitfall 10).
import type { DesktopDevice } from "@/lib/desktopDevices"
import type { EdgeDevice } from "@/lib/edgeDevices"

export const NO_ROOM_VALUE = "none"

export interface RoomOption {
  value: string
  label: string
  disabled: boolean
}

/**
 * None first, then every active edge device by name. A saved edge device
 * that is now revoked stays visible as a disabled "{name} (revoked)" option,
 * so the Select can still show what is saved. Any other revoked device is
 * left out.
 */
export function roomOptions(edgeDevices: readonly EdgeDevice[], savedEdgeDeviceId: number | null): RoomOption[] {
  const active = edgeDevices
    .filter((edge) => !edge.revoked)
    .map((edge) => ({ value: String(edge.id), label: edge.name, disabled: false }))
    .sort((a, b) => a.label.localeCompare(b.label))
  const options: RoomOption[] = [{ value: NO_ROOM_VALUE, label: "None", disabled: false }, ...active]
  const saved = edgeDevices.find((edge) => edge.revoked && edge.id === savedEdgeDeviceId)
  if (saved) {
    options.push({ value: String(saved.id), label: `${saved.name} (revoked)`, disabled: true })
  }
  return options
}

export function selectedRoomValue(device: Pick<DesktopDevice, "edge_device_id">): string {
  return device.edge_device_id === null ? NO_ROOM_VALUE : String(device.edge_device_id)
}

export function roomChangeFor(value: string): { edge_device_id: number | null } {
  return { edge_device_id: value === NO_ROOM_VALUE ? null : Number(value) }
}
