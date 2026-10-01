import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { DesktopDevice } from "@/lib/desktopDevices"
import { deriveDesktopDevicesScreenState } from "./deriveDesktopDevicesScreenState"

function device(overrides: Partial<DesktopDevice> = {}): DesktopDevice {
  return {
    id: 1,
    name: "desk",
    created_at: "2026-10-01T00:00:00Z",
    revoked: false,
    last_seen_at: null,
    connected: false,
    edge_device_id: null,
    is_default: false,
    ...overrides,
  }
}

function ready(data: DesktopDevice[]) {
  const result = deriveDesktopDevicesScreenState({ devices: { status: "success", data, error: undefined } })
  if (result.kind !== "ready") throw new Error("unreachable")
  return result.devices
}

describe("deriveDesktopDevicesScreenState", () => {
  test("a pending query is loading", () => {
    expect(
      deriveDesktopDevicesScreenState({ devices: { status: "pending", data: undefined, error: undefined } }),
    ).toEqual({ kind: "loading" })
  })

  test("a failed query shows the ApiError message", () => {
    const error = new ApiError(500, "Something went wrong.")
    expect(deriveDesktopDevicesScreenState({ devices: { status: "error", data: undefined, error } })).toEqual({
      kind: "error",
      message: "Something went wrong.",
    })
  })

  test("any other failure shows the connection message", () => {
    expect(
      deriveDesktopDevicesScreenState({ devices: { status: "error", data: undefined, error: new Error("boom") } }),
    ).toEqual({ kind: "error", message: "Cannot reach the server. Check your connection and try again." })
  })

  test("active before revoked, online first, then by name", () => {
    const names = ready([
      device({ id: 1, name: "zed-revoked", revoked: true, connected: true }),
      device({ id: 2, name: "bravo", connected: false }),
      device({ id: 3, name: "zulu", connected: true }),
      device({ id: 4, name: "alpha", connected: false }),
      device({ id: 5, name: "yankee", connected: true }),
    ]).map((d) => d.name)
    expect(names).toEqual(["yankee", "zulu", "alpha", "bravo", "zed-revoked"])
  })

  test("equal names fall back to the id", () => {
    const ids = ready([device({ id: 9, name: "same" }), device({ id: 3, name: "same" }), device({ id: 5, name: "same" })]).map(
      (d) => d.id,
    )
    expect(ids).toEqual([3, 5, 9])
  })

  test("no devices is a ready state with an empty list", () => {
    expect(ready([])).toEqual([])
  })
})
