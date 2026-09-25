import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { EdgeDevice } from "@/lib/edgeDevices"
import { deriveEdgeDevicesScreenState } from "./deriveEdgeDevicesScreenState"

function device(overrides: Partial<EdgeDevice> = {}): EdgeDevice {
  return {
    id: 1,
    name: "kitchen",
    created_at: "2026-09-01T00:00:00Z",
    revoked: false,
    last_connected_at: null,
    connected: false,
    ...overrides,
  }
}

describe("deriveEdgeDevicesScreenState", () => {
  test("a pending query is loading", () => {
    expect(deriveEdgeDevicesScreenState({ devices: { status: "pending", data: undefined, error: undefined } })).toEqual({
      kind: "loading",
    })
  })

  test("a failed query surfaces the ApiError's own message", () => {
    const error = new ApiError(500, "Something went wrong.")
    expect(deriveEdgeDevicesScreenState({ devices: { status: "error", data: undefined, error } })).toEqual({
      kind: "error",
      message: "Something went wrong.",
    })
  })

  test("a non-ApiError failure gets the generic connection message", () => {
    expect(
      deriveEdgeDevicesScreenState({ devices: { status: "error", data: undefined, error: new Error("boom") } }),
    ).toEqual({ kind: "error", message: "Can't reach the server. Check your connection and try again." })
  })

  test("active devices sort connected-first, and revoked devices sink to the bottom", () => {
    const revokedButWasConnected = device({ id: 1, name: "revoked-but-was-connected", revoked: true, connected: true })
    const activeNotConnected = device({ id: 2, name: "active-not-connected", connected: false })
    const activeConnected = device({ id: 3, name: "active-connected", connected: true })

    const result = deriveEdgeDevicesScreenState({
      devices: {
        status: "success",
        data: [revokedButWasConnected, activeNotConnected, activeConnected],
        error: undefined,
      },
    })

    expect(result.kind).toBe("ready")
    if (result.kind !== "ready") throw new Error("unreachable")
    expect(result.devices.map((d) => d.name)).toEqual([
      "active-connected",
      "active-not-connected",
      "revoked-but-was-connected",
    ])
  })

  test("no devices is a ready state with an empty list", () => {
    expect(deriveEdgeDevicesScreenState({ devices: { status: "success", data: [], error: undefined } })).toEqual({
      kind: "ready",
      devices: [],
    })
  })
})
