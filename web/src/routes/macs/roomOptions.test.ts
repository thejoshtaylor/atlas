import { expect, test } from "bun:test"
import type { EdgeDevice } from "@/lib/edgeDevices"
import { roomChangeFor, roomOptions, selectedRoomValue } from "./roomOptions"

function edge(id: number, name: string, revoked = false): EdgeDevice {
  return { id, name, created_at: "2026-10-01T00:00:00Z", revoked, last_connected_at: null, connected: false }
}

test("no edge devices leaves only None", () => {
  expect(roomOptions([], null)).toEqual([{ value: "none", label: "None", disabled: false }])
})

test("active edge devices are listed after None, sorted by name", () => {
  const options = roomOptions([edge(2, "Bravo"), edge(1, "Alpha")], null)
  expect(options.map((option) => option.label)).toEqual(["None", "Alpha", "Bravo"])
  expect(options.map((option) => option.value)).toEqual(["none", "1", "2"])
  expect(options.every((option) => !option.disabled)).toBe(true)
})

test("a saved revoked edge device stays as a disabled revoked option", () => {
  const options = roomOptions([edge(1, "Alpha"), edge(9, "Hall", true)], 9)
  expect(options).toEqual([
    { value: "none", label: "None", disabled: false },
    { value: "1", label: "Alpha", disabled: false },
    { value: "9", label: "Hall (revoked)", disabled: true },
  ])
})

test("a revoked edge device that is not the saved one is left out", () => {
  const options = roomOptions([edge(1, "Alpha"), edge(9, "Hall", true)], 1)
  expect(options.map((option) => option.label)).toEqual(["None", "Alpha"])
  expect(roomOptions([edge(9, "Hall", true)], null).map((option) => option.label)).toEqual(["None"])
})

test("selectedRoomValue is none for a null room and the id as text otherwise", () => {
  expect(selectedRoomValue({ edge_device_id: null })).toBe("none")
  expect(selectedRoomValue({ edge_device_id: 7 })).toBe("7")
})

test("roomChangeFor maps none to null and an id to a number", () => {
  expect(roomChangeFor("none")).toEqual({ edge_device_id: null })
  expect(roomChangeFor("7")).toEqual({ edge_device_id: 7 })
})
