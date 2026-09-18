import { beforeEach, describe, expect, test } from "bun:test"
import type { Macro } from "@/lib/macros"
import {
  draftActionFrom,
  draftActionsToInput,
  isFirstAction,
  isLastAction,
  saveBlockedByEmptyActions,
  useMacroDraftStore,
  ZERO_ACTIONS_SAVE_BLOCKED_REASON,
  type DraftAction,
} from "./macroDraftStore"

function sampleMacro(overrides: Partial<Macro> = {}): Macro {
  return {
    id: 1,
    phrase: "goodnight",
    aliases: ["night night"],
    reply: "Good night.",
    actions: [
      { id: 10, position: 0, tool: "ha_call_service", arguments: { domain: "light", service: "turn_off", entity_id: "light.bedroom" }, conflict: "ok" },
    ],
    created_at: "2026-09-18T00:00:00Z",
    updated_at: "2026-09-18T00:00:00Z",
    created_by_user_id: 1,
    reply_cached: true,
    reply_synthesis_degraded: false,
    reply_synthesis_message: null,
    ...overrides,
  }
}

beforeEach(() => {
  useMacroDraftStore.getState().loadBlank()
})

describe("loadMacro / loadBlank -- server data seeds the draft, and only these two actions ever do", () => {
  test("loadMacro copies phrase, reply, actions and aliases, and starts clean (not dirty)", () => {
    useMacroDraftStore.getState().loadMacro(sampleMacro())
    const state = useMacroDraftStore.getState()
    expect(state.phrase).toBe("goodnight")
    expect(state.reply).toBe("Good night.")
    expect(state.aliases).toEqual(["night night"])
    expect(state.actions).toEqual([
      { key: "10", tool: "ha_call_service", domain: "light", service: "turn_off", entityId: "light.bedroom" },
    ])
    expect(state.dirty).toBe(false)
  })

  test("loadBlank clears everything for a new, unsaved macro", () => {
    useMacroDraftStore.getState().loadMacro(sampleMacro())
    useMacroDraftStore.getState().loadBlank()
    const state = useMacroDraftStore.getState()
    expect(state.phrase).toBe("")
    expect(state.actions).toEqual([])
    expect(state.aliases).toEqual([])
    expect(state.dirty).toBe(false)
  })
})

describe("action reorder -- moving the first item up and the last item down are no-ops", () => {
  test("moving the first action up leaves the list, and dirty, unchanged", () => {
    useMacroDraftStore.getState().loadMacro(
      sampleMacro({
        actions: [
          { id: 1, position: 0, tool: "ha_call_service", arguments: {}, conflict: "ok" },
          { id: 2, position: 1, tool: "ha_call_service", arguments: {}, conflict: "ok" },
        ],
      }),
    )
    const before = useMacroDraftStore.getState().actions
    useMacroDraftStore.getState().moveActionUp("1")
    const after = useMacroDraftStore.getState()
    expect(after.actions).toEqual(before)
    expect(after.dirty).toBe(false)
  })

  test("moving the last action down leaves the list, and dirty, unchanged", () => {
    useMacroDraftStore.getState().loadMacro(
      sampleMacro({
        actions: [
          { id: 1, position: 0, tool: "ha_call_service", arguments: {}, conflict: "ok" },
          { id: 2, position: 1, tool: "ha_call_service", arguments: {}, conflict: "ok" },
        ],
      }),
    )
    const before = useMacroDraftStore.getState().actions
    useMacroDraftStore.getState().moveActionDown("2")
    const after = useMacroDraftStore.getState()
    expect(after.actions).toEqual(before)
    expect(after.dirty).toBe(false)
  })

  test("moving a middle action up swaps it with its predecessor and marks the draft dirty", () => {
    useMacroDraftStore.getState().loadMacro(
      sampleMacro({
        actions: [
          { id: 1, position: 0, tool: "t", arguments: {}, conflict: "ok" },
          { id: 2, position: 1, tool: "t", arguments: {}, conflict: "ok" },
          { id: 3, position: 2, tool: "t", arguments: {}, conflict: "ok" },
        ],
      }),
    )
    useMacroDraftStore.getState().moveActionUp("2")
    const state = useMacroDraftStore.getState()
    expect(state.actions.map((a) => a.key)).toEqual(["2", "1", "3"])
    expect(state.dirty).toBe(true)
  })

  test("moving a middle action down swaps it with its successor", () => {
    useMacroDraftStore.getState().loadMacro(
      sampleMacro({
        actions: [
          { id: 1, position: 0, tool: "t", arguments: {}, conflict: "ok" },
          { id: 2, position: 1, tool: "t", arguments: {}, conflict: "ok" },
          { id: 3, position: 2, tool: "t", arguments: {}, conflict: "ok" },
        ],
      }),
    )
    useMacroDraftStore.getState().moveActionDown("2")
    expect(useMacroDraftStore.getState().actions.map((a) => a.key)).toEqual(["1", "3", "2"])
  })
})

describe("add / remove action -- edits the draft only, dirties it, contacts no server", () => {
  test("addAction appends a blank ha_call_service action and dirties the draft", () => {
    useMacroDraftStore.getState().addAction()
    const state = useMacroDraftStore.getState()
    expect(state.actions).toHaveLength(1)
    expect(state.actions[0]).toMatchObject({ tool: "ha_call_service", domain: "", service: "", entityId: "" })
    expect(state.dirty).toBe(true)
  })

  test("removeAction drops exactly the targeted action by key", () => {
    useMacroDraftStore.getState().loadMacro(sampleMacro())
    useMacroDraftStore.getState().addAction()
    const secondKey = useMacroDraftStore.getState().actions[1].key
    useMacroDraftStore.getState().removeAction(secondKey)
    expect(useMacroDraftStore.getState().actions.map((a) => a.key)).toEqual(["10"])
  })

  test("updateAction patches only the named field", () => {
    useMacroDraftStore.getState().addAction()
    const key = useMacroDraftStore.getState().actions[0].key
    useMacroDraftStore.getState().updateAction(key, { entityId: "light.kitchen" })
    expect(useMacroDraftStore.getState().actions[0]).toMatchObject({ entityId: "light.kitchen", domain: "" })
  })
})

describe("aliases -- an unordered add/remove list, no move controls", () => {
  test("addAlias appends a trimmed, non-duplicate alias and dirties the draft", () => {
    useMacroDraftStore.getState().addAlias("  night night  ")
    expect(useMacroDraftStore.getState().aliases).toEqual(["night night"])
    expect(useMacroDraftStore.getState().dirty).toBe(true)
  })

  test("addAlias is a no-op for a blank or already-present alias", () => {
    useMacroDraftStore.getState().addAlias("night night")
    useMacroDraftStore.getState().addAlias("night night")
    useMacroDraftStore.getState().addAlias("   ")
    expect(useMacroDraftStore.getState().aliases).toEqual(["night night"])
  })

  test("removeAlias drops exactly the named alias", () => {
    useMacroDraftStore.getState().addAlias("night night")
    useMacroDraftStore.getState().addAlias("lights out")
    useMacroDraftStore.getState().removeAlias("night night")
    expect(useMacroDraftStore.getState().aliases).toEqual(["lights out"])
  })
})

describe("isFirstAction / isLastAction -- the boundary facts the reorder controls disable on", () => {
  const actions: DraftAction[] = [
    { key: "a", tool: "t", domain: "", service: "", entityId: "" },
    { key: "b", tool: "t", domain: "", service: "", entityId: "" },
    { key: "c", tool: "t", domain: "", service: "", entityId: "" },
  ]

  test("the first row is first, and nothing else is", () => {
    expect(isFirstAction(actions, "a")).toBe(true)
    expect(isFirstAction(actions, "b")).toBe(false)
    expect(isFirstAction(actions, "c")).toBe(false)
  })

  test("the last row is last, and nothing else is", () => {
    expect(isLastAction(actions, "c")).toBe(true)
    expect(isLastAction(actions, "b")).toBe(false)
    expect(isLastAction(actions, "a")).toBe(false)
  })

  test("a single-action list is both first and last -- both controls stay disabled", () => {
    const single: DraftAction[] = [{ key: "only", tool: "t", domain: "", service: "", entityId: "" }]
    expect(isFirstAction(single, "only")).toBe(true)
    expect(isLastAction(single, "only")).toBe(true)
  })
})

describe("saveBlockedByEmptyActions -- the exact Copywriting Contract string, only when the list is empty", () => {
  test("empty list blocks save with the named reason", () => {
    expect(saveBlockedByEmptyActions([])).toBe(ZERO_ACTIONS_SAVE_BLOCKED_REASON)
  })

  test("a non-empty list does not block save", () => {
    expect(saveBlockedByEmptyActions([{ key: "a", tool: "t", domain: "", service: "", entityId: "" }])).toBeNull()
  })
})

describe("draftActionFrom -- loading a saved action tolerates arguments missing any of the three fields", () => {
  test("a non-ha_call_service-shaped action loads with blank domain/service/entity fields, not a crash", () => {
    const draft = draftActionFrom({ id: 5, tool: "some_other_tool", arguments: { unrelated: "value" } })
    expect(draft).toEqual({ key: "5", tool: "some_other_tool", domain: "", service: "", entityId: "" })
  })
})

describe("draftActionsToInput -- the draft's three fields fold back into arguments.{domain,service,entity_id}", () => {
  test("round-trips into the exact shape routes/macros.py's MacroActionInput expects", () => {
    const actions: DraftAction[] = [
      { key: "1", tool: "ha_call_service", domain: "light", service: "turn_on", entityId: "light.bedroom" },
    ]
    expect(draftActionsToInput(actions)).toEqual([
      { tool: "ha_call_service", arguments: { domain: "light", service: "turn_on", entity_id: "light.bedroom" } },
    ])
  })
})
