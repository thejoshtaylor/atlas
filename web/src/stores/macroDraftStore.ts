import { create } from "zustand"
import type { Macro, MacroActionInput } from "@/lib/macros"

// The macro editor's unsaved draft -- phrase, reply, an ordered action
// list, and an unordered alias list. This is the one store in this app
// that models an ordered, reorderable list (04-07's own key link), so it
// is built fresh against the Zustand conventions `policyDraftStore.ts`
// and `wizardStore.ts` already set rather than copied from either --
// neither models this shape. Whether the up control is disabled on the
// first row, or the down control on the last, is a fact about this list
// computed here, not in JSX: a fact computed in JSX is a fact this
// project cannot test, because there is no rendered-DOM test
// infrastructure here.
//
// The loaded macro itself (what the server has on record) is server
// state and lives in a query cache, never here -- `loadMacro`/
// `loadBlank` are the only two ways this store's fields are ever set
// from outside an operator's own typing, matching `wizardStore.ts`'s own
// rule that server data never lands in a draft store.

export interface DraftAction {
  /** A stable client-side key for React list rendering and for
   * targeting remove/move-up/move-down -- never sent to the server. An
   * existing action's own `id` (stringified) once loaded from a saved
   * macro, or a locally-minted key for one added in this session. */
  key: string
  tool: string
  domain: string
  service: string
  entityId: string
}

interface MacroDraftState {
  phrase: string
  reply: string
  actions: DraftAction[]
  aliases: string[]
  /** Set by any add/remove/reorder/update, cleared by `loadMacro`/
   * `loadBlank` -- reordering, adding or removing an action marks the
   * draft dirty without contacting the server (Task 1's own behaviour). */
  dirty: boolean

  loadMacro: (macro: Macro) => void
  loadBlank: () => void

  setPhrase: (phrase: string) => void
  setReply: (reply: string) => void

  addAction: () => void
  removeAction: (key: string) => void
  moveActionUp: (key: string) => void
  moveActionDown: (key: string) => void
  updateAction: (key: string, patch: Partial<Omit<DraftAction, "key">>) => void

  addAlias: (alias: string) => void
  removeAlias: (alias: string) => void
}

let nextDraftKey = 0
function freshKey(): string {
  nextDraftKey += 1
  return `draft-${nextDraftKey}`
}

function stringField(args: Record<string, unknown>, field: string): string {
  const value = args[field]
  return typeof value === "string" ? value : ""
}

export function draftActionFrom(action: { id: number; tool: string; arguments: Record<string, unknown> }): DraftAction {
  return {
    key: String(action.id),
    tool: action.tool,
    domain: stringField(action.arguments ?? {}, "domain"),
    service: stringField(action.arguments ?? {}, "service"),
    entityId: stringField(action.arguments ?? {}, "entity_id"),
  }
}

export const useMacroDraftStore = create<MacroDraftState>((set) => ({
  phrase: "",
  reply: "",
  actions: [],
  aliases: [],
  dirty: false,

  loadMacro: (macro) =>
    set({
      phrase: macro.phrase,
      reply: macro.reply,
      actions: macro.actions.map(draftActionFrom),
      aliases: [...macro.aliases],
      dirty: false,
    }),

  loadBlank: () => set({ phrase: "", reply: "", actions: [], aliases: [], dirty: false }),

  setPhrase: (phrase) => set({ phrase }),
  setReply: (reply) => set({ reply }),

  addAction: () =>
    set((state) => ({
      actions: [...state.actions, { key: freshKey(), tool: "ha_call_service", domain: "", service: "", entityId: "" }],
      dirty: true,
    })),

  removeAction: (key) =>
    set((state) => ({
      actions: state.actions.filter((action) => action.key !== key),
      dirty: true,
    })),

  // Moving the first item up, or the last item down, is a no-op: the
  // state object returned is unchanged, so no re-render and no dirty
  // flip happens for a boundary move that changed nothing.
  moveActionUp: (key) =>
    set((state) => {
      const index = state.actions.findIndex((action) => action.key === key)
      if (index <= 0) return state
      const actions = [...state.actions]
      const [item] = actions.splice(index, 1)
      actions.splice(index - 1, 0, item)
      return { actions, dirty: true }
    }),

  moveActionDown: (key) =>
    set((state) => {
      const index = state.actions.findIndex((action) => action.key === key)
      if (index === -1 || index >= state.actions.length - 1) return state
      const actions = [...state.actions]
      const [item] = actions.splice(index, 1)
      actions.splice(index + 1, 0, item)
      return { actions, dirty: true }
    }),

  updateAction: (key, patch) =>
    set((state) => ({
      actions: state.actions.map((action) => (action.key === key ? { ...action, ...patch } : action)),
      dirty: true,
    })),

  addAlias: (alias) =>
    set((state) => {
      const trimmed = alias.trim()
      if (!trimmed || state.aliases.includes(trimmed)) return state
      return { aliases: [...state.aliases, trimmed], dirty: true }
    }),

  removeAlias: (alias) =>
    set((state) => ({
      aliases: state.aliases.filter((existing) => existing !== alias),
      dirty: true,
    })),
}))

/** A fact about the list, not about any one row -- used to disable the
 * up control at the top and the down control at the bottom, matching
 * every other disabled-at-boundary control in this app (Copywriting
 * Contract's reorder-control row). Pure functions rather than store
 * methods so they stay trivially testable and never go stale relative
 * to a render. */
export function isFirstAction(actions: DraftAction[], key: string): boolean {
  return actions.length > 0 && actions[0].key === key
}

export function isLastAction(actions: DraftAction[], key: string): boolean {
  return actions.length > 0 && actions[actions.length - 1].key === key
}

/** UI-SPEC's exact "Error state -- zero actions on save" copy -- owned
 * here because it is a fact about the action list this store holds, so
 * `MacroEditorRoute.tsx` and this file's own tests assert on the same
 * string rather than a paraphrase of it. */
export const ZERO_ACTIONS_SAVE_BLOCKED_REASON = "Add at least one action before saving."

/** The draft store reports save as blocked when the action list is
 * empty, and reports the reason (Task 1's own behaviour) -- a pure
 * function over the list rather than a redundant boolean field that
 * could drift from `actions` itself. */
export function saveBlockedByEmptyActions(actions: DraftAction[]): string | null {
  return actions.length === 0 ? ZERO_ACTIONS_SAVE_BLOCKED_REASON : null
}

/** The draft's action list, shaped for a create/update request body --
 * the one place a `DraftAction`'s three structured fields (domain,
 * service, entity id) are folded back into the `tool`/`arguments` shape
 * the route actually stores (`routes/macros.py`'s own
 * `MacroActionInput`). */
export function draftActionsToInput(actions: DraftAction[]): MacroActionInput[] {
  return actions.map((action) => ({
    tool: action.tool,
    arguments: { domain: action.domain, service: action.service, entity_id: action.entityId },
  }))
}
