// The macro editor's fetch surface (MACRO-03, D-11). Every shape here
// matches `src/atlas/routes/macros.py`'s own response models
// field for field, the same discipline `web/src/lib/policy.ts` states at
// its own top of file -- a renamed or reshaped field here is a silent
// drift from what the server actually sends.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `ConflictAnnotation`'s exact four values (`routes/macros.py`).
 * `unknown` is a distinct member on purpose -- a policy or catalog read
 * that failed must never collapse into `ok` (T-04-36). */
export type ConflictAnnotation = "ok" | "denied" | "not_found" | "unknown"

/** `MacroActionResponse`'s exact shape. */
export interface MacroAction {
  id: number
  position: number
  tool: string
  arguments: Record<string, unknown>
  conflict: ConflictAnnotation
}

/** `MacroResponse`'s exact shape. `reply_synthesis_degraded`/
 * `reply_synthesis_message` are only ever meaningful on a create/update
 * response (`routes/macros.py`'s own comment) -- `false`/`null` on every
 * list/read response, since a read never re-synthesizes anything. */
export interface Macro {
  id: number
  phrase: string
  aliases: string[]
  reply: string
  actions: MacroAction[]
  created_at: string
  updated_at: string
  created_by_user_id: number | null
  reply_cached: boolean
  reply_synthesis_degraded: boolean
  reply_synthesis_message: string | null
}

export const MACROS_QUERY_KEY = ["macros"] as const

export function macroQueryKey(macroId: number) {
  return ["macros", macroId] as const
}

export function fetchMacros(): Promise<Macro[]> {
  return apiFetch<Macro[]>("/api/macros")
}

export function fetchMacro(macroId: number): Promise<Macro> {
  return apiFetch<Macro>(`/api/macros/${macroId}`)
}

/** `MacroActionInput`'s exact shape -- what a create/update request
 * sends per action. */
export interface MacroActionInput {
  tool: string
  arguments: Record<string, unknown>
}

/** `CreateMacroRequest`/`UpdateMacroRequest` share this shape exactly. */
export interface SaveMacroInput {
  phrase: string
  aliases: string[]
  reply: string
  actions: MacroActionInput[]
}

/**
 * No optimistic update, same reasoning `policy.ts` records for its own
 * write mutations: the server performs a real side effect that can fail
 * (the collision check, the reply re-synthesis) -- a row shown before
 * the server confirmed it would be a row the operator believes is saved
 * when it might not be. `onSuccess` seeds both the list and the
 * single-macro cache directly from the response rather than triggering a
 * second fetch, since the response already carries everything a GET
 * would return.
 */
export const createMacroMutationOptions: UseMutationOptions<Macro, unknown, SaveMacroInput> = {
  mutationFn: (input) => apiFetch<Macro>("/api/macros", { method: "POST", body: input }),
  onSuccess: (macro) => {
    void queryClient.invalidateQueries({ queryKey: MACROS_QUERY_KEY })
    queryClient.setQueryData(macroQueryKey(macro.id), macro)
  },
}

export interface UpdateMacroInput extends SaveMacroInput {
  macroId: number
}

export const updateMacroMutationOptions: UseMutationOptions<Macro, unknown, UpdateMacroInput> = {
  mutationFn: ({ macroId, ...input }) => apiFetch<Macro>(`/api/macros/${macroId}`, { method: "PUT", body: input }),
  onSuccess: (macro) => {
    void queryClient.invalidateQueries({ queryKey: MACROS_QUERY_KEY })
    queryClient.setQueryData(macroQueryKey(macro.id), macro)
  },
}

export interface DeleteMacroInput {
  macroId: number
}

export const deleteMacroMutationOptions: UseMutationOptions<void, unknown, DeleteMacroInput> = {
  mutationFn: ({ macroId }) => apiFetch<void>(`/api/macros/${macroId}`, { method: "DELETE" }),
  onSuccess: (_data, { macroId }) => {
    void queryClient.invalidateQueries({ queryKey: MACROS_QUERY_KEY })
    queryClient.removeQueries({ queryKey: macroQueryKey(macroId) })
  },
}
