// The provider-choice admin surface's fetch layer (PROV-01, D-01 .. D-04).
// Every shape here matches `src/spire_voice/routes/providers.py`'s own
// response models field for field, the same discipline `lib/plugins.ts`
// states at its own top of file -- a renamed or reshaped field here is a
// silent drift from what the server actually sends.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `ProviderOptionResponse`'s exact shape -- one selectable provider for
 * one slot, whether or not it is the currently selected one. */
export interface ProviderOption {
  name: string
  label: string
  requires_credential: boolean
  credential_set: boolean
  wrapped: boolean
  licence_note: string | null
  /** `ProviderEntry.needs_server_url` (07-04-PLAN.md) -- the flag, not a
   * provider name, that reveals the language-model slot's "Server URL"
   * field for exactly the option that reads one. */
  needs_server_url: boolean
  /** `ProviderEntry.measured_note` (D-12) -- the published local-set
   * latency figure, verbatim, for any local option across all three
   * slots. `null` for a non-local option. */
  measured_note: string | null
}

/** `ProviderSlotResponse`'s exact shape. `selected` is a fresh repository
 * read; `active` is what the running process actually built its client
 * from, or `null` for a degraded slot -- the two are never assumed equal
 * (D-02's own reason this pair exists at all). `state` is exactly
 * `"running"` or `"degraded"` (`ProviderSlotStatus`'s own closed set),
 * plus `"starting"` for the transitional read right after boot, before
 * `app.state.provider_slots` has this slot's entry. */
export type ProviderSlotState = "starting" | "running" | "degraded"

export interface ProviderSlot {
  slot: string
  label: string
  selected: string
  active: string | null
  state: ProviderSlotState
  reason: string | null
  wrapped: boolean
  measured_ms: number | null
  options: ProviderOption[]
  settings: Record<string, unknown>
  /** Whether the stored selection -- the provider name or its settings --
   * has moved since the running process built this slot. The only
   * "needs restart" signal that works for a degraded slot, whose
   * `active` is `null` (WR-08, code review). */
  selection_changed_since_boot: boolean
}

/** `ProvidersResponse`'s exact shape. */
export interface ProvidersState {
  slots: ProviderSlot[]
}

export const PROVIDERS_QUERY_KEY = ["providers"] as const

export function fetchProviders(): Promise<ProvidersState> {
  return apiFetch<ProvidersState>("/api/providers")
}

/** `SetProviderSlotInput`'s exact shape -- what a save request sends per
 * slot. */
export interface SetProviderSlotInput {
  provider_name: string
  settings?: Record<string, unknown>
}

/** `SetProvidersRequest`'s exact shape: every slot in one body
 * (07-UI-SPEC.md's probe addendum) -- one request, one outcome, no
 * per-slot save. */
export interface SetProvidersInput {
  slots: Record<string, SetProviderSlotInput>
}

/** `ProvidersWriteResponse`'s exact shape. */
export interface ProvidersWriteResult extends ProvidersState {
  applies_live: boolean
}

/**
 * No optimistic update, the same reasoning `plugins.ts` records for its
 * own write mutations: a provider choice takes effect on the assistant's
 * next restart (D-02), so a slot shown as changed before the server
 * confirmed it would be a choice the admin believes is saved when it is
 * not. `onSuccess` seeds the cache from the confirmed response.
 */
export const saveProvidersMutationOptions: UseMutationOptions<ProvidersWriteResult, unknown, SetProvidersInput> = {
  mutationFn: (input) => apiFetch<ProvidersWriteResult>("/api/providers", { method: "PUT", body: input }),
  onSuccess: (result) => {
    queryClient.setQueryData(PROVIDERS_QUERY_KEY, { slots: result.slots } satisfies ProvidersState)
  },
}
