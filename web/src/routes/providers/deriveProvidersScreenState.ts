// ProvidersRoute's own logic, kept apart from its JSX -- copied in shape
// from `routes/plugins/derivePluginsScreenState.ts`. Imports nothing from
// React: what belongs here is only ever a fact about data, never a fact
// about a render (`derivePolicyScreenState.ts`'s own opening line).
import { ApiError } from "@/lib/api"
import type { ProviderSlot, ProvidersState } from "@/lib/providers"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

// There is no "empty" member on purpose (07-UI-SPEC.md's Copywriting
// Contract, "Empty state: None"): D-03's boot-time registry validation
// guarantees at least one option per slot, so a `ready` state with a
// slot carrying zero options cannot occur on a real deployment.
export type ProvidersScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; slots: ProviderSlot[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load providers. Try again."
}

export function deriveProvidersScreenState(query: QueryLike<ProvidersState>): ProvidersScreenState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  const state = query.data
  if (!state) return { kind: "loading" }
  return { kind: "ready", slots: state.slots }
}

export interface SlotBadge {
  badgeVariant: "secondary" | "outline" | "denied"
  badgeText: string
}

/** Whether `slot`'s selected choice differs from what the process actually
 * built at boot (D-02) -- the one fact the "Needs restart" badge exists to
 * report. `active === null` (a degraded slot) is not itself "needs
 * restart" -- see `degradedBadge` below for that case. */
export function needsRestart(slot: Pick<ProviderSlot, "selected" | "active" | "state">): boolean {
  return slot.state !== "degraded" && slot.active !== null && slot.selected !== slot.active
}

/** The "Needs restart" badge plus the honest caption naming what is
 * actually running right now -- 07-UI-SPEC.md's Copywriting Contract,
 * verbatim. `null` when the slot's selected choice matches what is
 * running: steady state carries no noise. */
export function needsRestartBadge(slot: Pick<ProviderSlot, "selected" | "active" | "state" | "options">): {
  badge: SlotBadge
  caption: string
} | null {
  if (!needsRestart(slot)) return null
  const activeOption = slot.options.find((option) => option.name === slot.active)
  const activeLabel = activeOption?.label ?? slot.active ?? ""
  return {
    badge: { badgeVariant: "secondary", badgeText: "Needs restart" },
    caption: `Currently running: ${activeLabel}.`,
  }
}

/** The "Degraded — won't start" badge plus the server's own reason,
 * unchanged -- never paraphrased (the CMD-07/VOICE-02 lesson, restated
 * once more for this phase's own screen). `null` when the slot is not
 * degraded. */
export function degradedBadge(slot: Pick<ProviderSlot, "state" | "reason">): { badge: SlotBadge; caption: string } | null {
  if (slot.state !== "degraded") return null
  return {
    badge: { badgeVariant: "denied", badgeText: "Degraded — won't start" },
    caption: slot.reason ?? "",
  }
}

/** The "Wrapped" badge plus the honest synthesis-to-first-chunk figure
 * (D-06, D-08) -- `measuredMs` is `null` until a `BatchTtsAdapter`
 * actually measures one (plan 07-02), so the caption omits the number
 * rather than inventing a first-audio figure the wrapper did not earn.
 * `measuredMs` is a per-slot fact (the currently active provider's own
 * measurement, `ProviderSlot.measured_ms`), not a per-option one -- an
 * option that is not the active provider has never been measured. */
export function wrappedBadge(wrapped: boolean, measuredMs: number | null): { badge: SlotBadge; caption: string | null } | null {
  if (!wrapped) return null
  const caption =
    measuredMs === null
      ? null
      : `Renders the whole reply, then streams it out. Measured: ${Math.round(measuredMs)} ms synthesis-start to first chunk.`
  return { badge: { badgeVariant: "outline", badgeText: "Wrapped" }, caption }
}
