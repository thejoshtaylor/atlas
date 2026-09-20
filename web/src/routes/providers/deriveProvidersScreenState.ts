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

/** Whether `slot`'s stored choice differs from what the process actually
 * built at boot (D-02) -- the one fact the "Needs restart" badge exists to
 * report.
 *
 * WR-08 (code review): `selection_changed_since_boot` comes first because
 * it is the only one of the two that works for a DEGRADED slot. A
 * degraded slot's `active` is `null`, so `selected !== active` can never
 * fire for it -- and a degraded slot is precisely the one an admin has
 * just changed. Walk the case it got wrong: the language-model slot
 * boots degraded for want of a server URL, the admin types the URL and
 * saves, and the screen showed no restart badge and the OLD failure
 * reason, beside the field they had just filled in. Their fix looked
 * like it had not taken.
 *
 * The second clause stays for the running case, where the server reports
 * both names and the comparison is the more direct statement of the same
 * fact. */
export function needsRestart(
  slot: Pick<ProviderSlot, "selected" | "active" | "state" | "selection_changed_since_boot">,
): boolean {
  if (slot.selection_changed_since_boot) return true
  return slot.state !== "degraded" && slot.active !== null && slot.selected !== slot.active
}

/** The "Needs restart" badge plus the honest caption naming what is
 * actually running right now -- 07-UI-SPEC.md's Copywriting Contract,
 * verbatim. `null` when the slot's selected choice matches what is
 * running: steady state carries no noise. */
export function needsRestartBadge(
  slot: Pick<
    ProviderSlot,
    "selected" | "active" | "state" | "options" | "selection_changed_since_boot"
  >,
): {
  badge: SlotBadge
  caption: string
} | null {
  if (!needsRestart(slot)) return null
  // A degraded slot has no active provider to name, so the caption says
  // what is true instead of naming nothing (WR-08). Omitting the caption
  // here would repeat the very "states neither" failure this fix is
  // about: the badge would be shown with no honest present tense beside
  // it.
  if (slot.active === null) {
    return {
      badge: { badgeVariant: "secondary", badgeText: "Needs restart" },
      caption: "Currently running: nothing — this slot did not start.",
    }
  }
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
export function degradedBadge(
  slot: Pick<ProviderSlot, "state" | "reason" | "selection_changed_since_boot">,
): { badge: SlotBadge; caption: string } | null {
  if (slot.state !== "degraded") return null
  // WR-08 (code review): the reason describes the configuration this
  // process booted with. Once the admin has stored a different one, it
  // describes a configuration that is no longer saved anywhere -- so
  // rendering it beside the field they just corrected is not "the
  // server's reason, verbatim", it is a stale claim about the present.
  // The "Needs restart" badge above takes over for that state and says
  // the thing that is actually true.
  if (slot.selection_changed_since_boot) return null
  return {
    badge: { badgeVariant: "denied", badgeText: "Degraded — won't start" },
    caption: slot.reason ?? "",
  }
}

/** The "Wrapped" badge plus the honest synthesis-to-first-chunk figure
 * (D-06, D-08) -- `measuredMs` is `null` until a `BatchTtsAdapter`
 * actually measures one (plan 07-02), so the caption states the honest
 * absence (07-UI-SPEC.md's plan-07-02 addendum) rather than inventing a
 * first-audio figure the wrapper did not earn, and never omits the
 * caption outright: the null case is the honest sibling of the measured
 * sentence, not silence. `measuredMs` is a per-slot fact (the currently
 * active provider's own measurement, `ProviderSlot.measured_ms`), not a
 * per-option one -- an option that is not the active provider has never
 * been measured. */
export function wrappedBadge(wrapped: boolean, measuredMs: number | null): { badge: SlotBadge; caption: string } | null {
  if (!wrapped) return null
  const caption =
    measuredMs === null
      ? "Renders the whole reply, then streams it out. No synthesis measured since the last restart."
      : `Renders the whole reply, then streams it out. Measured: ${Math.round(measuredMs)} ms synthesis-start to first chunk.`
  return { badge: { badgeVariant: "outline", badgeText: "Wrapped" }, caption }
}
