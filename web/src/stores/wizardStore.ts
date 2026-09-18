import { create } from "zustand"

// Client-only scratch state for the first-run wizard: the not-yet-saved
// draft of a field on the current step (D-16). Step *completion* is
// server state (CONTEXT.md's "partial, wizard abandoned" row) and *which*
// step is showing is now the address itself (`WizardRoute.tsx`'s
// `/setup/:step` route, plan 03-10 Task 1) -- neither lives here. This
// store never claims to know whether a step actually finished, only what
// the operator is mid-typing right now; server data never lands here,
// matching D-16's whole point.
//
// `currentStep`/`setStep` used to live here before the URL owned the
// current step -- removed as dead state once `WizardRoute.tsx` took over
// routing (plan 03-10).
interface WizardState {
  draft: Record<string, unknown>
  updateDraft: (patch: Record<string, unknown>) => void
  resetDraft: () => void
}

export const useWizardStore = create<WizardState>((set) => ({
  draft: {},
  updateDraft: (patch) => set((state) => ({ draft: { ...state.draft, ...patch } })),
  resetDraft: () => set({ draft: {} }),
}))
