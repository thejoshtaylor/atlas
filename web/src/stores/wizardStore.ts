import { create } from "zustand"

// Client-only scratch state for the first-run wizard: which step is
// showing, and the not-yet-saved draft of the fields on the current step
// (D-16). Step *completion* is server state (CONTEXT.md's "partial,
// wizard abandoned" row) -- this store never claims to know whether a
// step actually finished, only what the operator is mid-typing right
// now. Server data never lands here, matching D-16's whole point.
interface WizardState {
  currentStep: number
  draft: Record<string, unknown>
  setStep: (step: number) => void
  updateDraft: (patch: Record<string, unknown>) => void
  resetDraft: () => void
}

export const useWizardStore = create<WizardState>((set) => ({
  currentStep: 0,
  draft: {},
  setStep: (step) => set({ currentStep: step }),
  updateDraft: (patch) => set((state) => ({ draft: { ...state.draft, ...patch } })),
  resetDraft: () => set({ draft: {} }),
}))
