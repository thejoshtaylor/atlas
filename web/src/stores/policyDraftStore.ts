import { create } from "zustand"

// The denylist/allowlist editor's unsaved buffer (D-16) -- the entity id
// or pattern the operator is about to add, before "Add to
// denylist"/"Add to allowlist" writes it through TanStack Query. The
// loaded policy itself (what is currently denied/allowed) is server
// state and lives in a query cache, never here; this store only ever
// holds what has not been saved yet.
interface PolicyDraftState {
  pendingValue: string
  setPendingValue: (value: string) => void
  clear: () => void
}

export const usePolicyDraftStore = create<PolicyDraftState>((set) => ({
  pendingValue: "",
  setPendingValue: (value) => set({ pendingValue: value }),
  clear: () => set({ pendingValue: "" }),
}))
