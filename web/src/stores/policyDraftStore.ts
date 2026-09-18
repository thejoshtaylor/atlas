import { create } from "zustand"

// The denylist/allowlist editor's unsaved buffer (D-16) -- the entity id
// or pattern the operator is about to add, before "Add to
// denylist"/"Add to allowlist" writes it through TanStack Query, and
// which row (if any) is mid-confirmation for removal. The loaded policy
// itself (what is currently denied/allowed) is server state and lives in
// a query cache, never here; this store only ever holds what has not
// been saved yet.
interface PolicyDraftState {
  pendingValue: string
  setPendingValue: (value: string) => void
  pendingRemovalId: number | null
  setPendingRemovalId: (id: number | null) => void
  clear: () => void
}

export const usePolicyDraftStore = create<PolicyDraftState>((set) => ({
  pendingValue: "",
  setPendingValue: (value) => set({ pendingValue: value }),
  pendingRemovalId: null,
  setPendingRemovalId: (id) => set({ pendingRemovalId: id }),
  clear: () => set({ pendingValue: "", pendingRemovalId: null }),
}))
