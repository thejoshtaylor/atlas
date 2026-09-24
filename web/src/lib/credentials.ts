// Provider credentials: write-only from the browser (PROV-04, D-07).
// `src/atlas/routes/credentials.py`'s own docstring: the list
// route never returns a ciphertext, a plaintext, or any prefix/suffix of
// either -- this module's `CredentialEntry` type has no field that could
// hold one, matching the server's response model field-for-field.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `CredentialListEntry`'s exact shape. */
export interface CredentialEntry {
  slot: string
  label: string
  is_set: boolean
  updated_at: string | null
  source: "database" | "environment" | "unset"
  applies_live: boolean
}

export const CREDENTIALS_QUERY_KEY = ["credentials"] as const

export function fetchCredentials(): Promise<CredentialEntry[]> {
  return apiFetch<CredentialEntry[]>("/api/credentials")
}

export interface SaveCredentialInput {
  slot: string
  value: string
}

/**
 * The value is never read back -- this mutation's own return type
 * (`CredentialEntry`) structurally cannot carry it. A failed save must
 * leave the field exactly where it was (masked if set, blank if not);
 * this mutation makes no attempt to guess or restore a value on
 * failure, since there is nothing here to restore it *from*.
 */
export const saveCredentialMutationOptions: UseMutationOptions<CredentialEntry, unknown, SaveCredentialInput> = {
  mutationFn: ({ slot, value }) =>
    apiFetch<CredentialEntry>(`/api/credentials/${slot}`, { method: "PUT", body: { value } }),
  onSuccess: (entry) => {
    queryClient.setQueryData<CredentialEntry[]>(CREDENTIALS_QUERY_KEY, (existing) =>
      existing ? existing.map((item) => (item.slot === entry.slot ? entry : item)) : existing,
    )
  },
}

/**
 * "saved 2 days ago" -- a relative rendering of `updated_at`, the same
 * shape `CalibrationRoute.tsx`'s own `formatAge` uses for the identical
 * reason (a raw ISO timestamp is not what an operator reads at a
 * glance).
 */
export function formatRelativeDate(iso: string): string {
  const then = new Date(iso).getTime()
  const now = Date.now()
  const hours = (now - then) / (1000 * 60 * 60)
  if (hours < 1) return "less than an hour ago"
  if (hours < 24) {
    const roundedHours = Math.round(hours)
    return `${roundedHours} hour${roundedHours === 1 ? "" : "s"} ago`
  }
  const days = Math.round(hours / 24)
  return `${days} day${days === 1 ? "" : "s"} ago`
}
