// The accounts and invites surface: who has access, the role each of
// them has, and the invites waiting to be accepted (WEB-04, WEB-05,
// 03-08 Task 1). Every route here matches
// `src/atlas/routes/accounts.py` exactly -- field names copied
// verbatim from its response models, not renamed to a convention of this
// module's own invention.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"
import type { Role } from "./session"

/** `AccountResponse`'s exact shape. Never a password hash -- the server
 * never sends one, and nothing here re-adds a field it removed. */
export interface Account {
  id: number
  email: string
  display_name: string
  role: Role
  disabled: boolean
}

/** `InviteResponse`'s exact shape -- no token, no hash. */
export interface Invite {
  id: number
  role: Role
  email: string | null
  expires_at: string
  accepted: boolean
}

/** `InviteCreatedResponse`'s exact shape -- the one response that ever
 * carries the plaintext invite token. */
export interface InviteCreated {
  id: number
  token: string
  role: Role
  email: string | null
  expires_at: string
}

export const ACCOUNTS_QUERY_KEY = ["accounts"] as const
export const INVITES_QUERY_KEY = ["invites"] as const

export function fetchAccounts(): Promise<Account[]> {
  return apiFetch<Account[]>("/api/accounts")
}

export function fetchInvites(): Promise<Invite[]> {
  return apiFetch<Invite[]>("/api/invites")
}

export interface RemoveAccountInput {
  accountId: number
}

/** Disables the account server-side (`remove_account`'s own docstring:
 * disables rather than deletes) -- this mutation's name matches the
 * Copywriting Contract's "Remove access" confirm button, not the
 * server's internal verb. */
export const removeAccountMutationOptions: UseMutationOptions<void, unknown, RemoveAccountInput> = {
  mutationFn: ({ accountId }) => apiFetch<void>(`/api/accounts/${accountId}`, { method: "DELETE" }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: ACCOUNTS_QUERY_KEY })
  },
}

export interface CreateInviteInput {
  role: Role
  email?: string
}

/**
 * Sends an invite and returns the plaintext token exactly once
 * (`InviteCreatedResponse`) -- the caller (`AccountsRoute`) is
 * responsible for presenting it as something to copy and never fetching
 * it again, since no route will ever return it a second time.
 */
export const createInviteMutationOptions: UseMutationOptions<InviteCreated, unknown, CreateInviteInput> = {
  mutationFn: (input) => apiFetch<InviteCreated>("/api/invites", { method: "POST", body: input }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: INVITES_QUERY_KEY })
  },
}

export interface RevokeInviteInput {
  inviteId: number
}

export const revokeInviteMutationOptions: UseMutationOptions<void, unknown, RevokeInviteInput> = {
  mutationFn: ({ inviteId }) => apiFetch<void>(`/api/invites/${inviteId}`, { method: "DELETE" }),
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: INVITES_QUERY_KEY })
  },
}

export interface AcceptInviteInput {
  token: string
  displayName: string
  password: string
  email?: string
}

/**
 * Unauthenticated by necessity -- the person accepting has no account
 * yet (`accept_invite`'s own docstring). Note what this does *not* do:
 * the response is an `AccountResponse`, not a `SessionResponse` -- no
 * cookie is set, so accepting an invite does not sign the new account
 * in. `AcceptInviteRoute` sends the person to `/sign-in` afterward.
 */
export function acceptInvite(input: AcceptInviteInput): Promise<Account> {
  return apiFetch<Account>(`/api/invites/${input.token}/accept`, {
    method: "POST",
    body: {
      display_name: input.displayName,
      password: input.password,
      ...(input.email ? { email: input.email } : {}),
    },
  })
}

/**
 * "1 pending invite" / "3 pending invites" -- 03-UI-SPEC.md's
 * zero-one-many row, factored out as a pure function so the singular
 * case is asserted directly rather than inferred from a rendered string.
 * `count` is never 0 here: `AccountsRoute` renders the Copywriting
 * Contract's empty state instead of this count line when there are no
 * pending invites.
 */
export function formatPendingInviteCount(count: number): string {
  return `${count} pending invite${count === 1 ? "" : "s"}`
}
