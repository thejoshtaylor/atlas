// AccountsRoute's own logic, kept apart from its JSX -- this project has
// no rendered-component test infrastructure (see CalibrationRoute's own
// derive module, 03-06, for the established reason and pattern this file
// copies). Two queries (accounts, invites) collapse into one screen
// state: a failed load of *either* disables invite and revoke together,
// since a screen that knows accounts but not invites (or the reverse)
// cannot safely offer either action.
import { ApiError } from "@/lib/api"
import type { Account, Invite } from "@/lib/accounts"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type AccountsScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; accounts: Account[]; invites: Invite[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Can't reach the server. Check your connection and try again."
}

export function deriveAccountsScreenState(input: {
  accounts: QueryLike<Account[]>
  invites: QueryLike<Invite[]>
}): AccountsScreenState {
  const { accounts, invites } = input

  if (accounts.status === "pending" || invites.status === "pending") {
    return { kind: "loading" }
  }
  if (accounts.status === "error") return { kind: "error", message: messageFor(accounts.error) }
  if (invites.status === "error") return { kind: "error", message: messageFor(invites.error) }

  return { kind: "ready", accounts: accounts.data ?? [], invites: invites.data ?? [] }
}
