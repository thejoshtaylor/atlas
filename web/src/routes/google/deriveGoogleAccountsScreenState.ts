// GoogleAccountsRoute's own logic, kept apart from its JSX -- copied in
// shape from `routes/plugins/derivePluginsScreenState.ts`. Imports
// nothing from React: what belongs here is only ever a fact about data,
// never a fact about a render.
import { ApiError } from "@/lib/api"
import type { GoogleAccount, GoogleClientStatus } from "@/lib/google"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

export type GoogleAccountsScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; client: GoogleClientStatus; accounts: GoogleAccount[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load Google accounts. Try again."
}

/** Two independent queries (the client status and the account list) fold
 * into one screen state -- either erroring wins over either loading, so
 * an operator whose client fetch failed sees the error rather than a
 * spinner that never resolves. */
export function deriveGoogleAccountsScreenState(
  clientQuery: QueryLike<GoogleClientStatus>,
  accountsQuery: QueryLike<GoogleAccount[]>,
): GoogleAccountsScreenState {
  if (clientQuery.status === "error") return { kind: "error", message: messageFor(clientQuery.error) }
  if (accountsQuery.status === "error") return { kind: "error", message: messageFor(accountsQuery.error) }
  if (clientQuery.status === "pending" || accountsQuery.status === "pending") return { kind: "loading" }
  const client = clientQuery.data
  const accounts = accountsQuery.data
  if (!client || !accounts) return { kind: "loading" }
  return { kind: "ready", client, accounts }
}

export type LinkAvailability = { available: true } | { available: false; reason: string }

/** D-02: linking must start from an https page -- Google only ever
 * returns to an https redirect URI, and the server independently refuses
 * a non-https Origin (09-03-PLAN.md). Named here so the route and its own
 * test share one literal, never two independently-typed copies. */
export const HTTPS_REQUIRED_REASON =
  "Linking needs this page served over https. Google only returns to an https address. Open ATLAS through its https address to link an account."

export function linkAvailability(protocol: string): LinkAvailability {
  if (protocol === "https:") return { available: true }
  return { available: false, reason: HTTPS_REQUIRED_REASON }
}
