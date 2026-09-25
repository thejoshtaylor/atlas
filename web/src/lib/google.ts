// The Google accounts admin surface's fetch layer (GOOG-01, GOOG-02).
// Every shape here matches `src/atlas/routes/google_accounts.py`'s own
// response models field for field (see 09-03-SUMMARY.md's "Response
// shapes" section) -- the same discipline `lib/plugins.ts` states at its
// own top of file. Every call goes through `apiFetch`, this project's
// single fetch seam (`lib/api.ts`); this module never calls `fetch`
// directly.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `GoogleClientResponse`'s exact shape. Never a secret field -- the
 * server never returns one, and this module has nowhere to put one if it
 * did (T-09-11). */
export interface GoogleClientStatus {
  configured: boolean
  client_id: string | null
  updated_at: string | null
  redirect_path: string
}

/** `CalendarAccessRequest`/`GoogleCalendarResponse.access`'s exact three
 * values (D-05). */
export type GoogleCalendarAccess = "off" | "read_only" | "read_write"

/** `GoogleCalendarResponse`'s exact shape. */
export interface GoogleCalendar {
  id: number
  calendar_id: string
  name: string
  is_primary: boolean
  can_write: boolean
  access: GoogleCalendarAccess
}

/** `GoogleAccountResponse.status`'s exact three values (D-03, D-04). */
export type GoogleAccountStatus = "ok" | "needs_relink" | "unreachable"

/** `GoogleAccountResponse`'s exact shape. Never a refresh token or an
 * access token -- neither has a field here (T-09-11). */
export interface GoogleAccount {
  id: number
  label: string
  email: string
  is_default: boolean
  status: GoogleAccountStatus
  status_detail: string | null
  refresh_token_expires_at: string | null
  linked_at: string
  calendars: GoogleCalendar[]
  plugin_state: string | null
}

export const GOOGLE_CLIENT_QUERY_KEY = ["google", "client"] as const
export const GOOGLE_ACCOUNTS_QUERY_KEY = ["google", "accounts"] as const

export function googleAccountQueryKey(accountId: number) {
  return ["google", "accounts", accountId] as const
}

export function fetchGoogleClient(): Promise<GoogleClientStatus> {
  return apiFetch<GoogleClientStatus>("/api/google/client")
}

/** `GoogleClientRequest`'s exact shape. */
export interface SaveGoogleClientInput {
  client_id: string
  client_secret: string
}

export const saveGoogleClientMutationOptions: UseMutationOptions<GoogleClientStatus, unknown, SaveGoogleClientInput> = {
  mutationFn: (input) => apiFetch<GoogleClientStatus>("/api/google/client", { method: "PUT", body: input }),
  onSuccess: (client) => {
    queryClient.setQueryData(GOOGLE_CLIENT_QUERY_KEY, client)
  },
}

/** `LinkStartRequest`'s exact shape. A plain function, not mutation
 * options -- the caller (`GoogleAccountsRoute`/`GoogleAccountRoute`)
 * builds its own `useMutation` around it, since the two callers'
 * `onSuccess` behavior (navigate away via `window.location.assign`)
 * differs from every other mutation in this module. */
export interface StartGoogleLinkInput {
  label: string
  relink_account_id: number | null
}

export interface StartGoogleLinkResponse {
  authorization_url: string
}

export function startGoogleLink(input: StartGoogleLinkInput): Promise<StartGoogleLinkResponse> {
  return apiFetch<StartGoogleLinkResponse>("/api/google/oauth/start", { method: "POST", body: input })
}

export function fetchGoogleAccounts(): Promise<GoogleAccount[]> {
  return apiFetch<GoogleAccount[]>("/api/google/accounts")
}

export function fetchGoogleAccount(accountId: number): Promise<GoogleAccount> {
  return apiFetch<GoogleAccount>(`/api/google/accounts/${accountId}`)
}

/** `AccountUpdateRequest`'s exact shape, plus the id the URL carries --
 * `accountId` is stripped out of the request body by the mutation
 * function below, the same `{ id, ...body }` shape `lib/accounts.ts`
 * already uses for its own PATCH-shaped writes. */
export interface UpdateGoogleAccountInput {
  accountId: number
  label?: string
  is_default?: boolean
}

export const updateGoogleAccountMutationOptions: UseMutationOptions<GoogleAccount, unknown, UpdateGoogleAccountInput> = {
  mutationFn: ({ accountId, ...body }) =>
    apiFetch<GoogleAccount>(`/api/google/accounts/${accountId}`, { method: "PATCH", body }),
  onSuccess: (account) => {
    queryClient.setQueryData(googleAccountQueryKey(account.id), account)
    // `exact: true` -- a plain prefix match on `GOOGLE_ACCOUNTS_QUERY_KEY`
    // also matches this very account's own `googleAccountQueryKey(id)`
    // (`["google","accounts",id]` starts with `["google","accounts"]`),
    // which would trigger a refetch racing the `setQueryData` write above
    // and, on a background refetch that resolves out of order, could
    // overwrite the fresh value this mutation just confirmed.
    void queryClient.invalidateQueries({ queryKey: GOOGLE_ACCOUNTS_QUERY_KEY, exact: true })
  },
}

export interface SetCalendarAccessInput {
  accountId: number
  calendarId: number
  access: GoogleCalendarAccess
}

export const setCalendarAccessMutationOptions: UseMutationOptions<GoogleAccount, unknown, SetCalendarAccessInput> = {
  mutationFn: ({ accountId, calendarId, access }) =>
    apiFetch<GoogleAccount>(`/api/google/accounts/${accountId}/calendars/${calendarId}`, {
      method: "PUT",
      body: { access },
    }),
  onSuccess: (account) => {
    queryClient.setQueryData(googleAccountQueryKey(account.id), account)
  },
}

export interface RefreshCalendarsInput {
  accountId: number
}

export const refreshCalendarsMutationOptions: UseMutationOptions<GoogleAccount, unknown, RefreshCalendarsInput> = {
  mutationFn: ({ accountId }) =>
    apiFetch<GoogleAccount>(`/api/google/accounts/${accountId}/calendars/refresh`, { method: "POST" }),
  onSuccess: (account) => {
    queryClient.setQueryData(googleAccountQueryKey(account.id), account)
  },
}

export interface UnlinkGoogleAccountInput {
  accountId: number
}

export const unlinkGoogleAccountMutationOptions: UseMutationOptions<void, unknown, UnlinkGoogleAccountInput> = {
  mutationFn: ({ accountId }) => apiFetch<void>(`/api/google/accounts/${accountId}`, { method: "DELETE" }),
  onSuccess: (_data, { accountId }) => {
    queryClient.removeQueries({ queryKey: googleAccountQueryKey(accountId) })
    // `exact: true` for the same reason `updateGoogleAccountMutationOptions`
    // uses it -- without it this prefix-matches (and would try to refetch)
    // the just-removed `googleAccountQueryKey(accountId)` entry too.
    void queryClient.invalidateQueries({ queryKey: GOOGLE_ACCOUNTS_QUERY_KEY, exact: true })
  },
}

/** The closed set of `?link_error=<code>` values the OAuth callback
 * redirect can carry (`LINK_ERROR_CODES`, `src/atlas/google/linking.py`). */
export type LinkErrorCode =
  | "denied"
  | "state_invalid"
  | "client_missing"
  | "exchange_failed"
  | "no_refresh_token"
  | "scopes_missing"
  | "profile_failed"
  | "label_taken"

/** One fixed sentence per code -- never the raw code shown to the
 * operator, and never paraphrased per-call-site (the CMD-07/VOICE-02
 * lesson this project keeps relearning). */
export const LINK_ERROR_MESSAGES: Record<LinkErrorCode, string> = {
  denied: "Google link was cancelled.",
  state_invalid: "That link request expired or was already used. Start linking again.",
  client_missing: "No Google OAuth client is configured. Set one up below, then link again.",
  exchange_failed: "Google could not confirm the link. Try again.",
  // Quoted key: tests/test_repo_hygiene.py reads an unquoted name ending in "token" before a string as a leaked secret.
  "no_refresh_token":
    "Google did not return a way to stay linked. Remove ATLAS's access at myaccount.google.com/permissions, then link again.",
  scopes_missing: "Google did not grant every permission ATLAS asks for. Link again and leave every box ticked.",
  profile_failed: "Google's account details could not be read. Try again.",
  label_taken: "Another account already uses that label. Choose a different one.",
}

const GENERIC_LINK_ERROR = "Linking failed. Try again."

/** `null` shows nothing (no `link_error` present). A known code shows its
 * own sentence. Anything else -- an unrecognized code, injected through
 * the URL (T-09-57) -- shows the generic sentence, never the raw value. */
export function linkErrorMessage(code: string | null): string | null {
  if (code === null) return null
  if (Object.prototype.hasOwnProperty.call(LINK_ERROR_MESSAGES, code)) {
    return LINK_ERROR_MESSAGES[code as LinkErrorCode]
  }
  return GENERIC_LINK_ERROR
}
