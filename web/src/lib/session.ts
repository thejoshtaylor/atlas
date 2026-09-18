// The session surface: the current-session query, the sign-in and
// sign-out mutations, and the role the rest of the application reads
// (03-08's Task 1, this plan's own file list). `AuthGuard`
// (components/layout/AuthGuard.tsx) and `SetupGuard`
// (components/layout/SetupGuard.tsx) both key off `SESSION_QUERY_KEY`,
// so a sign-in or sign-out here is what those guards see on their very
// next render -- no second signal needed.
//
// Session, fetchSession, and login used to live in `./api.ts`, pointed at
// `/auth/session`/`/auth/login` -- routes that were never real
// (`src/spire_voice/routes/auth.py` serves `/api/auth/me` and
// `/api/auth/login`). Both `03-03-SUMMARY.md` and `03-05-SUMMARY.md`
// flagged this exact gap as owed to whichever plan landed the session
// surface; this is that plan.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

export type Role = "admin" | "operator" | "viewer"

/** `SessionResponse`'s exact shape (`src/spire_voice/routes/auth.py`). */
export interface Session {
  id: number
  email: string
  display_name: string
  role: Role
}

export const SESSION_QUERY_KEY = ["session"] as const

/**
 * "The current-session query" every authenticated route's guard reads.
 * Its error modes are exactly the setup gate's two states:
 * `SetupIncompleteError` (the terminal, no-CTA message) versus anything
 * else -- a 401 from a signed-out visitor falls through to `AuthGuard`'s
 * own sign-in redirect, never rendered as a setup problem.
 */
export function fetchSession(): Promise<Session> {
  return apiFetch<Session>("/api/auth/me")
}

export const sessionQueryOptions = {
  queryKey: SESSION_QUERY_KEY,
  queryFn: fetchSession,
}

export interface LoginCredentials {
  email: string
  password: string
}

function login(credentials: LoginCredentials): Promise<Session> {
  return apiFetch<Session>("/api/auth/login", { method: "POST", body: credentials })
}

/**
 * A wrong email and a wrong password both reject with the same
 * `ApiError` (`_invalid_credentials_error`, `routes/auth.py`) -- this
 * mutation does not distinguish them either; `SignInRoute` is the one
 * place that turns a 401 here into the Copywriting Contract's exact
 * login-error text.
 */
export const loginMutationOptions: UseMutationOptions<Session, unknown, LoginCredentials> = {
  mutationFn: login,
  onSuccess: (session) => {
    queryClient.setQueryData(SESSION_QUERY_KEY, session)
  },
}

function logout(): Promise<void> {
  return apiFetch<void>("/api/auth/logout", { method: "POST" })
}

export const logoutMutationOptions: UseMutationOptions<void, unknown, void> = {
  mutationFn: logout,
  onSuccess: () => {
    queryClient.setQueryData(SESSION_QUERY_KEY, null)
  },
}

/**
 * The setup gate's "Retry" button (SetupGate/SetupGuard) calls this
 * rather than the session query's own `.refetch()` directly, so the
 * network-retrigger stays inside this module rather than a bare
 * `<query>.refetch()` call sitting in a layout component.
 */
export function retrySessionCheck(): void {
  void queryClient.refetchQueries({ queryKey: SESSION_QUERY_KEY })
}
