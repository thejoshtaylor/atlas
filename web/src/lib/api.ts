// The single fetch seam (03-UI-SPEC.md, this plan's objective). No screen
// may call `fetch` directly -- every request in this application goes
// through `apiFetch`, which turns a 401, a setup-incomplete 503, and any
// other non-2xx into three distinguishable outcomes:
//
//   - 401                           -> UnauthorizedError, and the cached
//                                      session is cleared so the guard
//                                      (components/layout/AuthGuard.tsx)
//                                      redirects to sign-in on its next
//                                      render, rather than this module
//                                      reaching into the router directly.
//   - 503 naming setup incomplete   -> SetupIncompleteError, rendered as
//                                      the full-page setup gate
//                                      (components/layout/SetupGate.tsx).
//   - any other non-2xx             -> ApiError carrying the server's own
//                                      `detail` string, never a
//                                      paraphrase -- this backend writes
//                                      named reasons into `detail` (the
//                                      `_calibration_disabled_error`
//                                      factory in `app.py` is the
//                                      convention this project already
//                                      established) and a paraphrase
//                                      would lose the thing that makes
//                                      them actionable.
import { queryClient } from "./queryClient"

export class ApiError extends Error {
  readonly status: number

  constructor(status: number, message: string) {
    super(message)
    this.name = "ApiError"
    this.status = status
  }
}

/** A 401: the session cookie is missing, expired, or was never valid. */
export class UnauthorizedError extends ApiError {
  constructor(message: string) {
    super(401, message)
    this.name = "UnauthorizedError"
  }
}

/**
 * A 503 whose body names setup being incomplete -- distinct from every
 * other 503 (a route that is merely down, say). This is the condition
 * the global setup gate renders for; the plan's objective is explicit
 * that only this shape and a failed status check itself (thrown as a
 * plain `ApiError`/network `TypeError`, never this class) are real.
 */
export class SetupIncompleteError extends ApiError {
  constructor(message: string) {
    super(503, message)
    this.name = "SetupIncompleteError"
  }
}

const SETUP_INCOMPLETE_MARKER = "setup incomplete"

async function readDetail(response: Response): Promise<string | undefined> {
  try {
    const body: unknown = await response.clone().json()
    if (body && typeof body === "object" && "detail" in body) {
      const detail = (body as { detail: unknown }).detail
      if (typeof detail === "string") return detail
    }
  } catch {
    // The body wasn't JSON, or had no `detail` -- fall through to the
    // generic message built from the status below.
  }
  return undefined
}

export interface ApiFetchInit extends Omit<RequestInit, "body"> {
  body?: unknown
}

/**
 * The one seam. `credentials: "same-origin"` on every request so the
 * HttpOnly session cookie travels; a JSON `body` (when given) is
 * serialized and content-typed automatically so callers never repeat
 * that boilerplate per screen.
 */
export async function apiFetch<T = unknown>(path: string, init: ApiFetchInit = {}): Promise<T> {
  const { body, headers, ...rest } = init
  const response = await fetch(path, {
    ...rest,
    credentials: "same-origin",
    headers: {
      ...(body !== undefined ? { "Content-Type": "application/json" } : {}),
      ...headers,
    },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })

  if (response.status === 401) {
    // Clear the cached session so the guard's next render sees "no
    // session" rather than a stale, now-invalid one.
    queryClient.setQueryData(["session"], null)
    const detail = await readDetail(response)
    throw new UnauthorizedError(detail ?? "Not authenticated.")
  }

  if (response.status === 503) {
    const detail = await readDetail(response)
    if (detail?.toLowerCase().includes(SETUP_INCOMPLETE_MARKER)) {
      throw new SetupIncompleteError(detail)
    }
    throw new ApiError(503, detail ?? "The server is unavailable.")
  }

  if (!response.ok) {
    const detail = await readDetail(response)
    throw new ApiError(response.status, detail ?? `Request failed with status ${response.status}.`)
  }

  if (response.status === 204) {
    return undefined as T
  }
  return (await response.json()) as T
}

export interface Session {
  email: string
  role: "admin" | "operator" | "viewer"
}

/**
 * "The session query" every authenticated route's guard reads (Task 2's
 * own text). Its error modes are exactly the setup gate's two states:
 * `SetupIncompleteError` (the terminal, no-CTA message) versus anything
 * else (the "can't reach the server" message with a retry) -- the
 * planner assumption this plan's objective records, asserted to have no
 * third state.
 */
export function fetchSession(): Promise<Session> {
  return apiFetch<Session>("/auth/session")
}

export interface LoginCredentials {
  email: string
  password: string
}

export function login(credentials: LoginCredentials): Promise<Session> {
  return apiFetch<Session>("/auth/login", { method: "POST", body: credentials })
}

/**
 * The setup gate's "Retry" button (SetupGate/SetupGuard) calls this
 * rather than the session query's own `.refetch()` directly, so the
 * network-retrigger stays inside this module -- the one seam -- rather
 * than a bare `<query>.refetch()` call sitting in a layout component.
 */
export function retrySessionCheck(): void {
  void queryClient.refetchQueries({ queryKey: ["session"] })
}
