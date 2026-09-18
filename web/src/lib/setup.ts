// The unauthenticated setup-status query -- the one call the application
// makes before it knows whether anything else will answer (`GET
// /api/setup/status`, `SETUP_GATE_EXEMPT_PATHS`, plan 03-09). It answers
// with step names and booleans only, before authentication is even
// possible, so `SetupGuard` and the wizard's own root (`WizardRoute.tsx`)
// both read this rather than inferring setup state from an incidental 401
// or 503 on some other route.
import { apiFetch } from "./api"

/** `SetupStep`'s exact shape (`src/spire_voice/routes/auth.py`) -- a name
 * and a boolean, never the fuller `detail` object `GET /api/wizard`
 * returns to an authenticated admin. */
export interface SetupStep {
  name: string
  complete: boolean
}

/** `SetupStatusResponse`'s exact shape. `steps` arrives in the server's
 * own `STEP_ORDER` (`routes/wizard.py`) -- this module never re-derives
 * that order itself. */
export interface SetupStatus {
  complete: boolean
  steps: SetupStep[]
}

export const SETUP_STATUS_QUERY_KEY = ["setup-status"] as const

export function fetchSetupStatus(): Promise<SetupStatus> {
  return apiFetch<SetupStatus>("/api/setup/status")
}

export const setupStatusQueryOptions = {
  queryKey: SETUP_STATUS_QUERY_KEY,
  queryFn: fetchSetupStatus,
}

/**
 * The first step this list reports incomplete, or `null` once every step
 * holds -- a pure function over the server's own ordered list, so
 * `WizardRoute`'s routing decision is unit-testable with no query, no
 * router, and no DOM.
 */
export function firstUnfinishedStep(steps: SetupStep[]): string | null {
  return steps.find((step) => !step.complete)?.name ?? null
}
