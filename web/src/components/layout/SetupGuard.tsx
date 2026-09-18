import type { ReactNode } from "react"
import { useQuery } from "@tanstack/react-query"
import { useLocation } from "react-router-dom"
import { setupStatusQueryOptions } from "@/lib/setup"
import { SetupGate } from "./SetupGate"

/**
 * Wraps the entire route tree, including the public sign-in route --
 * setup being incomplete is a product-wide state, not something only
 * authenticated screens can hit.
 *
 * Reads the dedicated, always-answering `GET /api/setup/status`
 * (`lib/setup.ts`, plan 03-10) rather than inferring setup state from an
 * incidental 401/503 on some other route (`session.ts`'s own `/api/auth/me`,
 * this component's pre-03-10 signal) -- a genuine failure to reach the
 * server (`status.isError`) is now distinguishable from every other
 * outcome by construction, since this route is exempt from the setup gate
 * and answers unauthenticated.
 *
 * The terminal gate below reads specifically the `admin_account` step,
 * not the wizard's full five-step completion -- matching the backend's
 * own `require_setup_complete` (`account_repo.any_user_exists()`,
 * unchanged since 03-05): once an admin exists, the rest of the
 * application is reachable even with the wizard mid-way, exactly as
 * WEB-02's "never a lockout" requires. `/setup*` and `/sign-in` are
 * exempt from the terminal message even before an admin exists -- this is
 * the fix this plan (03-10) makes: previously `SetupGuard` blocked
 * `/setup` itself, so a clean install's own create-admin screen was
 * unreachable through the very gate meant to route a visitor to it.
 */
export function SetupGuard({ children }: { children: ReactNode }) {
  const location = useLocation()
  const status = useQuery(setupStatusQueryOptions)

  // Unknown until the status query settles once -- rendering `children`
  // here would risk a flash of the app (or the sign-in form) before an
  // incomplete-setup verdict arrives.
  if (status.isLoading) {
    return <div aria-hidden className="min-h-svh" />
  }

  if (status.isError) {
    return <SetupGate status="unreachable" onRetry={() => void status.refetch()} />
  }

  const adminAccountComplete =
    status.data?.steps.find((step) => step.name === "admin_account")?.complete ?? false
  const exemptPath = location.pathname.startsWith("/setup") || location.pathname.startsWith("/sign-in")

  if (!adminAccountComplete && !exemptPath) {
    return <SetupGate status="incomplete" />
  }

  return <>{children}</>
}
