import { useQuery } from "@tanstack/react-query"
import { Navigate, useLocation } from "react-router-dom"
import { SkeletonList } from "@/components/state/SkeletonList"
import { ErrorState } from "@/components/state/ErrorState"
import { useSession } from "@/hooks/useSession"
import { firstUnfinishedStep, setupStatusQueryOptions } from "@/lib/setup"

/**
 * `/setup`'s own index: reads the unauthenticated status route (the one
 * call the application makes before it knows whether anything else will
 * answer) and routes to the first unfinished step -- never step one for
 * an admin who already finished it, and never a lockout for one who
 * closed the tab partway through (Task 1's own objective).
 *
 * `admin_account` is the only step reachable with no session (there is no
 * session to have yet); every other step needs one, so a visitor who
 * reaches here past `admin_account` with no session is sent to sign in
 * first, carrying `/setup` as `from` so `SignInRoute`'s own redirect
 * lands them back here once they do.
 */
export function WizardRoute() {
  const status = useQuery(setupStatusQueryOptions)
  const session = useSession()
  const location = useLocation()

  if (status.isLoading) {
    return (
      <main className="flex min-h-svh items-center justify-center p-6">
        <div className="w-full max-w-sm">
          <SkeletonList rows={2} />
        </div>
      </main>
    )
  }

  if (status.isError) {
    return (
      <main className="flex min-h-svh items-center justify-center p-6">
        <div className="w-full max-w-sm">
          <ErrorState
            message="Can't reach the server. Check your connection and try again."
            onRetry={() => void status.refetch()}
          />
        </div>
      </main>
    )
  }

  const nextStep = firstUnfinishedStep(status.data?.steps ?? [])
  if (nextStep === null) {
    return <Navigate to="/" replace />
  }

  if (nextStep === "admin_account") {
    return <Navigate to="/setup/admin_account" replace />
  }

  if (session.isLoading) {
    return (
      <main className="flex min-h-svh items-center justify-center p-6">
        <div className="w-full max-w-sm">
          <SkeletonList rows={2} />
        </div>
      </main>
    )
  }

  if (!session.data) {
    return (
      <Navigate to="/sign-in" replace state={{ from: { pathname: location.pathname } }} />
    )
  }

  return <Navigate to={`/setup/${nextStep}`} replace />
}
