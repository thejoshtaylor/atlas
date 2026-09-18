import { Navigate, Outlet, useLocation } from "react-router-dom"
import { useSession } from "@/hooks/useSession"
import { SkeletonList } from "@/components/state/SkeletonList"

/**
 * Every authenticated route sits behind this one guard. It renders a
 * loading state while the session is unknown -- never a flash of the
 * sign-in screen for a user who is signed in, which is what checking
 * `isLoading` before branching on `isError` guarantees. Setup-incomplete
 * is handled by `SetupGuard` above this in the tree (App.tsx); by the
 * time this guard runs, an error here means "not signed in," nothing
 * more.
 */
export function AuthGuard() {
  const location = useLocation()
  const session = useSession()

  if (session.isLoading) {
    return (
      <main className="flex min-h-svh items-center justify-center p-6">
        <div className="w-full max-w-sm">
          <SkeletonList rows={2} />
        </div>
      </main>
    )
  }

  if (session.isError || !session.data) {
    return <Navigate to="/sign-in" replace state={{ from: location }} />
  }

  return <Outlet />
}
