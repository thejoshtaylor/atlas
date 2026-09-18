import { Navigate, Outlet } from "react-router-dom"
import { useSession } from "@/hooks/useSession"
import type { Role } from "@/lib/session"

const ROLE_RANK: Record<Role, number> = { viewer: 0, operator: 1, admin: 2 }

/**
 * A route-tree gate mirroring `require_role` (`auth/dependencies.py`) --
 * this is presentation, not access control (T-03-47): the browser hides
 * a screen a role cannot use, and every write it could attempt is
 * refused by the route regardless. This component only decides what
 * renders; it holds no line the server does not already hold on its own.
 *
 * Sits inside `AuthGuard`, so `useSession().data` is always present by
 * the time this renders.
 */
export function RequireRole({ minimum }: { minimum: Role }) {
  const session = useSession()
  const role = session.data?.role ?? "viewer"

  if (ROLE_RANK[role] < ROLE_RANK[minimum]) {
    return <Navigate to="/" replace />
  }

  return <Outlet />
}
