import type { ReactNode } from "react"
import { useSession } from "@/hooks/useSession"
import { SetupIncompleteError, UnauthorizedError } from "@/lib/api"
import { retrySessionCheck } from "@/lib/session"
import { SetupGate } from "./SetupGate"

/**
 * Wraps the entire route tree, including the public sign-in route --
 * setup being incomplete is a product-wide state, not something only
 * authenticated screens can hit. A 401 from the session check is a
 * routine "not signed in yet" outcome, not a setup problem, so it falls
 * through to normal routing (AuthGuard handles it there); every other
 * error is treated as the status check itself having failed.
 */
export function SetupGuard({ children }: { children: ReactNode }) {
  const session = useSession()

  // Unknown until the session query settles once -- rendering `children`
  // here would risk a flash of the sign-in form (or the app itself)
  // before a setup-incomplete or unreachable-server verdict arrives.
  if (session.isLoading) {
    return <div aria-hidden className="min-h-svh" />
  }

  if (session.isError) {
    if (session.error instanceof SetupIncompleteError) {
      return <SetupGate status="incomplete" />
    }
    if (!(session.error instanceof UnauthorizedError)) {
      return <SetupGate status="unreachable" onRetry={retrySessionCheck} />
    }
  }

  return <>{children}</>
}
