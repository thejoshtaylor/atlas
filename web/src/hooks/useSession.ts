import { useQuery } from "@tanstack/react-query"
import { SetupIncompleteError } from "@/lib/api"
import { sessionQueryOptions } from "@/lib/session"

/**
 * "The session query" every authenticated route's guard reads (this
 * plan's Task 2). Its `error` is either a `SetupIncompleteError` (the
 * global setup gate's terminal state) or anything else (a 401 from a
 * signed-out visitor, a network failure reaching the status check
 * itself, ...) -- `AuthGuard` and `SetupGate` branch on `error
 * instanceof SetupIncompleteError` rather than re-deriving this
 * distinction themselves.
 */
export function useSession() {
  return useQuery(sessionQueryOptions)
}

export { SetupIncompleteError }
