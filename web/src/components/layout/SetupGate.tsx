import { Button } from "@/components/ui/button"

/**
 * The full-page treatment for the two states 03-UI-SPEC.md's Copywriting
 * Contract names, and this plan's objective asserts are the only two:
 *
 *   - `status="incomplete"`: the server answered that no admin exists
 *     yet. A visitor who reaches this state is, by definition, not the
 *     one who can complete setup (the create-admin route itself is the
 *     wizard's own entry, reachable separately) -- so this branch is
 *     deliberately a terminal message with no call to action.
 *   - `status="unreachable"`: the setup-status check itself failed (a
 *     different condition from "no admin exists yet") -- the
 *     "can't reach the server" message with a retry.
 */
export function SetupGate({
  status,
  onRetry,
}: {
  status: "incomplete" | "unreachable"
  onRetry?: () => void
}) {
  if (status === "incomplete") {
    return (
      <main className="flex min-h-svh flex-col items-center justify-center gap-2 p-6 text-center">
        <h1 className="text-display font-semibold text-foreground">spire-voice isn&apos;t set up yet.</h1>
        <p className="max-w-sm text-body text-muted-foreground">
          An admin needs to finish the first-run wizard before this works.
        </p>
      </main>
    )
  }

  return (
    <main className="flex min-h-svh flex-col items-center justify-center gap-3 p-6 text-center">
      <p className="max-w-sm text-body text-foreground">
        Can&apos;t reach the server. Check your connection and try again.
      </p>
      <Button variant="outline" onClick={onRetry}>
        Retry
      </Button>
    </main>
  )
}
