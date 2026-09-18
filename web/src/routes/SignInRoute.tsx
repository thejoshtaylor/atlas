import * as React from "react"
import { Navigate, useLocation } from "react-router-dom"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/state/SubmitButton"
import { useSession } from "@/hooks/useSession"
import { ApiError, login } from "@/lib/api"
import { queryClient } from "@/lib/queryClient"

/**
 * 03-UI-SPEC.md's Focal Point table: "the email/password pair as one
 * block; the submit button reads as part of it, not as a competing
 * anchor." Errors use the exact Copywriting Contract row for a login
 * failure -- never a paraphrase of whatever `ApiError.message` carries,
 * since a wrong-credentials 401 and a genuine server error are different
 * things an operator should be able to tell apart.
 */
export function SignInRoute() {
  const location = useLocation()
  const session = useSession()
  const [email, setEmail] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)

  if (session.data) {
    const from = (location.state as { from?: { pathname: string } } | null)?.from
    return <Navigate to={from?.pathname ?? "/"} replace />
  }

  const handleSubmit = async () => {
    setError(null)
    try {
      const nextSession = await login({ email, password })
      queryClient.setQueryData(["session"], nextSession)
    } catch (caught) {
      if (caught instanceof ApiError && caught.status === 401) {
        setError("Wrong email or password. Try again.")
      } else {
        setError(caught instanceof Error ? caught.message : "Sign-in failed.")
      }
      throw caught
    }
  }

  return (
    <main className="flex min-h-svh items-center justify-center p-6">
      <form
        className="flex w-full max-w-sm flex-col gap-4"
        onSubmit={(event) => event.preventDefault()}
      >
        <div className="flex flex-col gap-1">
          <p className="text-body text-muted-foreground">spire-voice</p>
          <h1 className="text-display font-semibold">Sign in</h1>
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="email">Email</Label>
          <Input
            id="email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="password">Password</Label>
          <Input
            id="password"
            type="password"
            autoComplete="current-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>

        {error ? <p className="text-body text-destructive">{error}</p> : null}

        <SubmitButton className="w-full" onSubmit={handleSubmit} pendingLabel="Signing in…">
          Sign in
        </SubmitButton>
      </form>
    </main>
  )
}
