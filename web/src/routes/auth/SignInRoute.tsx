import * as React from "react"
import { useMutation } from "@tanstack/react-query"
import { Navigate, useLocation } from "react-router-dom"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/state/SubmitButton"
import { useSession } from "@/hooks/useSession"
import { loginMutationOptions } from "@/lib/session"
import { classifyLoginError } from "./authErrors"

/**
 * 03-UI-SPEC.md's Focal Point table: "the email/password pair as one
 * block; the submit button reads as part of it, not as a competing
 * anchor." A wrong email and a wrong password get the exact same
 * message ("Wrong email or password. Try again.", the Copywriting
 * Contract's login-error row) -- the server does not tell an
 * unauthenticated caller which addresses exist
 * (`_invalid_credentials_error`, `routes/auth.py`), and this screen must
 * not undo that by distinguishing them. Blank fields never reach the
 * server: inline field validation catches them first.
 */
export function SignInRoute() {
  const location = useLocation()
  const session = useSession()
  const [email, setEmail] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [fieldError, setFieldError] = React.useState<string | null>(null)
  const [serverError, setServerError] = React.useState<string | null>(null)
  const login = useMutation(loginMutationOptions)

  if (session.data) {
    const from = (location.state as { from?: { pathname: string } } | null)?.from
    return <Navigate to={from?.pathname ?? "/"} replace />
  }

  const handleSubmit = async () => {
    setServerError(null)
    if (!email.trim() || !password) {
      setFieldError("Enter your email and password.")
      throw new Error("blank field")
    }
    setFieldError(null)
    try {
      await login.mutateAsync({ email, password })
    } catch (caught) {
      setServerError(classifyLoginError(caught))
      throw caught
    }
  }

  const error = fieldError ?? serverError

  return (
    <main className="flex min-h-svh items-center justify-center p-6">
      <form
        className="flex w-full max-w-sm flex-col gap-4"
        onSubmit={(event) => event.preventDefault()}
      >
        <div className="flex flex-col gap-1">
          <p className="text-body text-muted-foreground">ATLAS</p>
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
