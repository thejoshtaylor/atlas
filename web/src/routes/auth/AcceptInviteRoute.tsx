import * as React from "react"
import { useMutation } from "@tanstack/react-query"
import { Navigate, useNavigate, useParams } from "react-router-dom"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/state/SubmitButton"
import { acceptInvite } from "@/lib/accounts"
import { classifyAcceptInviteError } from "./authErrors"

/**
 * A public route taking the token from the address -- accepting an
 * invite is unauthenticated by necessity (the person accepting has no
 * account yet, `accept_invite`'s own docstring). Asks for a display name
 * and a password and nothing else: no role field, because the role is
 * on the invite and the server ignores anything the body claims
 * (T-03-31) -- there is structurally nothing here for the operator to
 * even try to change. An expired or already-used token gets the
 * server's own named reason (`_invite_invalid_error`), never a generic
 * failure.
 *
 * `accept_invite` returns an `AccountResponse`, not a session -- no
 * cookie is set on acceptance, so this screen sends the new account to
 * `/sign-in` afterward rather than into the authenticated shell.
 */
export function AcceptInviteRoute() {
  const params = useParams<{ token: string }>()
  const navigate = useNavigate()
  const [displayName, setDisplayName] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [email, setEmail] = React.useState("")
  const [fieldError, setFieldError] = React.useState<string | null>(null)
  const [serverError, setServerError] = React.useState<string | null>(null)
  const accept = useMutation({ mutationFn: acceptInvite })

  if (!params.token) {
    return <Navigate to="/sign-in" replace />
  }
  const token = params.token

  const handleSubmit = async () => {
    setServerError(null)
    if (!displayName.trim() || !password) {
      setFieldError("Enter a display name and a password.")
      throw new Error("blank field")
    }
    setFieldError(null)
    try {
      await accept.mutateAsync({ token, displayName, password, email: email || undefined })
      navigate("/sign-in", { replace: true })
    } catch (caught) {
      setServerError(classifyAcceptInviteError(caught))
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
          <p className="text-body text-muted-foreground">atlas</p>
          <h1 className="text-display font-semibold">Accept invite</h1>
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="accept-display-name">Your name</Label>
          <Input
            id="accept-display-name"
            autoComplete="name"
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="accept-email">Email</Label>
          <Input
            id="accept-email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="accept-password">Password</Label>
          <Input
            id="accept-password"
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>

        {error ? <p className="text-body text-destructive">{error}</p> : null}

        <SubmitButton className="w-full" onSubmit={handleSubmit} pendingLabel="Creating account…">
          Accept invite
        </SubmitButton>
      </form>
    </main>
  )
}
