import * as React from "react"
import { useMutation } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/state/SubmitButton"
import { createAdminMutationOptions } from "@/lib/wizard"
import { WizardStepShell } from "./WizardStepShell"

/**
 * The only screen a clean install offers (Task 1's own objective) --
 * reached with no session, because there is no session to have.
 * `POST /api/auth/create-admin` answers only while no user of any role
 * exists and is permanently closed afterward, so a second attempt after
 * this one names that reason rather than a generic failure -- rendered
 * verbatim below, never paraphrased.
 */
export function CreateAdminStep() {
  const navigate = useNavigate()
  const [email, setEmail] = React.useState("")
  const [displayName, setDisplayName] = React.useState("")
  const [password, setPassword] = React.useState("")
  const [fieldError, setFieldError] = React.useState<string | null>(null)
  const [serverError, setServerError] = React.useState<string | null>(null)
  const createAdmin = useMutation(createAdminMutationOptions)

  const handleSubmit = async () => {
    setServerError(null)
    if (!email.trim() || !displayName.trim() || !password) {
      setFieldError("Fill in every field.")
      throw new Error("blank field")
    }
    setFieldError(null)
    try {
      await createAdmin.mutateAsync({ email, display_name: displayName, password })
      navigate("/setup/hub", { replace: true })
    } catch (caught) {
      // The server's own named reason, unmodified -- an account that
      // already exists is the one case this step must explain rather
      // than paraphrase (Task 1's own instruction).
      setServerError(caught instanceof Error ? caught.message : "Could not create the admin account.")
      throw caught
    }
  }

  const error = fieldError ?? serverError

  return (
    <WizardStepShell step="admin_account" title="Create the admin account">
      <form className="flex flex-col gap-4" onSubmit={(event) => event.preventDefault()}>
        <p className="text-body text-muted-foreground">
          This is the only account that can be created this way, and it closes once it
          exists. Choose the password yourself here -- there is no default password, and
          nothing is ever printed anywhere.
        </p>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="admin-email">Email</Label>
          <Input
            id="admin-email"
            type="email"
            autoComplete="email"
            value={email}
            onChange={(event) => setEmail(event.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="admin-name">Display name</Label>
          <Input
            id="admin-name"
            autoComplete="name"
            value={displayName}
            onChange={(event) => setDisplayName(event.target.value)}
          />
        </div>

        <div className="flex flex-col gap-1.5">
          <Label htmlFor="admin-password">Password</Label>
          <Input
            id="admin-password"
            type="password"
            autoComplete="new-password"
            value={password}
            onChange={(event) => setPassword(event.target.value)}
          />
        </div>

        {error ? <p className="text-body text-destructive">{error}</p> : null}

        <SubmitButton className="w-full" onSubmit={handleSubmit} pendingLabel="Creating account…">
          Continue
        </SubmitButton>
      </form>
    </WizardStepShell>
  )
}
