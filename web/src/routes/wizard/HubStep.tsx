import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ErrorState } from "@/components/state/ErrorState"
import { SubmitButton } from "@/components/state/SubmitButton"
import { CREDENTIALS_QUERY_KEY, fetchCredentials } from "@/lib/credentials"
import { checkHubStepMutationOptions, classifyHubCheckError } from "@/lib/wizard"
import { TimezoneField } from "./TimezoneField"
import { WizardCredentialField } from "./WizardCredentialField"
import { WizardStepShell } from "./WizardStepShell"

const HOME_ASSISTANT_SLOT = "ha_token"

/**
 * The hub step: collects the address (informational only -- `HA_URL` is
 * bootstrap configuration the operator sets when deploying, per
 * CONTEXT.md's "config.yaml shrinks to bootstrap only"; this field is
 * never written anywhere, it is a place to note what was configured) and
 * the token (saved through the credential route's own contract,
 * `WizardCredentialField`, reused from the settings surface's own field
 * treatment). "Continue" then asks the server to verify -- a real call to
 * Home Assistant using the exact read the assistant itself performs
 * (`routes/wizard.py`'s own `_probe_home_assistant`), so a hub that
 * passes here is a hub the assistant can use. A new operator also sets
 * the house's own time zone on this step (`TimezoneField`, 260924-h2f) --
 * Home Assistant is the fallback source for it, so this is the natural
 * place to offer it before the operator ever leaves this screen.
 */
export function HubStep() {
  const navigate = useNavigate()
  const [address, setAddress] = React.useState("")
  const [errorHeading, setErrorHeading] = React.useState<string | null>(null)
  const [errorDetail, setErrorDetail] = React.useState<string | undefined>(undefined)

  const credentials = useQuery({ queryKey: CREDENTIALS_QUERY_KEY, queryFn: fetchCredentials })
  const checkHub = useMutation(checkHubStepMutationOptions)
  const haEntry = credentials.data?.find((entry) => entry.slot === HOME_ASSISTANT_SLOT)

  const handleContinue = async () => {
    setErrorHeading(null)
    setErrorDetail(undefined)
    try {
      await checkHub.mutateAsync()
      navigate("/setup/provider_set", { replace: true })
    } catch (caught) {
      const { heading, detail } = classifyHubCheckError(caught)
      setErrorHeading(heading)
      setErrorDetail(detail)
      throw caught
    }
  }

  return (
    <WizardStepShell step="hub" title="Connect Home Assistant">
      <div className="flex flex-col gap-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="hub-address">Home Assistant address</Label>
          <Input
            id="hub-address"
            className="scroll-field"
            placeholder="http://homeassistant.local:8123"
            value={address}
            onChange={(event) => setAddress(event.target.value)}
          />
          <p className="text-label text-muted-foreground">
            Set with the <code>HA_URL</code> environment variable when you deployed --
            noted here so you can confirm it matches, not saved from this field.
          </p>
        </div>

        <WizardCredentialField entry={haEntry} label="Home Assistant token" slot={HOME_ASSISTANT_SLOT} />

        <TimezoneField />

        {errorHeading ? <ErrorState message={errorHeading} detail={errorDetail} onRetry={handleContinue} /> : null}

        <SubmitButton className="w-full" onSubmit={handleContinue} pendingLabel="Checking connection…">
          Continue
        </SubmitButton>
      </div>
    </WizardStepShell>
  )
}
