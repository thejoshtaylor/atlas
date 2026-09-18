import { useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { CREDENTIALS_QUERY_KEY, fetchCredentials } from "@/lib/credentials"
import { wizardStatusQueryOptions } from "@/lib/wizard"
import { WizardCredentialField } from "./WizardCredentialField"
import { WizardStepShell } from "./WizardStepShell"

const PROVIDER_SET_SLOTS = ["stt_api_key", "brain_api_key", "tts_api_key"] as const

/**
 * The provider-set step: the three AI-provider credential slots
 * (`_PROVIDER_SET_SLOTS`, `routes/wizard.py`), each rendered with the
 * settings surface's own field treatment (`WizardCredentialField`). A
 * slot satisfied only by an environment value reads as set, because it
 * is -- `GET /api/credentials`'s own `is_set` already reflects that
 * resolution order, this step does not re-derive it.
 *
 * "Continue" re-reads `GET /api/wizard`'s own `provider_set` step rather
 * than trusting the browser's own idea of which fields look filled --
 * when the server still names a blank slot, that name is what renders
 * (the design contract's own rule: a disabled button with no explanation
 * is the exact failure this exists to stop), and nothing advances.
 */
export function ProviderSetStep() {
  const navigate = useNavigate()
  const credentials = useQuery({ queryKey: CREDENTIALS_QUERY_KEY, queryFn: fetchCredentials })
  const wizardStatus = useQuery(wizardStatusQueryOptions)

  const entryFor = (slot: string) => credentials.data?.find((entry) => entry.slot === slot)
  const providerStep = wizardStatus.data?.steps.find((step) => step.name === "provider_set")
  const missing = (providerStep?.detail?.missing as string[] | undefined) ?? []
  const missingLabels = missing
    .map((slot) => entryFor(slot)?.label ?? slot)
    .filter((label): label is string => Boolean(label))

  const handleContinue = async () => {
    const refreshed = await wizardStatus.refetch()
    const refreshedStep = refreshed.data?.steps.find((step) => step.name === "provider_set")
    if (refreshedStep?.complete) {
      navigate("/setup/audio_source", { replace: true })
    }
    // Not complete: the missing-slot list above already re-renders from
    // the refetched data -- nothing else to do here.
  }

  if (credentials.isLoading) {
    return (
      <WizardStepShell step="provider_set" title="Add your provider keys">
        <SkeletonList rows={3} />
      </WizardStepShell>
    )
  }

  if (credentials.isError) {
    return (
      <WizardStepShell step="provider_set" title="Add your provider keys">
        <ErrorState
          message="Can't reach the server. Check your connection and try again."
          onRetry={() => void credentials.refetch()}
        />
      </WizardStepShell>
    )
  }

  return (
    <WizardStepShell step="provider_set" title="Add your provider keys">
      <div className="flex flex-col gap-4">
        {PROVIDER_SET_SLOTS.map((slot) => {
          const entry = entryFor(slot)
          return <WizardCredentialField key={slot} entry={entry} label={entry?.label ?? slot} slot={slot} />
        })}

        {missingLabels.length > 0 ? (
          <p className="text-body text-destructive">Still needed: {missingLabels.join(", ")}.</p>
        ) : null}

        <SubmitButton className="w-full" onSubmit={handleContinue}>
          Continue
        </SubmitButton>
      </div>
    </WizardStepShell>
  )
}
