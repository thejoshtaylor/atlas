import { useMutation, useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ErrorState } from "@/components/state/ErrorState"
import { SubmitButton } from "@/components/state/SubmitButton"
import { setAudioSourceMutationOptions, wizardStatusQueryOptions } from "@/lib/wizard"
import { WizardStepShell } from "./WizardStepShell"

const CAMERA_SOURCE = "camera"

/**
 * The source step: `VALID_AUDIO_SOURCES` (`routes/wizard.py`) is a closed
 * one-member set today, so this is a single, pre-selected choice rather
 * than a real decision -- still a `RadioGroup`, not a plain confirmation,
 * so a second source arriving later needs no new component here. Carries
 * the "Needs restart" badge from the server's own `applies_live`
 * classification (03-UI-SPEC.md's Copywriting Contract) -- audio-source
 * changes take effect on the next restart, and this step says so in the
 * contract's own words rather than implying an effect the change does
 * not have.
 */
export function AudioSourceStep() {
  const navigate = useNavigate()
  const wizardStatus = useQuery(wizardStatusQueryOptions)
  const setSource = useMutation(setAudioSourceMutationOptions)

  const step = wizardStatus.data?.steps.find((s) => s.name === "audio_source")
  const appliesLive = (step?.detail?.applies_live as boolean | undefined) ?? false

  const handleContinue = async () => {
    await setSource.mutateAsync({ source: CAMERA_SOURCE })
    navigate("/setup/room", { replace: true })
  }

  return (
    <WizardStepShell step="audio_source" title="Choose the audio source">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card p-4">
          <RadioGroup value={CAMERA_SOURCE} className="flex-1">
            <div className="flex items-center gap-3">
              <RadioGroupItem value={CAMERA_SOURCE} id="source-camera" />
              <Label htmlFor="source-camera">Camera microphone</Label>
            </div>
          </RadioGroup>
          <Badge variant="secondary">{appliesLive ? "Live" : "Needs restart"}</Badge>
        </div>

        {setSource.isError ? (
          <ErrorState
            message="Couldn't save the audio source. Try again."
            onRetry={() => void handleContinue()}
          />
        ) : null}

        <SubmitButton className="w-full" onSubmit={handleContinue}>
          Continue
        </SubmitButton>
      </div>
    </WizardStepShell>
  )
}
