import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ErrorState } from "@/components/state/ErrorState"
import { SubmitButton } from "@/components/state/SubmitButton"
import { setAudioSourceMutationOptions, wizardStatusQueryOptions } from "@/lib/wizard"
import { WizardStepShell } from "./WizardStepShell"

type AudioSource = "camera" | "edge"

const CAMERA_SOURCE: AudioSource = "camera"
const EDGE_SOURCE: AudioSource = "edge"

/**
 * The source step: `VALID_AUDIO_SOURCES` (`routes/wizard.py`) held a
 * single member until D-15 -- the second source this step's `RadioGroup`
 * was built to receive with no new component has now arrived. Starts on
 * `detail.source` from the status query, falling back to camera when
 * nothing is stored yet ("seed once, never re-seed" -- `TimezoneField.tsx`'s
 * established pattern, so a later refetch never yanks the selection back
 * under the operator's cursor). Carries the "Needs restart" badge from the
 * server's own `applies_live` classification (03-UI-SPEC.md's Copywriting
 * Contract) regardless of which source is selected -- both sources take
 * effect on the next restart, and this step says so in the contract's own
 * words rather than implying an effect neither change has.
 */
export function AudioSourceStep() {
  const navigate = useNavigate()
  const wizardStatus = useQuery(wizardStatusQueryOptions)
  const setSource = useMutation(setAudioSourceMutationOptions)

  const step = wizardStatus.data?.steps.find((s) => s.name === "audio_source")
  const appliesLive = (step?.detail?.applies_live as boolean | undefined) ?? false

  const [source, setSourceValue] = React.useState<AudioSource>(CAMERA_SOURCE)
  const seededRef = React.useRef(false)
  React.useEffect(() => {
    if (!seededRef.current && step) {
      const stored = step.detail?.source as AudioSource | undefined
      setSourceValue(stored === EDGE_SOURCE ? EDGE_SOURCE : CAMERA_SOURCE)
      seededRef.current = true
    }
  }, [step])

  const handleContinue = async () => {
    await setSource.mutateAsync({ source })
    navigate("/setup/room", { replace: true })
  }

  return (
    <WizardStepShell step="audio_source" title="Choose the audio source">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card p-4">
          <RadioGroup
            value={source}
            onValueChange={(value) => setSourceValue(value as AudioSource)}
            className="flex-1"
          >
            <div className="flex items-center gap-3">
              <RadioGroupItem value={CAMERA_SOURCE} id="source-camera" />
              <Label htmlFor="source-camera">Camera microphone</Label>
            </div>
            <div className="flex flex-col gap-1">
              <div className="flex items-center gap-3">
                <RadioGroupItem value={EDGE_SOURCE} id="source-edge" />
                <Label htmlFor="source-edge">Edge microphone (Raspberry Pi)</Label>
              </div>
              <p className="pl-7 text-label text-muted-foreground">
                Pair the Pi under Edge devices first. The change applies after a restart.
              </p>
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
