import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Cpu, Mic, Volume2 } from "lucide-react"
import { Badge } from "@/components/ui/badge"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import {
  PROVIDERS_QUERY_KEY,
  fetchProviders,
  saveProvidersMutationOptions,
  type ProviderSlot,
} from "@/lib/providers"
import {
  degradedBadge,
  deriveProvidersScreenState,
  needsRestartBadge,
  wrappedBadge,
} from "./deriveProvidersScreenState"

// PROV-01, 07-UI-SPEC.md's Focal Point row: "whichever slot's selected
// option is degraded or pending restart". One card per slot, in the fixed
// order the server returns them (speech to text / text to speech /
// language model, per `routes/providers.py::_SERVED_SLOTS`) -- this order
// never changes, so returning to the page after a restart lands the eye
// on the same slot in the same place every time. Only the speech-to-text
// slot exists this plan; plan 07-02 adds the other two, never a second
// screen.

const SLOT_ICONS: Record<string, React.ComponentType<{ className?: string }>> = {
  stt: Mic,
  tts: Volume2,
  brain: Cpu,
}

function SlotCard({
  slot,
  selected,
  onSelect,
  disabled,
}: {
  slot: ProviderSlot
  selected: string
  onSelect: (name: string) => void
  disabled: boolean
}) {
  const Icon = SLOT_ICONS[slot.slot] ?? Mic
  const restart = needsRestartBadge(slot)
  const degraded = degradedBadge(slot)

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <Icon className="size-5" />
        <h2 className="text-heading font-semibold text-foreground">{slot.label}</h2>
      </div>

      {restart || degraded ? (
        <div className="flex flex-col gap-1">
          <div className="flex flex-wrap items-center gap-1.5">
            {restart ? <Badge variant={restart.badge.badgeVariant}>{restart.badge.badgeText}</Badge> : null}
            {degraded ? <Badge variant={degraded.badge.badgeVariant}>{degraded.badge.badgeText}</Badge> : null}
          </div>
          {restart ? <p className="text-label text-muted-foreground">{restart.caption}</p> : null}
          {degraded ? <p className="text-body text-destructive">{degraded.caption}</p> : null}
        </div>
      ) : null}

      <RadioGroup value={selected} onValueChange={onSelect} className="flex flex-col gap-3">
        {slot.options.map((option) => {
          const isActive = option.name === slot.active
          const wrapped = wrappedBadge(option.wrapped, isActive ? slot.measured_ms : null)
          const optionId = `provider-${slot.slot}-${option.name}`
          const needsCredential = option.requires_credential && !option.credential_set

          return (
            <div key={option.name} className="flex flex-col gap-1">
              <div className="flex touch-target items-center gap-3">
                <RadioGroupItem value={option.name} id={optionId} disabled={disabled} />
                <Label htmlFor={optionId}>{option.label}</Label>
              </div>

              {wrapped || option.licence_note || option.measured_note || needsCredential ? (
                <div className="flex flex-col gap-1 pl-7">
                  {wrapped ? (
                    <Badge variant={wrapped.badge.badgeVariant} className="w-fit">
                      {wrapped.badge.badgeText}
                    </Badge>
                  ) : null}
                  {wrapped?.caption ? <p className="text-label text-muted-foreground">{wrapped.caption}</p> : null}
                  {option.measured_note ? (
                    <p className="text-label text-muted-foreground">{option.measured_note}</p>
                  ) : null}
                  {option.licence_note ? (
                    <p className="text-label text-muted-foreground">{option.licence_note}</p>
                  ) : null}
                  {needsCredential ? (
                    <p className="text-label text-muted-foreground">
                      Needs an API key — add one in Settings before switching to this.
                    </p>
                  ) : null}
                </div>
              ) : null}
            </div>
          )
        })}
      </RadioGroup>
    </div>
  )
}

export function ProvidersRoute() {
  const query = useQuery({ queryKey: PROVIDERS_QUERY_KEY, queryFn: fetchProviders })
  const screen = deriveProvidersScreenState(query)
  const save = useMutation(saveProvidersMutationOptions)

  // The draft is retained on screen across a failed save (never reverted
  // to the last-saved value) so a failed save costs the admin no
  // re-entry -- 07-UI-SPEC.md's UI Considerations table, "Save failure".
  const [draft, setDraft] = React.useState<Record<string, string>>({})
  const [saveMessage, setSaveMessage] = React.useState<string | null>(null)
  const [saveError, setSaveError] = React.useState<string | null>(null)

  React.useEffect(() => {
    if (screen.kind !== "ready") return
    setDraft((current) => {
      let changed = false
      const next = { ...current }
      for (const slot of screen.slots) {
        if (!(slot.slot in next)) {
          next[slot.slot] = slot.selected
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [screen])

  const controlsDisabled = screen.kind !== "ready" || save.isPending

  const handleSave = async () => {
    setSaveMessage(null)
    setSaveError(null)
    try {
      await save.mutateAsync({
        slots: Object.fromEntries(
          Object.entries(draft).map(([slot, provider_name]) => [slot, { provider_name }]),
        ),
      })
      setSaveMessage("Saved. Restart the assistant for this to take effect.")
    } catch {
      setSaveError("Couldn't save your provider choices. Try again.")
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-display font-semibold">Providers</h1>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        <div className="flex flex-col gap-4">
          {screen.slots.map((slot) => (
            <SlotCard
              key={slot.slot}
              slot={slot}
              selected={draft[slot.slot] ?? slot.selected}
              onSelect={(name) => setDraft((current) => ({ ...current, [slot.slot]: name }))}
              disabled={controlsDisabled}
            />
          ))}

          <SubmitButton onSubmit={handleSave} disabled={controlsDisabled}>
            Save provider choices
          </SubmitButton>

          {saveMessage ? <p className="text-body text-foreground">{saveMessage}</p> : null}
          {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
        </div>
      ) : null}
    </div>
  )
}
