import * as React from "react"
import { useMutation } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/state/SubmitButton"
import { formatRelativeDate, saveCredentialMutationOptions, type CredentialEntry } from "@/lib/credentials"

/**
 * The mask/date/Update-or-blank-input field treatment the settings
 * surface uses (`routes/settings/SettingsRoute.tsx`'s own
 * `CredentialSection`, plan 03-08), reproduced here at the pattern level
 * for the wizard's hub and provider-set steps (Task 2's own instruction:
 * "the credential field treatment this step reuses rather than
 * re-implements"). `CredentialSection` itself is a private, unexported
 * function in a file outside this plan's declared scope, so this
 * component matches its behaviour and copy exactly -- a mask and a date
 * for a set slot, an update control that opens a fresh blank input, never
 * pre-filled -- rather than importing something that cannot be imported.
 *
 * Saves through the exact same `lib/credentials.ts` mutation the settings
 * surface uses -- there is one write path for a credential, not a second
 * one this step invents. A failed save clears the local draft and leaves
 * the mask/blank state exactly where it was: the stored value is never
 * readable back, so an ambiguous draft is worse here than anywhere else.
 */
export function WizardCredentialField({
  entry,
  label,
  slot,
}: {
  entry: CredentialEntry | undefined
  label: string
  slot: string
}) {
  const save = useMutation(saveCredentialMutationOptions)
  const [editing, setEditing] = React.useState(false)
  const [value, setValue] = React.useState("")
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const isSet = entry?.is_set ?? false

  const handleSave = async () => {
    setSaveError(null)
    try {
      await save.mutateAsync({ slot, value })
      setEditing(false)
      setValue("")
    } catch {
      setEditing(false)
      setValue("")
      setSaveError("Nothing was stored. Try again.")
      throw new Error("save failed")
    }
  }

  return (
    <div className="flex flex-col gap-1.5">
      <Label htmlFor={`wizard-credential-${slot}`}>{label}</Label>
      {isSet && !editing ? (
        <div className="flex items-center justify-between gap-2">
          <span className="text-body text-muted-foreground">
            {`•••••••• saved ${entry?.updated_at ? formatRelativeDate(entry.updated_at) : "previously"}`}
          </span>
          <Button type="button" variant="outline" size="sm" onClick={() => setEditing(true)}>
            Update
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-2">
          <Input
            id={`wizard-credential-${slot}`}
            type="password"
            className="scroll-field font-mono"
            placeholder="Not set"
            value={value}
            onChange={(event) => setValue(event.target.value)}
          />
          <SubmitButton size="sm" variant="outline" onSubmit={handleSave} disabled={!value}>
            Save
          </SubmitButton>
        </div>
      )}
      {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
    </div>
  )
}
