import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import {
  CREDENTIALS_QUERY_KEY,
  fetchCredentials,
  formatRelativeDate,
  saveCredentialMutationOptions,
  type CredentialEntry,
} from "@/lib/credentials"
import { PROVIDERS_QUERY_KEY, fetchProviders } from "@/lib/providers"
import { needsRestart } from "@/routes/providers/deriveProvidersScreenState"
import { TimezoneField } from "@/routes/wizard/TimezoneField"

// 03-UI-SPEC.md's Focal Point row: "the section headings and their Live
// / Needs-restart badges -- the badge is the information the screen
// exists to carry." This screen renders exactly what
// `GET /api/credentials` returns, section by section -- never a
// hardcoded list that happens to match today's slot set (this plan's
// own "Planner assumption carried forward" note: a setting the server
// reports and this screen does not render is a setting an operator
// cannot reach).

function CredentialSection({ entry }: { entry: CredentialEntry }) {
  const save = useMutation(saveCredentialMutationOptions)
  const [editing, setEditing] = React.useState(false)
  const [value, setValue] = React.useState("")
  const [saveError, setSaveError] = React.useState<string | null>(null)

  const handleSave = async () => {
    setSaveError(null)
    try {
      await save.mutateAsync({ slot: entry.slot, value })
      setEditing(false)
      setValue("")
    } catch {
      // The field returns to its prior state -- masked if it was
      // already set (`editing` resets to `false`, and `entry.is_set`,
      // untouched by this failure, puts the mask branch back on
      // screen), empty if it was not (the input branch renders
      // regardless of `editing` when `!entry.is_set`, and `value` is
      // cleared either way). This must never leave the operator looking
      // at a leftover, ambiguous draft: the stored value is never
      // readable back, so an ambiguous save is worse here than
      // anywhere else in the product (this plan's own text).
      setEditing(false)
      setValue("")
      setSaveError("Nothing was stored. Try again.")
      throw new Error("save failed")
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-heading font-semibold text-foreground">{entry.label}</p>
        <Badge variant="secondary">{entry.applies_live ? "Live" : "Needs restart"}</Badge>
      </div>

      {entry.is_set && !editing ? (
        <div className="flex items-center justify-between gap-2">
          <span className="text-body text-muted-foreground">
            {`•••••••• saved ${entry.updated_at ? formatRelativeDate(entry.updated_at) : "previously"}`}
          </span>
          <Button type="button" variant="outline" size="sm" onClick={() => setEditing(true)}>
            Update
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={`credential-${entry.slot}`}>{entry.label}</Label>
          <Input
            id={`credential-${entry.slot}`}
            type="password"
            className="scroll-field font-mono"
            placeholder="Not set"
            value={value}
            onChange={(event) => setValue(event.target.value)}
          />
          <SubmitButton onSubmit={handleSave} disabled={!value}>
            Save credentials
          </SubmitButton>
        </div>
      )}

      {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
    </div>
  )
}

export function SettingsRoute() {
  const query = useQuery({ queryKey: CREDENTIALS_QUERY_KEY, queryFn: fetchCredentials })
  // 07-UI-SPEC.md: a provider's choice and its credential are two
  // different facts edited in two different places -- this row only
  // links to /providers, it never grows a fourth credential section of
  // its own. `needsRestart` (deriveProvidersScreenState.ts) is the same
  // pure per-slot fact the /providers screen itself renders; the badge
  // here is present/absent for the whole screen, never a count, matching
  // "Safety policy"'s own binary Live/Needs-restart shape.
  const providersQuery = useQuery({ queryKey: PROVIDERS_QUERY_KEY, queryFn: fetchProviders })
  const providersNeedRestart = providersQuery.data?.slots.some((slot) => needsRestart(slot)) ?? false

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Settings</h1>
      </div>

      <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card p-4">
        <p className="text-heading font-semibold text-foreground">Safety policy</p>
        <div className="flex items-center gap-2">
          <Badge variant="secondary">Live</Badge>
          <Link to="/policy" className="touch-target flex items-center text-body text-primary underline-offset-4 hover:underline">
            Edit
          </Link>
        </div>
      </div>

      <div className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card p-4">
        <p className="text-heading font-semibold text-foreground">Providers</p>
        <div className="flex items-center gap-2">
          {providersNeedRestart ? <Badge variant="secondary">Needs restart</Badge> : null}
          <Link to="/providers" className="touch-target flex items-center text-body text-primary underline-offset-4 hover:underline">
            Edit
          </Link>
        </div>
      </div>

      <TimezoneField />

      {query.status === "pending" ? <SkeletonList rows={2} /> : null}

      {query.status === "error" ? (
        <ErrorState
          message="Can't reach the server. Check your connection and try again."
          onRetry={() => void query.refetch()}
        />
      ) : null}

      {query.status === "success"
        ? query.data.map((entry) => <CredentialSection key={entry.slot} entry={entry} />)
        : null}
    </div>
  )
}
