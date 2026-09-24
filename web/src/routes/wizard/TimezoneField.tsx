import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Badge } from "@/components/ui/badge"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import { initialTimezoneValue, setTimezoneMutationOptions, timezoneQueryOptions } from "@/lib/wizard"

/** `resolved_from`'s own four values (`routes/wizard.py::resolve_timezone`),
 * worded for an operator rather than shown as the raw key. */
const SOURCE_LABEL: Record<string, string> = {
  database: "saved here",
  config: "from server.timezone",
  home_assistant: "from Home Assistant",
  process: "the server's own zone",
}

/**
 * The house time zone: the setup hub step and the Settings screen both
 * render this exact card, with no props -- it owns its own query and
 * mutation (`timezoneQueryOptions`/`setTimezoneMutationOptions`,
 * `lib/wizard.ts`), so a new operator meets it once, in the hub step
 * (Home Assistant is the fallback source for it), and an operator who
 * already finished setup sees the identical card in Settings (260924-h2f,
 * issue #1). The server's own startup `WARNING` text (when present)
 * renders here verbatim, in the same words the boot log carries.
 */
export function TimezoneField() {
  const query = useQuery(timezoneQueryOptions)
  const mutation = useMutation(setTimezoneMutationOptions)

  // Seeded once from the server's own boot value the first time real
  // data lands, then never yanked back under the operator's cursor by a
  // later re-fetch -- the same "seed once, never re-seed" discipline
  // `WakeTuningRoute.tsx`'s own previewed-threshold state already uses.
  const [value, setValue] = React.useState("")
  const seededRef = React.useRef(false)
  React.useEffect(() => {
    if (!seededRef.current && query.data) {
      const browserZone = Intl.DateTimeFormat().resolvedOptions().timeZone
      setValue(initialTimezoneValue(query.data, browserZone))
      seededRef.current = true
    }
  }, [query.data])

  if (query.status === "pending") return <SkeletonList rows={1} />
  if (query.status === "error") {
    return (
      <ErrorState
        message="Can't reach the server. Check your connection and try again."
        onRetry={() => void query.refetch()}
      />
    )
  }

  const status = query.data
  const handleSave = async () => {
    await mutation.mutateAsync({ zone: value })
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-2">
        <p className="text-heading font-semibold text-foreground">Time zone</p>
        <Badge variant="secondary">{status.applies_live ? "Live" : "Needs restart"}</Badge>
      </div>

      <p className="text-body text-muted-foreground">
        {`Running in ${status.zone} (${SOURCE_LABEL[status.resolved_from] ?? status.resolved_from})`}
      </p>

      {status.warning ? (
        <p role="alert" className="text-body text-destructive">
          {status.warning}
        </p>
      ) : null}

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="timezone-name">House time zone</Label>
        <Input
          id="timezone-name"
          className="scroll-field"
          list="timezone-names"
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
        <datalist id="timezone-names">
          {Intl.supportedValuesOf("timeZone").map((zone) => (
            <option key={zone} value={zone} />
          ))}
        </datalist>
      </div>

      {status.stored && status.stored !== status.zone ? (
        <p className="text-label text-muted-foreground">{`Saved ${status.stored}. Restart ATLAS to use it.`}</p>
      ) : null}

      {mutation.isError ? (
        <ErrorState
          message="Could not save the time zone."
          detail={mutation.error instanceof ApiError ? mutation.error.message : undefined}
          onRetry={() => void handleSave()}
        />
      ) : null}

      <SubmitButton className="w-full" onSubmit={handleSave} disabled={!value} pendingLabel="Saving…">
        Save time zone
      </SubmitButton>
    </div>
  )
}
