import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import { followUpWindowQueryOptions, saveFollowUpWindowMutationOptions } from "@/lib/followUp"

/**
 * The answer window's own admin field (D-09, plan 09-07): a self-contained
 * Settings card, with no props -- the same shape `TimezoneField.tsx`
 * establishes (its own query and mutation, mounted directly below it on
 * `SettingsRoute.tsx`). Seeded once from the server's own current value
 * the first time real data lands, then never yanked back under the
 * operator's cursor by a later re-fetch -- `TimezoneField.tsx`'s own
 * "seed once, never re-seed" discipline.
 */
export function FollowUpWindowField() {
  const query = useQuery(followUpWindowQueryOptions)
  const mutation = useMutation(saveFollowUpWindowMutationOptions)

  const [value, setValue] = React.useState("")
  const seededRef = React.useRef(false)
  React.useEffect(() => {
    if (!seededRef.current && query.data) {
      setValue(String(query.data.window_s))
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

  const handleSave = async () => {
    await mutation.mutateAsync({ window_s: Number(value) })
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <p className="text-heading font-semibold text-foreground">Answer window</p>

      <p className="text-body text-muted-foreground">
        After ATLAS reads back a calendar change or asks which one you meant, it listens this many
        seconds without the wake word.
      </p>

      <div className="flex flex-col gap-1.5">
        <Label htmlFor="follow-up-window-s">Seconds</Label>
        <Input
          id="follow-up-window-s"
          type="number"
          className="scroll-field"
          min={3}
          max={15}
          step={0.5}
          value={value}
          onChange={(event) => setValue(event.target.value)}
        />
      </div>

      {mutation.isError ? (
        <ErrorState
          message="Could not save the answer window."
          detail={mutation.error instanceof ApiError ? mutation.error.message : undefined}
          onRetry={() => void handleSave()}
        />
      ) : null}

      <SubmitButton className="w-full" onSubmit={handleSave} disabled={!value} pendingLabel="Saving…">
        Save answer window
      </SubmitButton>
    </div>
  )
}
