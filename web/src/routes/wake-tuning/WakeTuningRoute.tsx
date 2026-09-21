import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { WAKE_EVENTS_QUERY_KEY, fetchWakeEvents, setWakeThresholdMutationOptions } from "@/lib/wakeTuning"
import { deriveWakeTuningScreenState, type WakeEventDisplay } from "./deriveWakeTuningScreenState"

// DBG-05, D-13, D-14a, D-15, D-16, 08-UI-SPEC.md's Focal Point row: "the
// histogram and the live-updating count ... the same 'outcome outranks
// the control' precedent 03-UI-SPEC.md gives the wizard's calibration
// step." Follows `CalibrationRoute.tsx`'s layout for that reason: the
// outcome sits above the control, the control above the expandable list.
//
// Deliberately does NOT render the household-audio disclosure
// (08-UI-SPEC.md's "Read this first"): this screen shows timestamps,
// source names and numeric scores, never a transcript or audio, so the
// sensitivity that sentence exists to name does not apply here.
// WakeTuningRoute.test.tsx asserts its absence so a later reader does not
// add it back "for consistency."

const DEFAULT_LIST_SIZE = 20

const HISTORICAL_OUTCOME_LABEL: Record<WakeEventDisplay["historicalOutcome"], string> = {
  started_turn: "Started a turn",
  blocked_below_threshold: "Blocked — below threshold",
  blocked_refractory: "Blocked — refractory window",
  blocked_media_playing: "Blocked — media playing",
}

function formatRecordedAt(recordedAt: string): string {
  return new Date(recordedAt).toLocaleString()
}

function EventRow({ event }: { event: WakeEventDisplay }) {
  return (
    <li className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-label text-muted-foreground">{formatRecordedAt(event.recordedAt)}</span>
        <Badge variant="outline">{event.source}</Badge>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <Badge variant={event.clearsPreviewedThreshold ? "secondary" : "outline"}>
          {event.clearsPreviewedThreshold ? "Clears" : "Below"}
        </Badge>
        <Badge variant="outline">{HISTORICAL_OUTCOME_LABEL[event.historicalOutcome]}</Badge>
      </div>
    </li>
  )
}

export function WakeTuningRoute() {
  const query = useQuery({ queryKey: WAKE_EVENTS_QUERY_KEY, queryFn: fetchWakeEvents })
  const setThreshold = useMutation(setWakeThresholdMutationOptions)

  // The previewed threshold: seeded once from the server's own live value
  // the first time real data lands, then never yanked back under the
  // operator's finger by a later re-fetch (e.g. the invalidation this
  // mutation's own `onSuccess` triggers).
  const [previewedThreshold, setPreviewedThreshold] = React.useState<number | null>(null)
  const seededRef = React.useRef(false)
  React.useEffect(() => {
    if (!seededRef.current && query.data) {
      setPreviewedThreshold(query.data.threshold)
      seededRef.current = true
    }
  }, [query.data])

  const [showAll, setShowAll] = React.useState(false)
  const [committedValue, setCommittedValue] = React.useState<number | null>(null)

  const threshold = previewedThreshold ?? query.data?.threshold ?? 0
  const screen = deriveWakeTuningScreenState(query, threshold)

  const handleThresholdChange = (value: number) => {
    if (Number.isNaN(value)) return
    setPreviewedThreshold(value)
    // A failed commit leaves the previewed value on screen rather than
    // reverting it (Copywriting Contract) -- but a *new* drag after a
    // failed or successful commit clears that commit's own transient
    // status line, since it now describes a value the operator has
    // already moved away from.
    setCommittedValue(null)
    setThreshold.reset()
  }

  const handleCommit = () =>
    setThreshold.mutateAsync({ threshold }).then((result) => {
      setCommittedValue(result.threshold)
    })

  const visibleEvents =
    screen.kind === "ready" ? (showAll ? screen.events : screen.events.slice(0, DEFAULT_LIST_SIZE)) : []
  const maxBucketCount = screen.kind === "ready" ? Math.max(1, ...screen.histogram.map((bucket) => bucket.count)) : 1

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Wake threshold</h1>
        <p className="text-body text-muted-foreground">
          Move the control to see which recorded wake attempts would have cleared it.
        </p>
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? <ErrorState message={screen.message} onRetry={() => void query.refetch()} /> : null}

      {screen.kind === "empty" ? (
        <EmptyState
          heading="No wake attempts recorded yet."
          body="This fills in the first time the wake word is spoken near a source."
        />
      ) : null}

      {screen.kind === "ready" ? (
        <div className="flex flex-col gap-6">
          <div className="flex flex-col gap-3">
            <div className="flex h-24 items-end gap-1" aria-hidden="true">
              {/* One column per bucket, split where the threshold falls
                  inside it: the clearing part and the below part are two
                  segments of the same bar, sized by their own counts. A
                  single shade per bucket could not be true of the bucket
                  the threshold lands in, which is the one the operator is
                  looking at while they drag. */}
              {screen.histogram.map((bucket, index) => (
                <div
                  key={index}
                  className="flex flex-1 flex-col justify-end"
                  style={{ height: `${Math.max(4, (bucket.count / maxBucketCount) * 100)}%` }}
                >
                  {bucket.count === 0 ? <div className="w-full flex-1 bg-border" /> : null}
                  {bucket.clearingCount > 0 ? (
                    <div className="w-full bg-foreground" style={{ flexGrow: bucket.clearingCount }} />
                  ) : null}
                  {bucket.belowCount > 0 ? (
                    <div className="w-full bg-border" style={{ flexGrow: bucket.belowCount }} />
                  ) : null}
                </div>
              ))}
            </div>
            <p className="text-body text-foreground">
              {screen.clearingCount} clear this threshold, {screen.belowCount} do not, out of {screen.totalScored}{" "}
              recorded {screen.engine} wake attempts.
            </p>
            {screen.notScoredCount > 0 ? (
              <p className="text-label text-muted-foreground">
                {screen.notScoredCount} earlier session{screen.notScoredCount === 1 ? "" : "s"} predate wake-score
                recording and aren't included above.
              </p>
            ) : null}
            {screen.capped ? (
              <p className="text-label text-muted-foreground">
                Older wake attempts than these exist and aren't included above.
              </p>
            ) : null}
            {!screen.engineGrades ? (
              <p className="text-label text-muted-foreground">
                The configured wake engine ({screen.engine}) reports the same score for every wake it recognizes.
                Moving this control will not change what it detects.
              </p>
            ) : null}
          </div>

          <div className="flex flex-col gap-3">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="wake-threshold-range">Threshold</Label>
              {/* Native range input, not a shadcn slider (`slider` is not
                  installed): a threshold is one continuous value, nothing
                  to pick between. `touch-target` composes the existing
                  44px minimum onto this thumb -- the track may stay
                  visually thin, the target may not (08-UI-SPEC.md
                  Spacing Scale). Previewing costs nothing -- no request
                  goes out until "Set as active threshold" is pressed. */}
              <input
                id="wake-threshold-range"
                type="range"
                className="touch-target w-full"
                min={0}
                max={1}
                step={0.01}
                value={threshold}
                onChange={(event) => handleThresholdChange(Number(event.target.value))}
              />
            </div>
            <Input
              id="wake-threshold-value"
              type="number"
              className="scroll-field"
              min={0}
              max={1}
              step={0.01}
              value={threshold}
              onChange={(event) => handleThresholdChange(event.target.valueAsNumber)}
            />
            <SubmitButton onSubmit={handleCommit} pendingLabel="Setting threshold…">
              Set as active threshold
            </SubmitButton>
            {committedValue !== null ? (
              <p className="text-label text-muted-foreground">
                Threshold set to {committedValue.toFixed(2)}. This takes effect immediately — no restart needed.
              </p>
            ) : null}
            {setThreshold.status === "error" ? (
              <p className="text-label text-muted-foreground">Couldn't set the threshold. Try again.</p>
            ) : null}
          </div>

          <div className="flex flex-col gap-2">
            <h2 className="text-heading font-semibold">Recorded wake attempts</h2>
            <ul className="flex flex-col gap-2">
              {visibleEvents.map((event) => (
                <EventRow key={event.id} event={event} />
              ))}
            </ul>
            {/* A rendering cutoff, not a new pagination API: every event
                is already fetched for the histogram above regardless
                (08-PATTERNS.md, 08-UI-SPEC.md's "populated" row) -- this
                departs from Phases 6/7's plain, unpaginated list on
                purpose, because this is the first list in this product
                whose realistic scale is hundreds rather than a dozen. */}
            {!showAll && screen.events.length > DEFAULT_LIST_SIZE ? (
              <Button type="button" variant="outline" onClick={() => setShowAll(true)}>
                Show all {screen.events.length} events
              </Button>
            ) : null}
          </div>
        </div>
      ) : null}
    </div>
  )
}
