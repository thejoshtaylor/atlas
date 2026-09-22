import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import {
  CALIBRATION_QUERY_KEY,
  fetchLatestCalibration,
  runCalibrationMutationOptions,
  type EchoCalibrationResult,
} from "@/lib/calibration"
import { deriveCalibrationScreenState } from "./deriveCalibrationScreenState"

// WEB-03, D-18: this screen drives Phase 2's echo-path calibration from
// the browser -- it is a second caller of `GET /calibration/echo-path`
// and `POST /calibration/echo-path/run`, never a second measurement.
// It is built standalone, before plan 03-10's wizard exists: `onFinish`
// is optional so 03-10 can mount this as one wizard step later, passing
// its own "advance to the next step" handler, while a direct visit today
// (this plan's own held-out human-check) falls back to leaving the
// screen where it is -- there is no wizard to finish yet.
//
// No `useEffect` appears anywhere in this file. That is not a style
// preference: it is what makes "no calibration is started automatically
// on page load" true by construction rather than by care. The initial
// `useQuery` below reads the last stored result; nothing in this module
// ever calls the run mutation except the two explicit click handlers.

const AGC_VERDICT_LABEL: Record<string, string> = {
  present: "Detected — the camera changes level while it plays back",
  absent: "Not detected — the camera holds a steady level",
  indeterminate: "Could not be determined from this run",
}

const FAILURE_HEADING: Record<string, string> = {
  disabled: "Calibration is turned off.",
  "in-progress": "A calibration run is already in progress.",
  unmeasurable: "The run couldn't measure anything.",
  unknown: "Calibration couldn't complete. Check the camera is reachable and try again.",
}

function formatDelay(delaySeconds: number): string {
  return `${Math.round(delaySeconds * 1000)} ms`
}

function formatAge(ageDays: number): string {
  const hours = ageDays * 24
  if (hours < 1) return "less than an hour ago"
  if (ageDays < 1) {
    const roundedHours = Math.round(hours)
    return `${roundedHours} hour${roundedHours === 1 ? "" : "s"} ago`
  }
  const days = Math.round(ageDays)
  return `${days} day${days === 1 ? "" : "s"} ago`
}

/**
 * The three measured results -- 03-UI-SPEC.md's Focal Point table: "this
 * is the one screen where the outcome outranks the button." Labelled in
 * words the operator can act on, never as `delay_s`/`gain`/`agc_verdict`
 * themselves; confidence sits beside them, in the header row above.
 */
// D-260922-eca: an echo-cancelling camera (`aec` mode) removes its own
// speaker output before its microphone ever records it, so a run against
// one measures no delay and no gain to show -- `delay_s`/`gain` are
// meaningless zeros in that case (`calibration/runner.py`), never a real
// measurement, and showing them as if they were would read as a passing
// score for a camera that simply gave the probe nothing to find.
function CalibrationResults({ result }: { result: EchoCalibrationResult }) {
  if (result.echo_cancelled) {
    return (
      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
        <div className="flex items-baseline justify-between gap-2">
          <span className="text-label text-muted-foreground">Confidence</span>
          <span className="text-label text-muted-foreground">{Math.round(result.confidence * 100)}%</span>
        </div>
        <p className="text-body text-foreground">
          No echo came back. The camera cancels its own speaker from its microphone, so the
          assistant will not hear itself. If you did not hear the test sound, check the speaker
          and run the test again.
        </p>
        {result.placement_note ? (
          <p className="text-label text-muted-foreground">Recorded from: {result.placement_note}</p>
        ) : null}
      </div>
    )
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-label text-muted-foreground">Confidence</span>
        <span className="text-label text-muted-foreground">{Math.round(result.confidence * 100)}%</span>
      </div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-body text-foreground">Round-trip delay</span>
        <span className="text-heading font-semibold">{formatDelay(result.delay_s)}</span>
      </div>
      <div className="flex items-baseline justify-between gap-2">
        <span className="text-body text-foreground">Arrival level</span>
        <span className="text-heading font-semibold">{result.gain.toFixed(2)}</span>
      </div>
      <div className="flex flex-col gap-1">
        <span className="text-body text-foreground">Automatic gain control</span>
        <span className="text-heading font-semibold">
          {AGC_VERDICT_LABEL[result.agc_verdict] ?? result.agc_verdict}
        </span>
      </div>
      {result.placement_note ? (
        <p className="text-label text-muted-foreground">Recorded from: {result.placement_note}</p>
      ) : null}
    </div>
  )
}

export interface CalibrationRouteProps {
  /** Passed by plan 03-10's wizard once it exists. Defaults to leaving this screen in place. */
  onFinish?: () => void
}

export function CalibrationRoute({ onFinish }: CalibrationRouteProps = {}) {
  const navigate = useNavigate()
  const handleFinish = onFinish ?? (() => navigate("/"))

  const [placementNote, setPlacementNote] = React.useState("")

  const latest = useQuery({
    queryKey: CALIBRATION_QUERY_KEY,
    queryFn: fetchLatestCalibration,
  })
  const run = useMutation(runCalibrationMutationOptions)

  const screen = deriveCalibrationScreenState({ latest, run })

  // `.catch` here swallows the rejection `SubmitButton`'s guard would
  // otherwise leave unhandled -- the failure itself is already tracked
  // reactively through `run.status`/`run.error`, which is what
  // `deriveCalibrationScreenState` reads to render the failed state.
  const handleRun = () => run.mutateAsync({ placementNote }).catch(() => undefined)

  const handleRetry = () => {
    if (screen.kind !== "failed") return
    if (screen.reason === "in-progress" || screen.source === "latest") {
      run.reset()
      void latest.refetch()
      return
    }
    void handleRun()
  }

  const showRunControls = screen.kind === "empty" || screen.kind === "stored" || screen.kind === "success"
  const showFinish = screen.kind === "stored" || screen.kind === "success"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Test the microphone and speaker</h1>
        <p className="text-body text-muted-foreground">
          Plays a short sound through the speaker and measures how it returns through the
          microphone — the same measurement <code>scripts/dev-calibrate-echo.sh</code> runs from
          the command line, started here instead.
        </p>
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={2} /> : null}

      {screen.kind === "empty" ? (
        <EmptyState
          heading="No calibration yet."
          body="The microphone and speaker haven't been tested from the browser yet. Camera barge-in stays off until a real measurement exists."
        />
      ) : null}

      {screen.kind === "stored" ? (
        <div className="flex flex-col gap-2">
          <p className="text-label text-muted-foreground">Last measured {formatAge(screen.result.age_days)}</p>
          <CalibrationResults result={screen.result} />
        </div>
      ) : null}

      {screen.kind === "running" ? (
        <div
          role="status"
          aria-live="polite"
          className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border p-8 text-center"
        >
          <p className="text-body text-foreground">Testing microphone and speaker…</p>
        </div>
      ) : null}

      {screen.kind === "success" ? (
        <div className="flex flex-col gap-2">
          <p className="text-label text-muted-foreground">Just measured</p>
          <CalibrationResults result={screen.result} />
        </div>
      ) : null}

      {screen.kind === "failed" ? (
        <ErrorState
          message={FAILURE_HEADING[screen.reason]}
          detail={screen.detail}
          retryLabel={screen.reason === "in-progress" ? "Check again" : "Retry"}
          onRetry={handleRetry}
        />
      ) : null}

      {showRunControls ? (
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="placement-note">Where you're standing</Label>
            <Input
              id="placement-note"
              className="scroll-field"
              placeholder="e.g. kitchen counter, 2m from camera"
              value={placementNote}
              onChange={(event) => setPlacementNote(event.target.value)}
            />
          </div>
          <SubmitButton onSubmit={handleRun} pendingLabel="Testing microphone and speaker…">
            {screen.kind === "empty" ? "Test microphone and speaker" : "Run again"}
          </SubmitButton>
          {showFinish ? (
            <Button type="button" variant="outline" onClick={handleFinish}>
              Finish setup
            </Button>
          ) : null}
        </div>
      ) : null}
    </div>
  )
}
