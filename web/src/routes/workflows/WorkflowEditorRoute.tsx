import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { ChevronDown, ChevronUp } from "lucide-react"
import { Link, useNavigate, useParams } from "react-router-dom"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ErrorState } from "@/components/state/ErrorState"
import { SubmitButton } from "@/components/state/SubmitButton"
import {
  cancelWorkflowMutationOptions,
  createWorkflowMutationOptions,
  fetchWorkflow,
  replaceWorkflowStepsMutationOptions,
  workflowQueryKey,
  type ConflictAnnotation,
  type WorkflowStep,
} from "@/lib/workflows"
import {
  COMPOSER_KEY,
  draftStepsToInput,
  formatStepSummary,
  isFirstStep,
  isLastStep,
  useWorkflowDraftStore,
} from "@/stores/workflowDraftStore"
import { cancelDialogCopy, lateCaption, runBadges } from "./deriveWorkflowsScreenState"
import {
  BLANK_SUMMARY_SAVE_BLOCKED_REASON,
  deriveWorkflowEditorState,
  editorLocked,
  saveBlockedByBlankSummary,
  saveBlockedByPastSchedule,
  saveBlockedByZeroSteps,
  speakStepPrecacheState,
  stepConflictDisplay,
  stepIsValid,
} from "./deriveWorkflowEditorState"

// FLOW-09, D-11, D-15, 05-UI-SPEC.md's Focal Point row: "the ordered step
// list and the 'Run at' schedule field together ... 'Save workflow', then
// 'Cancel run'." The densest row this product has attempted (a step's
// shape varies by kind) built entirely from shapes this product already
// ships -- no new component, no new token (D-15). Every rule this screen
// enforces inline is enforced again on the route in plan 05-04 (T-05-31);
// these blocks are for the operator's benefit, not the boundary's.

export function WorkflowEditorRoute() {
  const params = useParams<{ id: string }>()
  const navigate = useNavigate()
  const runId = params.id ? Number(params.id) : null

  const runQuery = useQuery({
    queryKey: runId !== null ? workflowQueryKey(runId) : ["workflows", "new"],
    queryFn: () => fetchWorkflow(runId as number),
    enabled: runId !== null,
  })
  const screen = deriveWorkflowEditorState(runId === null ? null : runQuery)

  const summary = useWorkflowDraftStore((state) => state.summary)
  const runAt = useWorkflowDraftStore((state) => state.runAt)
  const steps = useWorkflowDraftStore((state) => state.steps)
  const composer = useWorkflowDraftStore((state) => state.composer)
  const setSummary = useWorkflowDraftStore((state) => state.setSummary)
  const setRunAt = useWorkflowDraftStore((state) => state.setRunAt)
  const updateStep = useWorkflowDraftStore((state) => state.updateStep)
  const setStepKind = useWorkflowDraftStore((state) => state.setStepKind)
  const addStep = useWorkflowDraftStore((state) => state.addStep)
  const removeStep = useWorkflowDraftStore((state) => state.removeStep)
  const moveStepUp = useWorkflowDraftStore((state) => state.moveStepUp)
  const moveStepDown = useWorkflowDraftStore((state) => state.moveStepDown)
  const loadRun = useWorkflowDraftStore((state) => state.loadRun)
  const loadBlank = useWorkflowDraftStore((state) => state.loadBlank)

  // The draft only ever loads from the server once per record version --
  // keyed on `id:updated_at`, the exact rule `MacroEditorRoute.tsx`
  // already established, so a background refetch of the *same* version
  // never clobbers an operator's unsaved typing.
  const loadedKeyRef = React.useRef<string | null>(null)
  React.useEffect(() => {
    if (screen.kind !== "ready") return
    const key = screen.run ? `${screen.run.id}:${screen.run.updated_at}` : "new"
    if (loadedKeyRef.current === key) return
    loadedKeyRef.current = key
    if (screen.run) loadRun(screen.run)
    else loadBlank()
  }, [screen, loadRun, loadBlank])

  const [synthesisFailedMessage, setSynthesisFailedMessage] = React.useState<string | null>(null)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const [cancelConfirmOpen, setCancelConfirmOpen] = React.useState(false)

  const createWorkflow = useMutation(createWorkflowMutationOptions)
  const replaceSteps = useMutation(replaceWorkflowStepsMutationOptions)
  const cancelWorkflow = useMutation(cancelWorkflowMutationOptions)

  const run = screen.kind === "ready" ? screen.run : null
  const locked = editorLocked(run)
  const recordLoading = screen.kind !== "ready"
  const now = new Date()

  const zeroStepsReason = saveBlockedByZeroSteps(steps)
  const pastScheduleReason = runId === null ? saveBlockedByPastSchedule(runAt, now) : null
  const blankSummary = runId === null && saveBlockedByBlankSummary(summary)
  const saveDisabled = recordLoading || locked || zeroStepsReason !== null || pastScheduleReason !== null || blankSummary

  const canAddStep = stepIsValid(composer) && !locked

  const conflictByKey = new Map<string, ConflictAnnotation>(run ? run.steps.map((s) => [String(s.id), s.conflict]) : [])
  const serverStepByKey = new Map<string, WorkflowStep>(run ? run.steps.map((s) => [String(s.id), s]) : [])

  const handleSave = async () => {
    setSaveError(null)
    const stepsInput = draftStepsToInput(steps)
    try {
      if (runId === null) {
        const result = await createWorkflow.mutateAsync({ summary, run_at: runAt, steps: stepsInput })
        setSynthesisFailedMessage(result.reply_synthesis_degraded ? result.reply_synthesis_message : null)
        navigate(`/workflows/${result.id}`, { replace: true })
      } else {
        const result = await replaceSteps.mutateAsync({ runId, steps: stepsInput })
        setSynthesisFailedMessage(result.reply_synthesis_degraded ? result.reply_synthesis_message : null)
      }
    } catch {
      setSaveError("Couldn't save this workflow. Try again.")
    }
  }

  const handleCancelRun = async () => {
    if (runId === null) return
    await cancelWorkflow.mutateAsync({ runId })
    navigate("/workflows")
  }

  const title = run ? run.summary : "New workflow"
  const canCancelRun = run !== null && (run.status === "pending" || run.status === "firing")
  const dialogCopy = run ? cancelDialogCopy(run) : null
  const caption = run ? lateCaption(run, now) : null
  const earliestDueAt = run && run.steps.length > 0 ? run.steps[0].due_at : null

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="truncate text-display font-semibold">{title}</h1>
        {run ? (
          <div className="flex flex-wrap gap-1.5">
            {runBadges(run).map((badge) => (
              <Badge key={badge.key} variant="secondary">
                {badge.label}
              </Badge>
            ))}
          </div>
        ) : null}
        {caption ? <span className="text-label text-muted-foreground">{caption}</span> : null}
      </div>

      {screen.kind === "loading" ? (
        <div
          role="status"
          aria-live="polite"
          className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border p-8 text-center"
        >
          <p className="text-body text-foreground">Loading workflow…</p>
        </div>
      ) : null}

      {screen.kind === "error" ? (
        <div className="flex flex-col gap-3">
          <ErrorState message={screen.message} onRetry={() => void runQuery.refetch()} />
          <Link to="/workflows" className="touch-target flex items-center text-body text-primary underline-offset-4 hover:underline">
            Back to pending runs
          </Link>
        </div>
      ) : null}

      {screen.kind === "ready" ? (
        <>
          {runId === null ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="workflow-summary">Summary</Label>
              <Input
                id="workflow-summary"
                required
                placeholder="What to say to cancel this run later"
                value={summary}
                onChange={(event) => setSummary(event.target.value)}
              />
              {blankSummary ? <p className="text-body text-destructive">{BLANK_SUMMARY_SAVE_BLOCKED_REASON}</p> : null}
            </div>
          ) : null}

          {runId === null ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="workflow-run-at">Run at</Label>
              <Input
                id="workflow-run-at"
                type="datetime-local"
                required
                value={runAt}
                onChange={(event) => setRunAt(event.target.value)}
              />
              {pastScheduleReason ? <p className="text-body text-destructive">{pastScheduleReason}</p> : null}
            </div>
          ) : (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="workflow-run-at">Run at</Label>
              <Input id="workflow-run-at" type="datetime-local" disabled value="" />
              <p className="text-label text-muted-foreground">
                {earliestDueAt
                  ? `Scheduled for ${new Date(earliestDueAt).toLocaleString()}. The schedule can't be changed after creation.`
                  : "The schedule can't be changed after creation."}
              </p>
            </div>
          )}

          <div className="flex flex-col gap-2">
            <p className="text-heading font-semibold text-foreground">Steps</p>
            {zeroStepsReason ? <p className="text-body text-destructive">{zeroStepsReason}</p> : null}

            <ol className="flex flex-col gap-2">
              {steps.map((step, index) => {
                const conflict = stepConflictDisplay(conflictByKey.get(step.key) ?? "ok")
                const serverStep = serverStepByKey.get(step.key) ?? null
                const precache = step.kind === "speak" ? speakStepPrecacheState(serverStep, synthesisFailedMessage) : null
                return (
                  <li key={step.key} className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-1 items-baseline gap-2">
                        <span className="text-label text-muted-foreground">{index + 1}.</span>
                        <span className="truncate text-body font-medium text-foreground">{formatStepSummary(step)}</span>
                      </div>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="shrink-0"
                        disabled={locked}
                        onClick={() => removeStep(step.key)}
                      >
                        Remove
                      </Button>
                    </div>
                    {conflict.kind === "denied" || conflict.kind === "not_found" ? (
                      <div className="flex flex-col gap-1">
                        <Badge variant="denied" className="w-fit">
                          {conflict.text}
                        </Badge>
                        <span className="text-label text-muted-foreground">
                          Checked against today&rsquo;s policy — checked again the moment each step fires.
                        </span>
                      </div>
                    ) : null}
                    {conflict.kind === "unknown" ? <p className="text-label text-muted-foreground">{conflict.text}</p> : null}
                    {precache === "cached" ? <Badge variant="secondary">Reply cached · ready</Badge> : null}
                    {precache === "failed" && synthesisFailedMessage ? (
                      <p className="text-body text-destructive">{synthesisFailedMessage}</p>
                    ) : null}
                    <div className="flex justify-end gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-sm"
                        aria-label="Move up"
                        disabled={locked || isFirstStep(steps, step.key)}
                        onClick={() => moveStepUp(step.key)}
                      >
                        <ChevronUp className="size-4" />
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-sm"
                        aria-label="Move down"
                        disabled={locked || isLastStep(steps, step.key)}
                        onClick={() => moveStepDown(step.key)}
                      >
                        <ChevronDown className="size-4" />
                      </Button>
                    </div>
                  </li>
                )
              })}
            </ol>

            {!locked ? (
              <div className="flex flex-col gap-3 rounded-lg border border-dashed border-border p-4">
                <RadioGroup
                  value={composer.kind}
                  onValueChange={(value) => setStepKind(COMPOSER_KEY, value as typeof composer.kind)}
                  aria-label="Step kind"
                >
                  <div className="touch-target flex items-center gap-2">
                    <RadioGroupItem value="wait" id="kind-wait" />
                    <Label htmlFor="kind-wait">Wait</Label>
                  </div>
                  <div className="touch-target flex items-center gap-2">
                    <RadioGroupItem value="call_service" id="kind-call-service" />
                    <Label htmlFor="kind-call-service">Call a service</Label>
                  </div>
                  <div className="touch-target flex items-center gap-2">
                    <RadioGroupItem value="speak" id="kind-speak" />
                    <Label htmlFor="kind-speak">Speak</Label>
                  </div>
                </RadioGroup>

                {composer.kind === "wait" ? (
                  <div className="flex flex-col gap-2">
                    <div className="flex flex-col gap-1.5">
                      <Label htmlFor="wait-duration">Duration</Label>
                      <Input
                        id="wait-duration"
                        type="number"
                        min={0}
                        value={composer.durationValue}
                        onChange={(event) => updateStep(COMPOSER_KEY, { durationValue: event.target.value })}
                      />
                    </div>
                    <RadioGroup
                      value={composer.durationUnit}
                      onValueChange={(value) => updateStep(COMPOSER_KEY, { durationUnit: value as typeof composer.durationUnit })}
                      aria-label="Duration unit"
                    >
                      <div className="touch-target flex items-center gap-2">
                        <RadioGroupItem value="seconds" id="unit-seconds" />
                        <Label htmlFor="unit-seconds">Seconds</Label>
                      </div>
                      <div className="touch-target flex items-center gap-2">
                        <RadioGroupItem value="minutes" id="unit-minutes" />
                        <Label htmlFor="unit-minutes">Minutes</Label>
                      </div>
                      <div className="touch-target flex items-center gap-2">
                        <RadioGroupItem value="hours" id="unit-hours" />
                        <Label htmlFor="unit-hours">Hours</Label>
                      </div>
                    </RadioGroup>
                  </div>
                ) : null}

                {composer.kind === "call_service" ? (
                  <div className="flex flex-col gap-2">
                    <div className="flex flex-col gap-1.5">
                      <Label htmlFor="cs-domain">Domain</Label>
                      <Input
                        id="cs-domain"
                        placeholder="light"
                        value={composer.domain}
                        onChange={(event) => updateStep(COMPOSER_KEY, { domain: event.target.value })}
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Label htmlFor="cs-service">Service</Label>
                      <Input
                        id="cs-service"
                        placeholder="turn_off"
                        value={composer.service}
                        onChange={(event) => updateStep(COMPOSER_KEY, { service: event.target.value })}
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Label htmlFor="cs-entity">Entity id</Label>
                      <Input
                        id="cs-entity"
                        placeholder="light.bedroom"
                        value={composer.entityId}
                        onChange={(event) => updateStep(COMPOSER_KEY, { entityId: event.target.value })}
                      />
                    </div>
                    <div className="flex flex-col gap-1.5">
                      <Label htmlFor="cs-transition">Transition (seconds)</Label>
                      <Input
                        id="cs-transition"
                        placeholder="optional"
                        value={composer.transition}
                        onChange={(event) => updateStep(COMPOSER_KEY, { transition: event.target.value })}
                      />
                    </div>
                  </div>
                ) : null}

                {composer.kind === "speak" ? (
                  <div className="flex flex-col gap-1.5">
                    <Label htmlFor="speak-text">Words to say</Label>
                    <Input
                      id="speak-text"
                      className="scroll-field"
                      value={composer.text}
                      onChange={(event) => updateStep(COMPOSER_KEY, { text: event.target.value })}
                    />
                  </div>
                ) : null}

                <Button type="button" variant="outline" onClick={() => addStep()} disabled={!canAddStep}>
                  Add step
                </Button>
              </div>
            ) : null}
          </div>

          {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
          {!locked ? (
            <SubmitButton onSubmit={handleSave} disabled={saveDisabled}>
              Save workflow
            </SubmitButton>
          ) : null}

          {canCancelRun && dialogCopy ? (
            <div className="flex flex-col gap-2 border-t border-border pt-4">
              <AlertDialog open={cancelConfirmOpen} onOpenChange={setCancelConfirmOpen}>
                <Button type="button" variant="destructive" onClick={() => setCancelConfirmOpen(true)}>
                  Cancel run
                </Button>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Cancel run</AlertDialogTitle>
                    <AlertDialogDescription>{dialogCopy.body}</AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel onClick={() => setCancelConfirmOpen(false)}>Keep run</AlertDialogCancel>
                    <AlertDialogAction
                      variant="destructive"
                      onClick={() => {
                        setCancelConfirmOpen(false)
                        void handleCancelRun()
                      }}
                    >
                      {dialogCopy.confirmLabel}
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
