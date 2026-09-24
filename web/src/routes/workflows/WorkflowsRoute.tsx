import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Link } from "react-router-dom"
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
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { WORKFLOWS_QUERY_KEY, cancelWorkflowMutationOptions, fetchWorkflows, type WorkflowRun } from "@/lib/workflows"
import {
  cancelDialogCopy,
  deriveWorkflowsScreenState,
  formatRunCount,
  formatStepCount,
  lateCaption,
  runBadges,
  runConflictDisplay,
} from "./deriveWorkflowsScreenState"

// FLOW-10, D-16, 05-UI-SPEC.md's Focal Point row: "the card list itself
// ... this is the screen that answers 'what is about to happen without
// me' ... 'New workflow', then any Late/Running-now/fire-time-conflict
// flags on individual rows." Built from the same twelve shipped
// components and four shared state primitives the macro list uses (D-15)
// -- no new component is added here.

function WorkflowRow({ run, now, disabled }: { run: WorkflowRun; now: Date; disabled: boolean }) {
  const [confirmOpen, setConfirmOpen] = React.useState(false)
  const cancel = useMutation(cancelWorkflowMutationOptions)
  const conflict = runConflictDisplay(run)
  const dialogCopy = cancelDialogCopy(run)
  const caption = lateCaption(run, now)

  return (
    <li className="flex items-center justify-between gap-3 px-4 py-3">
      <Link to={`/workflows/${run.id}`} className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="truncate text-body font-medium text-foreground">{run.summary}</span>
        <span className="truncate text-label text-muted-foreground">{formatStepCount(run.step_count)}</span>
        <div className="flex flex-wrap gap-1.5">
          {runBadges(run).map((badge) => (
            <Badge key={badge.key} variant="secondary">
              {badge.label}
            </Badge>
          ))}
        </div>
        {caption ? <span className="text-label text-muted-foreground">{caption}</span> : null}
        {conflict.kind === "denied" ? (
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
      </Link>
      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="shrink-0"
          disabled={disabled || cancel.isPending}
          onClick={() => setConfirmOpen(true)}
        >
          Cancel
        </Button>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Cancel run</AlertDialogTitle>
            <AlertDialogDescription>{dialogCopy.body}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setConfirmOpen(false)}>Keep run</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmOpen(false)
                void cancel.mutateAsync({ runId: run.id })
              }}
            >
              {dialogCopy.confirmLabel}
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  )
}

export function WorkflowsRoute() {
  const query = useQuery({ queryKey: WORKFLOWS_QUERY_KEY, queryFn: fetchWorkflows })
  const screen = deriveWorkflowsScreenState(query)
  // Cancel is disabled on every row while the list itself is unknown --
  // acting against an unknown state is unsafe, the same instinct
  // 04-UI-SPEC.md's macro-list-load-failure row already established.
  const controlsDisabled = screen.kind !== "ready"
  const now = new Date()

  const newWorkflowButton = (
    <Button asChild disabled={controlsDisabled}>
      <Link to="/workflows/new">New workflow</Link>
    </Button>
  )

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-3">
        <div className="flex flex-col gap-1">
          <h1 className="text-display font-semibold">Pending runs</h1>
          {screen.kind === "ready" && screen.runs.length > 0 ? (
            <span className="text-label text-muted-foreground">{formatRunCount(screen.runs.length)}</span>
          ) : null}
        </div>
        {screen.kind === "ready" && screen.runs.length > 0 ? newWorkflowButton : null}
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        screen.runs.length === 0 ? (
          <EmptyState
            heading="Nothing scheduled."
            body="A scheduled workflow runs steps later, by voice or from here."
            action={newWorkflowButton}
          />
        ) : (
          <ul className="panel-list">
            {screen.runs.map((run) => (
              <WorkflowRow key={run.id} run={run} now={now} disabled={controlsDisabled} />
            ))}
          </ul>
        )
      ) : null}
    </div>
  )
}
