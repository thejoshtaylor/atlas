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
import { MACROS_QUERY_KEY, deleteMacroMutationOptions, fetchMacros, type Macro } from "@/lib/macros"
import {
  conflictedActionCount,
  deriveMacrosScreenState,
  formatActionCount,
  formatConflictSummary,
} from "./deriveMacrosScreenState"

// MACRO-03, D-11, 04-UI-SPEC.md's Focal Point row: "the card list itself
// ... 'New macro', then any denied-action flags on individual cards."
// Built from the twelve components Phase 3 already shipped and the four
// shared state primitives beside them -- no new component is added here
// (04-UI-SPEC.md's own re-enumeration, T-04-SC).

function MacroRow({ macro, disabled }: { macro: Macro; disabled: boolean }) {
  const [confirmOpen, setConfirmOpen] = React.useState(false)
  const remove = useMutation(deleteMacroMutationOptions)
  const conflicted = conflictedActionCount(macro)

  return (
    <li className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card p-4">
      <Link to={`/macros/${macro.id}`} className="flex min-w-0 flex-1 flex-col gap-1">
        <span className="truncate text-body font-medium text-foreground">{macro.phrase}</span>
        <span className="truncate text-label text-muted-foreground">{formatActionCount(macro.actions.length)}</span>
        {conflicted > 0 ? (
          <div className="flex flex-wrap gap-1.5">
            <Badge variant="denied">{formatConflictSummary(conflicted)}</Badge>
          </div>
        ) : null}
      </Link>
      <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="shrink-0"
          disabled={disabled || remove.isPending}
          onClick={() => setConfirmOpen(true)}
        >
          Delete
        </Button>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Delete macro</AlertDialogTitle>
            <AlertDialogDescription>
              {`Delete "${macro.phrase}"? This macro will no longer run, and its cached reply is discarded.`}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setConfirmOpen(false)}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setConfirmOpen(false)
                void remove.mutateAsync({ macroId: macro.id })
              }}
            >
              Delete macro
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  )
}

export function MacrosRoute() {
  const query = useQuery({ queryKey: MACROS_QUERY_KEY, queryFn: fetchMacros })
  const screen = deriveMacrosScreenState(query)
  const controlsDisabled = screen.kind !== "ready"

  const newMacroButton = (
    <Button asChild disabled={controlsDisabled}>
      <Link to="/macros/new">New macro</Link>
    </Button>
  )

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-display font-semibold">Macros</h1>
        {screen.kind === "ready" && screen.macros.length > 0 ? newMacroButton : null}
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        screen.macros.length === 0 ? (
          <EmptyState
            heading="No macros yet."
            body="A macro is a phrase that runs one or more actions without waiting on the model."
            action={newMacroButton}
          />
        ) : (
          <ul className="flex flex-col gap-2">
            {screen.macros.map((macro) => (
              <MacroRow key={macro.id} macro={macro} disabled={controlsDisabled} />
            ))}
          </ul>
        )
      ) : null}
    </div>
  )
}
