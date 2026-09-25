import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
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
import { Button } from "@/components/ui/button"
import { ApiError } from "@/lib/api"
import {
  fetchGoogleStyle,
  googleStyleQueryKey,
  relearnGoogleStyleMutationOptions,
} from "@/lib/google"
import { deriveWritingStyleState, shouldPoll } from "./deriveWritingStyleState"

// D-19, D-20 (09-11-PLAN.md): one account's writing style -- its learned
// status, its profile, its samples, and its Gmail signature -- plus the
// one deliberate way to refresh it. Mounted at the end of
// `GoogleAccountRoute.tsx`, before Unlink.

const RELEARN_TITLE = "Re-learn the writing style?"
const RELEARN_BODY =
  "This reads the latest Sent mail again and replaces the profile, including your edits."

// D-20: no automatic re-learning -- polling every 3 seconds while
// `learning` only refreshes THIS account's own status, it never starts a
// new learn itself.
const POLL_INTERVAL_MS = 3000

export function WritingStyleSection({ accountId }: { accountId: number }) {
  const query = useQuery({
    queryKey: googleStyleQueryKey(accountId),
    queryFn: () => fetchGoogleStyle(accountId),
    refetchInterval: (q) => (q.state.data && shouldPoll(q.state.data.status) ? POLL_INTERVAL_MS : false),
  })
  const style = query.data

  const relearn = useMutation(relearnGoogleStyleMutationOptions)
  const [relearnOpen, setRelearnOpen] = React.useState(false)
  const [relearnError, setRelearnError] = React.useState<string | null>(null)

  const handleRelearn = async () => {
    setRelearnError(null)
    try {
      await relearn.mutateAsync({ accountId })
    } catch (err) {
      // `AlertDialogAction` is Radix's `Dialog.Close` (see the primitive's
      // own source) -- it closes the dialog on click unconditionally, so a
      // failure's error text is rendered in the section body below, never
      // inside `AlertDialogContent`, which is already unmounted by the
      // time this catch runs.
      setRelearnError(err instanceof ApiError ? err.message : "Couldn't re-learn the style. Try again.")
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <p className="text-heading font-semibold text-foreground">Writing style</p>
      {style ? <p className="text-body text-muted-foreground">{deriveWritingStyleState(style)}</p> : null}

      <AlertDialog open={relearnOpen} onOpenChange={setRelearnOpen}>
        <Button
          type="button"
          variant="outline"
          disabled={style ? shouldPoll(style.status) : true}
          onClick={() => setRelearnOpen(true)}
          className="sm:self-start"
        >
          Re-learn style
        </Button>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>{RELEARN_TITLE}</AlertDialogTitle>
            <AlertDialogDescription>{RELEARN_BODY}</AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setRelearnOpen(false)}>Cancel</AlertDialogCancel>
            <AlertDialogAction disabled={relearn.isPending} onClick={() => void handleRelearn()}>
              Re-learn
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
      {relearnError ? <p className="text-body text-destructive">{relearnError}</p> : null}
    </div>
  )
}
