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
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/state/SubmitButton"
import { Textarea } from "@/components/ui/textarea"
import { ApiError } from "@/lib/api"
import {
  fetchGoogleStyle,
  googleStyleQueryKey,
  relearnGoogleStyleMutationOptions,
  saveGoogleStyleMutationOptions,
  type GoogleStyle,
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

// 09-09's own `StyleUpdateRequest` limit -- shown here as a live count,
// enforced server-side.
const MAX_PROFILE_CHARS = 4000

function StyleProfileEditor({ accountId, style }: { accountId: number; style: GoogleStyle }) {
  // Seeded once from the loaded style at mount, like `AccountDetail`'s own
  // `labelDraft` -- `StyleProfileEditor` only ever mounts once `style` is
  // loaded (its caller gates on that), so a lazy initial value is enough;
  // no effect is needed, and a background poll (while `learning`) never
  // overwrites text the admin is mid-edit on, since only the initial
  // render reads this value.
  const [profileDraft, setProfileDraft] = React.useState(style.profile)

  const effectiveProfile = profileDraft
  const save = useMutation(saveGoogleStyleMutationOptions)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const [samplesOpen, setSamplesOpen] = React.useState(false)

  const handleSave = async () => {
    setSaveError(null)
    try {
      await save.mutateAsync({ accountId, profile: effectiveProfile })
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Couldn't save this profile. Try again.")
      throw err
    }
  }

  return (
    <>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="google-style-profile">Style profile</Label>
        <Textarea
          id="google-style-profile"
          rows={6}
          value={effectiveProfile}
          onChange={(event) => setProfileDraft(event.target.value)}
        />
        <p className="text-label text-muted-foreground">
          {`${effectiveProfile.length} / ${MAX_PROFILE_CHARS.toLocaleString()} characters`}
        </p>
      </div>
      {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
      <SubmitButton onSubmit={handleSave} disabled={effectiveProfile === style.profile}>
        Save profile
      </SubmitButton>

      {style.samples.length > 0 ? (
        <div className="flex flex-col gap-2">
          <Button
            type="button"
            variant="outline"
            size="sm"
            className="sm:self-start"
            onClick={() => setSamplesOpen((open) => !open)}
          >
            {samplesOpen ? `Hide samples (${style.samples.length})` : `Show samples (${style.samples.length})`}
          </Button>
          {samplesOpen ? (
            <ul className="flex flex-col gap-2">
              {style.samples.map((sample, index) => (
                <li
                  key={index}
                  className="readout whitespace-pre-wrap rounded-md border border-border bg-muted p-2 text-body text-foreground"
                >
                  {sample}
                </li>
              ))}
            </ul>
          ) : null}
        </div>
      ) : null}

      <div className="flex flex-col gap-1.5">
        <p className="text-label font-medium text-foreground">Signature from Gmail</p>
        {style.signature_text ? (
          <p className="readout whitespace-pre-wrap text-body text-foreground">{style.signature_text}</p>
        ) : (
          <p className="text-body text-muted-foreground">No Gmail signature.</p>
        )}
      </div>
    </>
  )
}

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

      {style ? <StyleProfileEditor accountId={accountId} style={style} /> : null}

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
