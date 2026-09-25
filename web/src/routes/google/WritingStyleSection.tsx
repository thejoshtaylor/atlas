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

// R2-WR-10: shown when a learn finished while the admin held an unsaved
// edit -- the edit is kept, never silently replaced.
const RELEARNED_UNDER_EDITS =
  "A new profile was learned. Your unsaved edits are still here. Save them to replace it, or discard them."

function StyleProfileEditor({ accountId, style }: { accountId: number; style: GoogleStyle }) {
  // Seeded once from the loaded style at mount, like `AccountDetail`'s own
  // `labelDraft` -- but `style` is truthy (and this component mounts) the
  // moment the FIRST `GET .../style` resolves, which is routinely while
  // `status === "learning"` (a new account's `learn_style` is scheduled
  // in the background before the redirect). The draft MUST resync the one
  // time `status` actually leaves `learning` -- otherwise it stays frozen
  // at its stale (often empty) mount-time value, and "Save profile" PUTs
  // it back over the newly-learned server profile (C-CR-02).
  //
  // R2-WR-10: that resync must never discard an unsaved edit. The draft
  // resyncs only when it still equals the profile it was based on (the
  // server profile just before the learn finished). Otherwise the edit is
  // kept and `RELEARNED_UNDER_EDITS` offers a discard. The check runs
  // during render -- React's own "adjust state when a prop changes"
  // pattern -- so no effect writes state after the paint.
  const [profileDraft, setProfileDraft] = React.useState(style.profile)
  const [seenStatus, setSeenStatus] = React.useState(style.status)
  const [seenProfile, setSeenProfile] = React.useState(style.profile)
  const [relearnedUnderEdits, setRelearnedUnderEdits] = React.useState(false)
  if (seenStatus !== style.status || seenProfile !== style.profile) {
    if (shouldPoll(seenStatus) && !shouldPoll(style.status)) {
      if (profileDraft === seenProfile) {
        setProfileDraft(style.profile)
      } else if (profileDraft !== style.profile) {
        setRelearnedUnderEdits(true)
      }
    }
    setSeenStatus(style.status)
    setSeenProfile(style.profile)
  }

  // R2-WR-10: while a learn runs, the server overwrites the profile when
  // it finishes -- editing or saving now would be lost, so both wait.
  const learning = shouldPoll(style.status)
  const effectiveProfile = profileDraft
  const showRelearnedNotice = relearnedUnderEdits && profileDraft !== style.profile
  const save = useMutation(saveGoogleStyleMutationOptions)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const [samplesOpen, setSamplesOpen] = React.useState(false)

  const handleSave = async () => {
    setSaveError(null)
    try {
      await save.mutateAsync({ accountId, profile: effectiveProfile })
      setRelearnedUnderEdits(false)
    } catch (err) {
      setSaveError(err instanceof ApiError ? err.message : "Couldn't save this profile. Try again.")
      throw err
    }
  }

  const handleDiscard = () => {
    setProfileDraft(style.profile)
    setRelearnedUnderEdits(false)
  }

  return (
    <>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="google-style-profile">Style profile</Label>
        <Textarea
          id="google-style-profile"
          rows={6}
          value={effectiveProfile}
          disabled={learning}
          onChange={(event) => setProfileDraft(event.target.value)}
        />
        <p className="text-label text-muted-foreground">
          {`${effectiveProfile.length} / ${MAX_PROFILE_CHARS.toLocaleString()} characters`}
        </p>
      </div>
      {showRelearnedNotice ? (
        <div className="flex flex-col gap-2">
          <p className="text-body text-foreground">{RELEARNED_UNDER_EDITS}</p>
          <Button type="button" variant="outline" size="sm" className="sm:self-start" onClick={handleDiscard}>
            Discard my edits
          </Button>
        </div>
      ) : null}
      {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
      <SubmitButton onSubmit={handleSave} disabled={learning || effectiveProfile === style.profile}>
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
