import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import { createSpeakerMutationOptions, speakersQueryOptions, type Speaker } from "@/lib/speakers"
import { deriveSpeakersScreenState, formatEnrollmentProgress } from "./deriveSpeakersScreenState"
import { EnrollmentPanel } from "./EnrollmentPanel"

// D-01 through D-04: an admin manages household members here -- listing
// enrollment progress, adding a member by name, enrolling one through the
// real Pi microphone (Task 2), and deleting one along with its voice data
// (Task 3). Every write here sits behind the admin `RequireRole` block in
// `App.tsx` (presentation only, matching `EdgeDevicesRoute.tsx`'s own
// convention) -- `require_role(Role.ADMIN)` (`routes/speakers.py`) is what
// actually holds the line.

function SpeakerRow({ speaker }: { speaker: Speaker }) {
  // A fresh `key` on each open (`enrollSession`) remounts `EnrollmentPanel`
  // with a clean reducer -- "Re-record phrases" starts at phrase 0 again
  // even if a previous session for this same member was left mid-way.
  const [enrollmentOpen, setEnrollmentOpen] = React.useState(false)
  const [enrollSession, setEnrollSession] = React.useState(0)
  const isEnrolled = speaker.enrolled_phrases >= speaker.required_phrases

  return (
    <li className="flex flex-col gap-2 px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-col">
          <span className="truncate text-body font-medium text-foreground">{speaker.display_name}</span>
          <span className="truncate text-label text-muted-foreground">{formatEnrollmentProgress(speaker)}</span>
        </div>
        <Button
          type="button"
          variant="outline"
          size="sm"
          onClick={() => {
            setEnrollmentOpen((wasOpen) => !wasOpen)
            setEnrollSession((count) => count + 1)
          }}
        >
          {isEnrolled ? "Re-record phrases" : "Enroll"}
        </Button>
      </div>
      {enrollmentOpen ? <EnrollmentPanel key={enrollSession} speaker={speaker} /> : null}
    </li>
  )
}

export function SpeakersRoute() {
  const speakers = useQuery(speakersQueryOptions)
  const screen = deriveSpeakersScreenState({ speakers })

  const [displayName, setDisplayName] = React.useState("")
  const [createError, setCreateError] = React.useState<string | null>(null)

  const createSpeaker = useMutation(createSpeakerMutationOptions)

  const handleAddSpeaker = async () => {
    setCreateError(null)
    try {
      await createSpeaker.mutateAsync({ displayName, linkedUserId: null })
      setDisplayName("")
    } catch (error) {
      setCreateError(error instanceof ApiError ? error.message : "Couldn't add the member. Try again.")
    }
  }

  const controlsDisabled = screen.kind !== "ready"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Speakers</h1>
        <p className="text-body text-muted-foreground">
          A speaker is a household member the assistant can recognize by voice. A voice match
          personalizes replies. It never grants permission.
        </p>
      </div>

      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
        <p className="text-heading font-semibold text-foreground">Add member</p>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="speaker-display-name">Name</Label>
          <Input
            id="speaker-display-name"
            value={displayName}
            disabled={controlsDisabled}
            onChange={(event) => setDisplayName(event.target.value)}
          />
        </div>
        {createError ? <p className="text-label text-destructive">{createError}</p> : null}
        <SubmitButton
          onSubmit={handleAddSpeaker}
          disabled={controlsDisabled || displayName.length === 0}
          pendingLabel="Adding…"
        >
          Add member
        </SubmitButton>
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void speakers.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        screen.speakers.length === 0 ? (
          <EmptyState
            heading="No household members yet."
            body="Add a member, then record five phrases through the Pi."
          />
        ) : (
          <ul className="panel-list">
            {screen.speakers.map((speaker) => (
              <SpeakerRow key={speaker.id} speaker={speaker} />
            ))}
          </ul>
        )
      ) : null}
    </div>
  )
}
