import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import {
  assignVoiceMutationOptions,
  voiceInboxAudioUrl,
  voiceInboxQueryOptions,
  type AssignTarget,
  type VoiceInboxItem,
} from "@/lib/speakerInbox"
import type { Speaker } from "@/lib/speakers"

// 260929-j08: recent edge turns that no member matched, while their
// recordings are kept. The admin listens, then assigns the voice to a
// member. A voice match never grants permission (D-15). The transcript is
// untrusted text, so it is only ever a React text child, never raw HTML.

const NEW_MEMBER = "new"

function AssignForm({ item, speakers }: { item: VoiceInboxItem; speakers: Speaker[] }) {
  const assign = useMutation(assignVoiceMutationOptions)
  const [choice, setChoice] = React.useState<string | null>(null)
  const [newName, setNewName] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const idPrefix = `assign-${item.session_id}`

  const target: AssignTarget | null =
    choice === null
      ? null
      : choice === NEW_MEMBER
        ? newName.trim().length > 0
          ? { kind: "new", displayName: newName.trim() }
          : null
        : { kind: "existing", speakerId: Number(choice) }

  const handleAssign = async () => {
    if (target === null) return
    setError(null)
    try {
      await assign.mutateAsync({ sessionId: item.session_id, target })
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Couldn't assign the voice. Try again.")
    }
  }

  return (
    <div className="flex flex-col gap-3">
      <RadioGroup value={choice ?? undefined} onValueChange={setChoice} aria-label="Assign to" className="flex flex-col gap-2">
        {speakers.map((speaker) => (
          <div key={speaker.id} className="flex items-center gap-3">
            <RadioGroupItem value={String(speaker.id)} id={`${idPrefix}-${speaker.id}`} />
            <Label htmlFor={`${idPrefix}-${speaker.id}`}>{speaker.display_name}</Label>
          </div>
        ))}
        <div className="flex items-center gap-3">
          <RadioGroupItem value={NEW_MEMBER} id={`${idPrefix}-new`} />
          <Label htmlFor={`${idPrefix}-new`}>New member</Label>
        </div>
      </RadioGroup>
      {choice === NEW_MEMBER ? (
        <div className="flex flex-col gap-1.5">
          <Label htmlFor={`${idPrefix}-name`}>Name</Label>
          <Input id={`${idPrefix}-name`} value={newName} onChange={(event) => setNewName(event.target.value)} />
        </div>
      ) : null}
      {error ? <p className="text-label text-destructive">{error}</p> : null}
      <SubmitButton onSubmit={handleAssign} disabled={target === null} pendingLabel="Assigning…">
        Assign voice
      </SubmitButton>
    </div>
  )
}

function VoiceInboxRow({ item, speakers }: { item: VoiceInboxItem; speakers: Speaker[] }) {
  const [formOpen, setFormOpen] = React.useState(false)
  const time = new Date(item.started_at).toLocaleString()

  return (
    <li className="flex flex-col gap-2 px-4 py-3">
      <div className="flex items-center gap-2">
        <span className="text-label text-muted-foreground">{time}</span>
        {item.blocked ? <Badge variant="outline">Blocked</Badge> : null}
      </div>
      <p className="text-body text-foreground">{item.transcript ?? "No transcript"}</p>
      <div className="flex flex-wrap items-center gap-3">
        <audio controls preload="none" src={voiceInboxAudioUrl(item.session_id)} aria-label={`Play the turn from ${time}`} />
        <Button type="button" variant="outline" size="sm" onClick={() => setFormOpen((wasOpen) => !wasOpen)}>
          Assign
        </Button>
      </div>
      {formOpen ? <AssignForm item={item} speakers={speakers} /> : null}
    </li>
  )
}

export function VoiceInbox({ speakers }: { speakers: Speaker[] }) {
  const inbox = useQuery(voiceInboxQueryOptions)

  return (
    <section className="flex flex-col gap-3">
      <div className="flex flex-col gap-1">
        <h2 className="text-heading font-semibold text-foreground">Recent unrecognized voices</h2>
        <p className="text-body text-muted-foreground">
          Turns from the edge microphone that no member matched, while their recordings are kept.
          Listen first, then assign the voice to a member. A voice match never grants permission.
        </p>
      </div>

      {inbox.status === "pending" ? <SkeletonList rows={2} /> : null}

      {inbox.status === "error" ? (
        <ErrorState
          message={inbox.error instanceof ApiError ? inbox.error.message : "Can't reach the server. Check your connection and try again."}
          onRetry={() => void inbox.refetch()}
        />
      ) : null}

      {inbox.status === "success" ? (
        inbox.data.items.length === 0 ? (
          <EmptyState heading="No unrecognized voices." body="Turns that no member matched appear here." />
        ) : (
          <ul className="panel-list">
            {inbox.data.items.map((item) => (
              <VoiceInboxRow key={item.session_id} item={item} speakers={speakers} />
            ))}
          </ul>
        )
      ) : null}
    </section>
  )
}
