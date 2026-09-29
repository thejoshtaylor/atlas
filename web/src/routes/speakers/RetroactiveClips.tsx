import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { ApiError } from "@/lib/api"
import { formatSeconds } from "@/lib/format"
import {
  deleteRetroactiveClipMutationOptions,
  retroactiveClipAudioUrl,
  retroactiveClipsQueryOptions,
  type RetroactiveClip,
} from "@/lib/speakerInbox"
import type { Speaker } from "@/lib/speakers"

// 260929-j08: the clips of one member that came from recorded turns, each
// with play and Remove. Remove has no confirm step. It is the undo path,
// and the source turn returns to the inbox while its recording is kept.

function ClipRow({ speaker, clip }: { speaker: Speaker; clip: RetroactiveClip }) {
  const remove = useMutation(deleteRetroactiveClipMutationOptions)
  const [error, setError] = React.useState<string | null>(null)
  const time = clip.created_at === null ? "Unknown time" : new Date(clip.created_at).toLocaleString()

  const handleRemove = async () => {
    setError(null)
    try {
      await remove.mutateAsync({ speakerId: speaker.id, phraseIndex: clip.phrase_index })
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Couldn't remove the clip. Try again.")
    }
  }

  return (
    <li className="flex flex-col gap-1">
      <div className="flex flex-wrap items-center gap-3">
        <span className="text-label text-muted-foreground">{time}</span>
        {clip.speech_ms !== null ? (
          <span className="text-label text-muted-foreground">{formatSeconds(clip.speech_ms / 1000)}</span>
        ) : null}
        <audio
          controls
          preload="none"
          src={retroactiveClipAudioUrl(speaker.id, clip.phrase_index)}
          aria-label={`Play the clip from ${time}`}
        />
        <Button type="button" variant="outline" size="sm" disabled={remove.isPending} onClick={() => void handleRemove()}>
          Remove
        </Button>
      </div>
      {error ? <p className="text-label text-destructive">{error}</p> : null}
    </li>
  )
}

export function RetroactiveClips({ speaker }: { speaker: Speaker }) {
  const clips = useQuery(retroactiveClipsQueryOptions(speaker.id))
  const list = clips.data?.clips ?? []
  if (list.length === 0) return null

  return (
    <div className="flex flex-col gap-2">
      <p className="text-label font-medium text-foreground">Clips from recordings</p>
      <ul className="flex flex-col gap-2">
        {list.map((clip) => (
          <ClipRow key={clip.phrase_index} speaker={speaker} clip={clip} />
        ))}
      </ul>
    </div>
  )
}
