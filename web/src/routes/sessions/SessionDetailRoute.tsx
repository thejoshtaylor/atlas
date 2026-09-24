import * as React from "react"
import { useQuery } from "@tanstack/react-query"
import { Link, useParams } from "react-router-dom"
import { ErrorState } from "@/components/state/ErrorState"
import { formatSeconds } from "@/lib/format"
import { fetchSession, sessionAudioUrl, sessionQueryKey, type TimelineEntry } from "@/lib/sessions"
import { activeTimelineIndexAt, deriveSessionDetailScreenState } from "./deriveSessionDetailScreenState"

// DBG-03, D-02, D-03, D-04, 08-UI-SPEC.md's Focal Point row: "the
// transcript/reply/outcome block at the top ... the timeline immediately
// below." Follows `CalibrationRoute.tsx`'s multi-branch rendering shape,
// with two terminal branches that analog does not have -- "removed" and
// "not_found" -- which is exactly why `ErrorState`'s `onRetry` became
// optional (08-01-PLAN.md): retrying a fetch for a session that is gone
// by design, or was never real, would offer an action that cannot work.

function formatStartedAt(startedAt: string): string {
  return new Date(startedAt).toLocaleString()
}

// D-09: the <audio> element is the clock; this row only ever reads
// `isActive` (derived from that element's own `currentTime` via
// `activeTimelineIndexAt`) and calls `onSelect` to seek it -- it holds no
// clock of its own. `aria-current` names the row the same way any other
// "this one is the current one in a sequence" UI does; the neutral
// `bg-accent` token is 08-UI-SPEC.md's Color section's explicit choice,
// not the reserved 10% primary accent.
function TimelineRow({ entry, isActive, onSelect }: { entry: TimelineEntry; isActive: boolean; onSelect: () => void }) {
  const label = typeof entry.stage === "string" ? entry.stage : typeof entry.type === "string" ? entry.type : entry.kind
  const offset = entry.offset_s === null ? "—" : `${entry.offset_s.toFixed(2)}s`
  return (
    <li>
      <button
        type="button"
        onClick={onSelect}
        aria-current={isActive ? "true" : undefined}
        className={`touch-target flex w-full flex-col gap-0.5 rounded-lg border border-border p-4 text-left ${
          isActive ? "bg-accent" : "bg-card"
        }`}
      >
        <span className="readout text-label text-muted-foreground">{offset}</span>
        <span className="text-body text-foreground">{label}</span>
      </button>
    </li>
  )
}

export function SessionDetailRoute() {
  const { id } = useParams<{ id: string }>()
  const sessionId = id ?? ""
  const query = useQuery({
    queryKey: sessionQueryKey(sessionId),
    queryFn: () => fetchSession(sessionId),
    enabled: sessionId !== "",
  })

  // The one piece of state `deriveSessionDetailScreenState` itself cannot
  // hold (it is a pure function, never storing anything between calls,
  // matching `timeline.py`'s own "derived, and never the source of
  // truth" discipline) -- whether this session id has ever resolved to
  // real data in this component's lifetime, per D-04's client-side case.
  // Set from an effect, never read or written during render itself (React
  // refs/state must only be mutated outside render) -- a later render
  // that reaches the `error` branch still sees the flag `true`, since the
  // commit from the earlier `success` render always precedes it.
  const [hasEverLoaded, setHasEverLoaded] = React.useState(false)
  React.useEffect(() => {
    if (query.status === "success" && query.data) {
      setHasEverLoaded(true)
    }
  }, [query.status, query.data])

  const screen = deriveSessionDetailScreenState({ query, hasEverLoaded })

  // The one piece of state the audio element's own clock drives (D-09):
  // `null` until the first `timeupdate` fires, so no timeline row is
  // marked current before playback (or a seek) has reported a real time.
  // `audioFailed` is set from the element's own `error` event and is the
  // one-region degradation D-11/the Copywriting Contract require -- the
  // rest of the screen (transcript, reply, timeline) is untouched by it.
  const [currentTime, setCurrentTime] = React.useState<number | null>(null)
  const [audioFailed, setAudioFailed] = React.useState(false)
  const audioRef = React.useRef<HTMLAudioElement>(null)

  React.useEffect(() => {
    setCurrentTime(null)
    setAudioFailed(false)
  }, [sessionId])

  const activeIndex =
    screen.kind === "ready" && currentTime !== null ? activeTimelineIndexAt(screen.session.timeline, currentTime) : null

  const handleTimeUpdate = (event: React.SyntheticEvent<HTMLAudioElement>) => {
    setCurrentTime(event.currentTarget.currentTime)
  }
  const handleAudioError = () => {
    setAudioFailed(true)
  }
  const handleRowSelect = (offset: number | null) => {
    if (offset === null || audioRef.current === null) return
    audioRef.current.currentTime = offset
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">
          {screen.kind === "ready" ? `Session — ${formatStartedAt(screen.session.started_at)}` : "Session"}
        </h1>
        <p className="text-body text-muted-foreground">
          This shows real speech and audio recorded in your home.
        </p>
      </div>

      {screen.kind === "loading" ? (
        <p className="text-body text-muted-foreground">Loading session…</p>
      ) : null}

      {screen.kind === "removed" ? (
        <ErrorState
          message="This session was removed."
          detail="Recordings expire automatically on this deployment's retention schedule. This one is gone, and that's expected."
        />
      ) : null}

      {screen.kind === "failed" ? (
        <div className="flex flex-col gap-3">
          {/* The server's own reason, verbatim -- `routes/sessions.py`
              answers 409 by name for a recording that was never finished
              being written and for one this deployment cannot play back.
              Neither is a missing session and neither may read as one. */}
          <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
          <Link to="/sessions" className="text-body text-primary underline touch-target">
            Back to Sessions
          </Link>
        </div>
      ) : null}

      {screen.kind === "not_found" ? (
        <div className="flex flex-col gap-3">
          <ErrorState
            message="Couldn't find this session."
            detail="It may have expired, or the link may be wrong."
          />
          <Link to="/sessions" className="text-body text-primary underline touch-target">
            Back to Sessions
          </Link>
        </div>
      ) : null}

      {screen.kind === "ready" ? (
        <div className="flex flex-col gap-6">
          {/* Data this screen genuinely holds, shown as what it is: a
              failed refresh is not evidence the recording went away, and
              discarding it to say so was the defect. */}
          {screen.stale ? (
            <p className="text-label text-muted-foreground">
              Couldn't refresh this session just now. Showing what was loaded earlier.
            </p>
          ) : null}
          <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
            <p className="text-body text-foreground">{screen.session.turn_outcome}</p>
            <p className="text-body text-foreground">
              Heard: {screen.session.transcript ?? "—"}
            </p>
            <p className="text-body text-foreground">
              Replied: {screen.session.reply_text ?? "No reply."}
            </p>
          </div>

          {/* `has_audio` is the server's own answer to "was anything
              recorded for this turn" (`_audio_not_recorded_error` is the
              named refusal behind it). Rendering the player anyway meant
              a 404, an `onError`, and "Couldn't load the recording." for a
              turn where nothing was ever recorded -- a load failure
              reported where there was nothing to load. The load-failure
              copy is reserved for a real failure now. */}
          {!screen.session.has_audio ? (
            <p className="text-body text-muted-foreground">No audio was recorded for this turn.</p>
          ) : audioFailed ? (
            <p className="text-body text-muted-foreground">
              Couldn't load the recording. The rest of this session is still shown below.
            </p>
          ) : (
            <audio
              ref={audioRef}
              controls
              className="w-full"
              src={sessionAudioUrl(screen.session.id)}
              onTimeUpdate={handleTimeUpdate}
              onError={handleAudioError}
            />
          )}

          <div className="flex flex-col gap-2">
            <h2 className="text-heading font-semibold">Timeline</h2>
            {/* Only for a camera-path recording (preroll_s > 0): the first
                seconds are room audio captured before the wake word, so no
                timeline row highlights during them. Without this sentence
                that silence reads as the highlight lagging -- exactly the
                defect plan 08-11 fixed (DBG-03, D-03). Built from the
                number alone; no text carried by any recorded event may
                reach it. */}
            {screen.session.preroll_s > 0 ? (
              <p className="text-label text-muted-foreground">
                The recording starts {formatSeconds(screen.session.preroll_s)} before the wake word. No row is
                highlighted until the assistant starts listening.
              </p>
            ) : null}
            <ul className="flex flex-col gap-2">
              {screen.session.timeline.map((entry, index) => (
                <TimelineRow
                  key={index}
                  entry={entry}
                  isActive={activeIndex === index}
                  onSelect={() => handleRowSelect(entry.offset_s)}
                />
              ))}
            </ul>
          </div>
        </div>
      ) : null}
    </div>
  )
}
