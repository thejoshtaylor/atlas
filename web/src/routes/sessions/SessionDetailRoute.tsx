import * as React from "react"
import { useQuery } from "@tanstack/react-query"
import { Link, useParams } from "react-router-dom"
import { ErrorState } from "@/components/state/ErrorState"
import { fetchSession, sessionQueryKey, type TimelineEntry } from "@/lib/sessions"
import { deriveSessionDetailScreenState } from "./deriveSessionDetailScreenState"

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

function TimelineRow({ entry }: { entry: TimelineEntry }) {
  const label = typeof entry.stage === "string" ? entry.stage : typeof entry.type === "string" ? entry.type : entry.kind
  const offset = entry.offset_s === null ? "—" : `${entry.offset_s.toFixed(2)}s`
  return (
    <li className="flex flex-col gap-0.5 rounded-lg border border-border bg-card p-4">
      <span className="text-label text-muted-foreground">{offset}</span>
      <span className="text-body text-foreground">{label}</span>
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
  const everReadyRef = React.useRef(false)
  if (query.status === "success" && query.data) {
    everReadyRef.current = true
  }

  const screen = deriveSessionDetailScreenState({ query, hasEverLoaded: everReadyRef.current })

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
          <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
            <p className="text-body text-foreground">{screen.session.turn_outcome}</p>
            <p className="text-body text-foreground">
              Heard: {screen.session.transcript ?? "—"}
            </p>
            <p className="text-body text-foreground">
              Replied: {screen.session.reply_text ?? "No reply."}
            </p>
          </div>

          <div className="flex flex-col gap-2">
            <h2 className="text-heading font-semibold">Timeline</h2>
            <ul className="flex flex-col gap-2">
              {screen.session.timeline.map((entry, index) => (
                <TimelineRow key={index} entry={entry} />
              ))}
            </ul>
          </div>
        </div>
      ) : null}
    </div>
  )
}
