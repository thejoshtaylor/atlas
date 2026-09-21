import { useQuery } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SESSIONS_QUERY_KEY, fetchSessions, type SessionSummary } from "@/lib/sessions"
import { deriveSessionsScreenState, formatSessionDuration, summarizeSessionOutcome } from "./deriveSessionsScreenState"

// WEB-07, D-01, D-02, 08-UI-SPEC.md's Focal Point row: "the card list
// itself, in reverse-chronological order ... there is no primary action on
// this screen." Copies `PluginsRoute.tsx`'s list/fetch/Skeleton/Error
// shape, with two deliberate deltas: the household-audio disclosure (D-03)
// beneath the title, and no "New" button -- a session is never created
// from this screen.

function SessionRow({ session }: { session: SessionSummary }) {
  return (
    <li className="rounded-lg border border-border bg-card p-4">
      <Link to={`/sessions/${session.id}`} className="flex flex-col gap-1 touch-target">
        <span className="truncate text-body text-foreground">{summarizeSessionOutcome(session)}</span>
        <span className="text-label text-muted-foreground">{formatSessionDuration(session.duration_ms)}</span>
      </Link>
    </li>
  )
}

export function SessionsRoute() {
  const query = useQuery({ queryKey: SESSIONS_QUERY_KEY, queryFn: fetchSessions })
  const screen = deriveSessionsScreenState(query)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Sessions</h1>
        {/* D-03, 08-UI-SPEC.md Copywriting Contract: one fixed sentence,
         * plain text, the same disclosure register 07-UI-SPEC.md's Piper
         * licence line already established -- never an icon, never an
         * alert box. */}
        <p className="text-body text-muted-foreground">
          This shows real speech and audio recorded in your home.
        </p>
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
      ) : null}

      {screen.kind === "empty" ? (
        <EmptyState
          heading="No sessions yet."
          body="A session appears here the first time the wake word starts a turn."
        />
      ) : null}

      {screen.kind === "ready" ? (
        <ul className="flex flex-col gap-2">
          {screen.sessions.map((session) => (
            <SessionRow key={session.id} session={session} />
          ))}
        </ul>
      ) : null}
    </div>
  )
}
