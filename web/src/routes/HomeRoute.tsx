import { useQuery } from "@tanstack/react-query"
import { ArrowRight, AudioLines, Mic } from "lucide-react"
import { Link, useNavigate } from "react-router-dom"
import { AtlasGlobe } from "@/components/brand/AtlasGlobe"
import { Button } from "@/components/ui/button"
import { isListening, startListening, useListenStore } from "@/lib/listener"
import { useSession } from "@/hooks/useSession"
import { formatMs } from "@/lib/format"
import { SESSIONS_QUERY_KEY, fetchSessions, type SessionSummary } from "@/lib/sessions"
import { summarizeSessionOutcome } from "@/routes/sessions/deriveSessionsScreenState"

const RECENT_TURNS = 5

function RecentTurn({ session }: { session: SessionSummary }) {
  return (
    <li>
      <Link
        to={`/sessions/${session.id}`}
        className="touch-target grid grid-cols-[4.5rem_1fr_auto] items-center gap-4 border-b border-border py-3 text-label transition-colors hover:bg-accent/60"
      >
        <span className="readout text-muted-foreground">
          {new Date(session.started_at).toLocaleTimeString([], { hour: "numeric", minute: "2-digit" })}
        </span>
        <span className="truncate text-foreground">{summarizeSessionOutcome(session)}</span>
        <span className="readout text-muted-foreground">{formatMs(session.duration_ms)}</span>
      </Link>
    </li>
  )
}

/**
 * The authenticated shell's index: the ATLAS globe as the assistant's
 * presence, and -- for a role that may read sessions -- the last few
 * turns it heard, each linking to its full session.
 */
export function HomeRoute() {
  const session = useSession()
  const role = session.data?.role ?? "viewer"
  const canObserve = role !== "viewer"
  const recent = useQuery({ queryKey: SESSIONS_QUERY_KEY, queryFn: fetchSessions, enabled: canObserve })
  const firstName = (session.data?.display_name || "").split(" ")[0]
  const turns = recent.data?.slice(0, RECENT_TURNS) ?? []
  const listening = isListening(useListenStore((state) => state.status))
  const navigate = useNavigate()

  return (
    <div className="flex flex-col gap-12">
      <section className="grid items-center gap-8 md:grid-cols-[auto_1fr] md:gap-12">
        <AtlasGlobe spinning className="mx-auto size-48 text-foreground/75 sm:size-60 md:mx-0" />
        <div className="flex flex-col gap-3">
          <h1 className="text-display font-semibold">{firstName ? `Hello, ${firstName}.` : "Hello."}</h1>
          <p className="max-w-prose text-body text-muted-foreground">
            ATLAS listens on every configured source. Say the wake phrase near one to start a turn.
          </p>
          {canObserve ? (
            <div className="mt-3 flex flex-col gap-2 sm:flex-row sm:items-center sm:gap-6">
              {/* Started from the click itself: browsers only open a
                  microphone and an audio context inside a user gesture. */}
              <Button
                type="button"
                size="lg"
                variant={listening ? "outline" : "default"}
                className="h-11 w-full gap-2 sm:w-auto sm:px-6"
                onClick={() => {
                  if (!listening) void startListening()
                  navigate("/listen")
                }}
              >
                {listening ? <AudioLines className="size-4" aria-hidden /> : <Mic className="size-4" aria-hidden />}
                {listening ? "Open the listener" : "Listen on this device"}
              </Button>
              <Link
                to="/live"
                className="touch-target inline-flex w-fit items-center gap-2 text-label font-semibold text-primary underline-offset-4 hover:underline"
              >
                Watch it live <ArrowRight className="size-4" aria-hidden />
              </Link>
            </div>
          ) : null}
        </div>
      </section>

      {canObserve && turns.length > 0 ? (
        <section className="flex flex-col gap-3">
          <div className="flex items-baseline justify-between">
            <h2 className="text-heading font-semibold">Recent turns</h2>
            <Link to="/sessions" className="touch-target inline-flex items-center text-label text-primary underline-offset-4 hover:underline">
              All sessions
            </Link>
          </div>
          <ul className="border-t border-border">
            {turns.map((turn) => (
              <RecentTurn key={turn.id} session={turn} />
            ))}
          </ul>
        </section>
      ) : null}
    </div>
  )
}
