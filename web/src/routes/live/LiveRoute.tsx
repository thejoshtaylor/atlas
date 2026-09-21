import * as React from "react"
import { Link } from "react-router-dom"
import { Badge } from "@/components/ui/badge"
import {
  connect as connectObserverSocket,
  type ObserverConnection,
  type ObserverConnectionState,
  type ObserverHandlers,
} from "./observerSocket"
import {
  appendBoundedObserverMessage,
  deriveLiveScreenState,
  describeTurnOutcome,
  formatElapsed,
  humanizeSourceName,
  type RawObserverMessage,
  type TurnCard,
} from "./deriveLiveScreenState"

const HOUSEHOLD_AUDIO_DISCLOSURE = "This shows real speech and audio recorded in your home."

function capitalize(text: string): string {
  return text.length === 0 ? text : text.charAt(0).toUpperCase() + text.slice(1)
}

function sourceBadgeLabel(source: string): string {
  return capitalize(humanizeSourceName(source))
}

interface ConnectionBadgeProps {
  state: ObserverConnectionState
}

function ConnectionBadge({ state }: ConnectionBadgeProps) {
  // 08-UI-SPEC.md Color section: the single amber-family use across all
  // four of this phase's screens. A socket that dropped is a real,
  // worth-noticing fact; every other badge on this screen is routine by
  // design (`variant="outline"`/`"secondary"`, never `"denied"`).
  const connected = state === "connected"
  return (
    <Badge variant={connected ? "secondary" : "denied"}>
      {connected ? "Connected" : "Disconnected — reconnecting…"}
    </Badge>
  )
}

function TurnCardRow({ card }: { card: TurnCard }) {
  return (
    <li className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <Badge variant="outline">{sourceBadgeLabel(card.source)}</Badge>
      {card.transcript ? <p className="text-body text-foreground">{card.transcript}</p> : null}
      {card.reply ? <p className="text-body text-foreground">{card.reply}</p> : null}
      {card.closed ? (
        <p className="text-label text-muted-foreground">{describeTurnOutcome(card.outcome ?? "")}</p>
      ) : (
        <p className="text-label text-muted-foreground">Listening…</p>
      )}
      {card.reply !== null && card.sessionId ? (
        <Link to={`/sessions/${card.sessionId}`} className="touch-target text-label text-primary">
          View full session →
        </Link>
      ) : null}
    </li>
  )
}

export interface LiveRouteProps {
  /** Injectable, the same shape `DevMicRoute` gives its transports, so a
   * mount test can drive real messages through a fake connection without
   * a real socket. Defaults to the real `observerSocket.connect`. */
  connect?: (handlers: ObserverHandlers) => ObserverConnection
}

/**
 * WEB-06: a rolling, read-only feed of the turns every configured source
 * hears, over `/ws/sessions/live` -- watching never opens a microphone of
 * the operator's own (D-05). The idle state (D-07) is the screen's most
 * common state by far: it names the wake phrase the server actually
 * reports, never a hardcoded one, and a caption that ticks once a second
 * proves the connection is alive without a spinner.
 */
export function LiveRoute({ connect = connectObserverSocket }: LiveRouteProps) {
  const [connectionState, setConnectionState] = React.useState<ObserverConnectionState>("connecting")
  const [messages, setMessages] = React.useState<RawObserverMessage[]>([])
  const [openedAt] = React.useState(() => Date.now())
  // `Date.now()` itself is only ever read inside an effect, never during
  // render (oxlint's own `react(purity)` rule) -- `now` is state, ticked
  // once a second, and `elapsedSeconds` below is a pure derivation of two
  // already-stored numbers.
  const [now, setNow] = React.useState(() => Date.now())

  React.useEffect(() => {
    const connection = connect({
      // Bounded at the source (WR-11): the buffer holds household speech,
      // and an unbounded one on a page meant to stay open is an
      // ever-growing, ever-slower accumulation of it in memory.
      onMessage: (message) => setMessages((history) => appendBoundedObserverMessage(history, message)),
      onStateChange: setConnectionState,
    })
    return () => connection.close()
  }, [connect])

  React.useEffect(() => {
    const interval = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(interval)
  }, [])

  // Memoised so the once-a-second `now` tick above stops re-folding the
  // whole buffer -- the tick changes `now` and nothing the fold reads.
  const screen = React.useMemo(() => deriveLiveScreenState(messages), [messages])
  const elapsedSeconds = (now - openedAt) / 1000

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Live</h1>
        <p className="text-body text-muted-foreground">{HOUSEHOLD_AUDIO_DISCLOSURE}</p>
      </div>

      <ConnectionBadge state={connectionState} />

      {screen.kind === "idle" ? (
        <div className="flex flex-col gap-2">
          <h2 className="text-heading font-semibold">Listening for &quot;{screen.wakePhrase}&quot;.</h2>
          <p className="text-body text-muted-foreground">
            Say it near any source to start a turn. This page updates as it happens.
          </p>
          <p className="text-label text-muted-foreground">
            Listening for {formatElapsed(elapsedSeconds)}
            {screen.sources.length > 0
              ? ` — ${screen.sources.map(humanizeSourceName).join(", ")}.`
              : "."}
          </p>
        </div>
      ) : null}

      {screen.kind === "feed" ? (
        <ul className="flex flex-col gap-2">
          {screen.cards.map((card) => (
            <TurnCardRow key={card.turnId} card={card} />
          ))}
        </ul>
      ) : null}
    </div>
  )
}
