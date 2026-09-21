// Imports nothing from React (the same discipline every `derive*ScreenState`
// module in this project follows, `derivePluginsScreenState.ts`'s own
// module comment): what belongs here is only ever a fact about data, never
// a fact about a render. Folds the observer socket's own raw message list
// into the idle-or-feed shape `LiveRoute.tsx` renders (D-06, D-07, D-08).

/** The feed keeps at most this many turn cards, dropping the oldest --
 * Claude's own discretion within D-06, named here so the number is
 * changeable without reading the fold below. */
export const MAX_FEED_CARDS = 20

export interface TurnCard {
  turnId: string
  sessionId: string | null
  source: string
  transcript: string
  reply: string | null
  /** Set only once a `turn.timing` message closes this card -- `null`
   * while the turn is still in progress. */
  outcome: string | null
  closed: boolean
}

export type LiveScreenState =
  | { kind: "idle"; wakePhrase: string; sources: string[] }
  | { kind: "feed"; wakePhrase: string; sources: string[]; cards: TurnCard[] }

/** A raw message off the observer socket -- loosely typed on purpose
 * (`{ type: string, [key: string]: unknown }`, matching
 * `observerSocket.ts`'s own `ObserverMessage`) rather than a closed union:
 * this module tolerates an unrecognised `type` by ignoring it, the same
 * forward-compatible posture `deriveWakeTuningScreenState.ts`-style folds
 * elsewhere in this project take with a server-reported enum. */
export interface RawObserverMessage {
  type: string
  [key: string]: unknown
}

function asString(value: unknown, fallback = ""): string {
  return typeof value === "string" ? value : fallback
}

function asStringArray(value: unknown): string[] {
  return Array.isArray(value) ? value.filter((entry): entry is string => typeof entry === "string") : []
}

/**
 * Folds every message received so far into the current screen shape.
 * `kind: "idle"` means exactly "no turn has ever been seen on this
 * connection" (D-07) -- once any `turn.started` has arrived, the screen
 * stays in `"feed"` mode for the rest of the connection's life, even once
 * every card in it has closed (D-06's rolling feed, not a "currently
 * live" indicator).
 */
export function deriveLiveScreenState(messages: readonly RawObserverMessage[]): LiveScreenState {
  let wakePhrase = ""
  let sources: string[] = []
  const cards: TurnCard[] = []

  for (const message of messages) {
    switch (message.type) {
      case "observer.opened": {
        wakePhrase = asString(message.wake_phrase)
        sources = asStringArray(message.sources)
        break
      }
      case "turn.started": {
        cards.unshift({
          turnId: asString(message.turn_id),
          sessionId: typeof message.session_id === "string" ? message.session_id : null,
          source: asString(message.source),
          transcript: "",
          reply: null,
          outcome: null,
          closed: false,
        })
        if (cards.length > MAX_FEED_CARDS) {
          cards.length = MAX_FEED_CARDS
        }
        break
      }
      case "transcript.partial": {
        // Exactly one turn per source is ever open at a time (D-08's own
        // "every turn is labelled with the source it came from" holds
        // because `SourceRunner` and `/ws/turn` each run one turn to
        // completion before starting another) -- so the open, unclosed
        // card for this event's source is unambiguous.
        const card = cards.find((candidate) => candidate.source === message.source && !candidate.closed)
        if (card) card.transcript = asString(message.text)
        break
      }
      case "reply.text": {
        const card = cards.find((candidate) => candidate.source === message.source && !candidate.closed)
        if (card) card.reply = asString(message.text)
        break
      }
      case "turn.timing": {
        const card = cards.find((candidate) => candidate.source === message.source && !candidate.closed)
        if (card) {
          card.outcome = asString(message.turn_outcome)
          card.closed = true
        }
        break
      }
      default:
        break
    }
  }

  if (cards.length === 0) {
    return { kind: "idle", wakePhrase, sources }
  }
  return { kind: "feed", wakePhrase, sources, cards }
}

/** "Xm Ys" for a minute or more, "Ys" under one -- the idle caption's own
 * ticking elapsed-time format (08-UI-SPEC.md: "Listening for 3m 12s"). */
export function formatElapsed(totalSeconds: number): string {
  const seconds = Math.max(0, Math.floor(totalSeconds))
  const minutes = Math.floor(seconds / 60)
  const remainingSeconds = seconds % 60
  return minutes > 0 ? `${minutes}m ${remainingSeconds}s` : `${remainingSeconds}s`
}

/** "camera" / "browser_mic" -> "camera" / "browser mic" -- the lowercase
 * form the idle caption's source list uses. The badge label capitalizes
 * this same result (`LiveRoute.tsx`), rather than maintaining two maps. */
export function humanizeSourceName(source: string): string {
  return source.replace(/_/g, " ")
}

const OUTCOME_COPY: Record<string, string> = {
  empty_transcript: "Understood no speech.",
  empty_reply: "No reply was given.",
}

/** The closing caption for a completed turn card: fixed copy for the two
 * named outcomes, the raw server string verbatim otherwise -- the same
 * never-fabricate-a-friendlier-label rule `deriveSessionsScreenState.ts`'s
 * `summarizeSessionOutcome` already applies to the Sessions list. */
export function describeTurnOutcome(outcome: string): string {
  return OUTCOME_COPY[outcome] ?? outcome
}
