import { expect, test } from "bun:test"
import {
  MAX_FEED_CARDS,
  deriveLiveScreenState,
  describeTurnOutcome,
  formatElapsed,
  humanizeSourceName,
  type RawObserverMessage,
} from "./deriveLiveScreenState"

function opened(overrides: Partial<RawObserverMessage> = {}): RawObserverMessage {
  return {
    type: "observer.opened",
    wake_phrase: "hey spire",
    sources: ["camera", "browser_mic"],
    ...overrides,
  }
}

function started(source: string, turnId: string, sessionId = `20260101T000000000000Z-${turnId}`): RawObserverMessage {
  return { type: "turn.started", source, turn_id: turnId, session_id: sessionId }
}

test("with no messages at all, the screen is idle with no wake phrase known yet", () => {
  const screen = deriveLiveScreenState([])
  expect(screen).toEqual({ kind: "idle", wakePhrase: "", sources: [] })
})

test("the opening message alone carries the wake phrase and sources into the idle state", () => {
  const screen = deriveLiveScreenState([opened()])
  expect(screen).toEqual({ kind: "idle", wakePhrase: "hey spire", sources: ["camera", "browser_mic"] })
})

test("a turn.started message opens a card and switches the screen to feed", () => {
  const screen = deriveLiveScreenState([opened(), started("camera", "turn-1")])
  expect(screen.kind).toBe("feed")
  if (screen.kind !== "feed") throw new Error("unreachable")
  expect(screen.cards).toHaveLength(1)
  expect(screen.cards[0]).toEqual({
    turnId: "turn-1",
    sessionId: "20260101T000000000000Z-turn-1",
    source: "camera",
    transcript: "",
    reply: null,
    outcome: null,
    closed: false,
  })
})

test("partial transcripts update the open card's text in place", () => {
  const screen = deriveLiveScreenState([
    opened(),
    started("camera", "turn-1"),
    { type: "transcript.partial", text: "turn the", source: "camera" },
    { type: "transcript.partial", text: "turn the lights on", source: "camera" },
  ])
  if (screen.kind !== "feed") throw new Error("unreachable");
  expect(screen.cards[0]!.transcript).toBe("turn the lights on")
})

test("a reply message adds the reply without closing the card", () => {
  const screen = deriveLiveScreenState([
    opened(),
    started("camera", "turn-1"),
    { type: "reply.text", text: "done", source: "camera" },
  ])
  if (screen.kind !== "feed") throw new Error("unreachable");
  expect(screen.cards[0]!.reply).toBe("done")
  expect(screen.cards[0]!.closed).toBe(false)
})

test("a timing message closes the card and records the outcome", () => {
  const screen = deriveLiveScreenState([
    opened(),
    started("camera", "turn-1"),
    { type: "reply.text", text: "done", source: "camera" },
    { type: "turn.timing", turn_outcome: "completed", source: "camera" },
  ])
  if (screen.kind !== "feed") throw new Error("unreachable");
  expect(screen.cards[0]!.closed).toBe(true)
  expect(screen.cards[0]!.outcome).toBe("completed")
})

test("a second turn's card appears above the first", () => {
  const screen = deriveLiveScreenState([
    opened(),
    started("camera", "turn-1"),
    { type: "turn.timing", turn_outcome: "completed", source: "camera" },
    started("browser_mic", "turn-2"),
  ])
  if (screen.kind !== "feed") throw new Error("unreachable");
  expect(screen.cards.map((card) => card.turnId)).toEqual(["turn-2", "turn-1"])
})

test("the feed keeps at most the most recent twenty cards, dropping the oldest", () => {
  const messages: RawObserverMessage[] = [opened()]
  for (let i = 0; i < MAX_FEED_CARDS + 5; i += 1) {
    messages.push(started("camera", `turn-${i}`))
  }
  const screen = deriveLiveScreenState(messages)
  if (screen.kind !== "feed") throw new Error("unreachable");
  expect(screen.cards).toHaveLength(MAX_FEED_CARDS)
  expect(screen.cards[0]!.turnId).toBe(`turn-${MAX_FEED_CARDS + 4}`)
  expect(screen.cards[screen.cards.length - 1]!.turnId).toBe("turn-5")
})

test("two sources can each have their own open card at once", () => {
  const screen = deriveLiveScreenState([
    opened(),
    started("camera", "turn-1"),
    started("browser_mic", "turn-2"),
    { type: "transcript.partial", text: "camera heard this", source: "camera" },
    { type: "transcript.partial", text: "mic heard this", source: "browser_mic" },
  ])
  if (screen.kind !== "feed") throw new Error("unreachable");
  const bySource = Object.fromEntries(screen.cards.map((card) => [card.source, card.transcript]))
  expect(bySource).toEqual({ camera: "camera heard this", browser_mic: "mic heard this" })
})

test("once a turn has ever been seen, the screen stays in feed mode even with every card closed", () => {
  const screen = deriveLiveScreenState([
    opened(),
    started("camera", "turn-1"),
    { type: "turn.timing", turn_outcome: "completed", source: "camera" },
  ])
  expect(screen.kind).toBe("feed")
})

test("an event for a source with no open card is ignored, not crashed on", () => {
  const screen = deriveLiveScreenState([
    opened(),
    { type: "transcript.partial", text: "orphaned", source: "camera" },
  ])
  expect(screen).toEqual({ kind: "idle", wakePhrase: "hey spire", sources: ["camera", "browser_mic"] })
})

test("formatElapsed renders seconds alone under a minute, minutes and seconds at or above it", () => {
  expect(formatElapsed(0)).toBe("0s")
  expect(formatElapsed(45)).toBe("45s")
  expect(formatElapsed(59)).toBe("59s")
  expect(formatElapsed(60)).toBe("1m 0s")
  expect(formatElapsed(192)).toBe("3m 12s")
})

test("humanizeSourceName replaces underscores with spaces and nothing else", () => {
  expect(humanizeSourceName("camera")).toBe("camera")
  expect(humanizeSourceName("browser_mic")).toBe("browser mic")
})

test("describeTurnOutcome maps the two named outcomes and passes everything else through verbatim", () => {
  expect(describeTurnOutcome("empty_transcript")).toBe("Understood no speech.")
  expect(describeTurnOutcome("empty_reply")).toBe("No reply was given.")
  expect(describeTurnOutcome("completed")).toBe("completed")
  expect(describeTurnOutcome("round_cap")).toBe("round_cap")
})
