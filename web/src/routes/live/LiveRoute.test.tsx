// Mount tests for `/live` (WEB-06, D-05, D-06, D-07, D-08). No
// `mock.module`/dynamic-import dance is needed here, unlike the Sessions
// screens' own mount tests -- `LiveRoute` takes its connection as an
// injectable prop (the same shape `DevMicRoute` gives its transports), so
// a fake connection is passed directly rather than replacing a fetch
// module.
import { afterEach, expect, test } from "bun:test"
import { act } from "react"
import { cleanup, render, screen } from "@testing-library/react"
import { MemoryRouter } from "react-router-dom"
import { LiveRoute } from "./LiveRoute"
import type { ObserverConnection, ObserverHandlers, ObserverMessage } from "./observerSocket"

afterEach(() => {
  cleanup()
})

/** A fake `connect` that hands the test direct control over what the
 * component receives -- `emit`/`setState` call straight into the handlers
 * `LiveRoute` registered, synchronously, so a test can drive the whole
 * message sequence without a real socket. `close` records whether the
 * component actually cleaned up on unmount. */
function fakeConnection() {
  let handlers: ObserverHandlers | null = null
  let closed = false
  const connect = (givenHandlers: ObserverHandlers): ObserverConnection => {
    handlers = givenHandlers
    return { close: () => (closed = true) }
  }
  return {
    connect,
    // `onMessage`/`onStateChange` fire outside any React-tracked event (the
    // real `WebSocket.onmessage`/`onclose` handlers do too) -- `act(...)`
    // is what makes the resulting state update flush before the next
    // assertion, the same requirement React's own testing docs state.
    emit: (message: ObserverMessage) => act(() => handlers?.onMessage(message)),
    setState: (state: "connecting" | "connected" | "disconnected") =>
      act(() => handlers?.onStateChange(state)),
    isClosed: () => closed,
  }
}

function renderLive(connect: (handlers: ObserverHandlers) => ObserverConnection) {
  render(
    <MemoryRouter>
      <LiveRoute connect={connect} />
    </MemoryRouter>,
  )
}

test("with no message ever received, the idle state shows no wake phrase and no media request is made", () => {
  const fake = fakeConnection()
  const originalGetUserMedia = navigator.mediaDevices?.getUserMedia
  let getUserMediaCalled = false
  if (navigator.mediaDevices) {
    // @ts-expect-error -- test double, narrower than the real signature
    navigator.mediaDevices.getUserMedia = () => {
      getUserMediaCalled = true
      return Promise.reject(new Error("must never be called"))
    }
  }

  renderLive(fake.connect)

  expect(screen.getByText("This shows real speech and audio recorded in your home.")).toBeTruthy()
  expect(getUserMediaCalled).toBe(false)

  if (navigator.mediaDevices && originalGetUserMedia) {
    navigator.mediaDevices.getUserMedia = originalGetUserMedia
  }
})

test("the idle heading names the server-reported wake phrase and the caption names the sources", () => {
  const fake = fakeConnection()
  renderLive(fake.connect)

  fake.emit({ type: "observer.opened", wake_phrase: "hey spire", sources: ["camera", "browser_mic"] })

  expect(screen.getByText('Listening for "hey spire".')).toBeTruthy()
  expect(
    screen.getByText("Say it near any source to start a turn. This page updates as it happens."),
  ).toBeTruthy()
  expect(screen.getByText(/Listening for .* — camera, browser mic\./)).toBeTruthy()
})

test("the connection badge reads connected while open and disconnected once closed", () => {
  const fake = fakeConnection()
  renderLive(fake.connect)

  fake.setState("connected")
  expect(screen.getByText("Connected")).toBeTruthy()

  fake.setState("disconnected")
  expect(screen.getByText("Disconnected — reconnecting…")).toBeTruthy()
})

test("a socket that never opened at all reads the same as one that dropped", () => {
  const fake = fakeConnection()
  renderLive(fake.connect)
  // No setState call at all -- the component's own initial state.
  expect(screen.getByText("Disconnected — reconnecting…")).toBeTruthy()
})

test("the whole turn sequence: start, partials, reply, timing, and a second turn on top", () => {
  const fake = fakeConnection()
  renderLive(fake.connect)

  fake.emit({ type: "observer.opened", wake_phrase: "hey spire", sources: ["camera"] })
  fake.emit({ type: "turn.started", source: "camera", turn_id: "t1", session_id: "20260101T000000000000Z-t1" })

  expect(screen.getByText("Camera")).toBeTruthy()
  expect(screen.getByText("Listening…")).toBeTruthy()

  fake.emit({ type: "transcript.partial", text: "turn the", source: "camera" })
  expect(screen.getByText("turn the")).toBeTruthy()

  fake.emit({ type: "transcript.partial", text: "turn the lights on", source: "camera" })
  expect(screen.getByText("turn the lights on")).toBeTruthy()
  expect(screen.queryByText("turn the")).toBeNull()

  fake.emit({ type: "reply.text", text: "done", source: "camera" })
  expect(screen.getByText("done")).toBeTruthy()
  expect(screen.getByText("View full session →").getAttribute("href")).toBe(
    "/sessions/20260101T000000000000Z-t1",
  )

  fake.emit({ type: "turn.timing", turn_outcome: "completed", source: "camera" })
  expect(screen.queryByText("Listening…")).toBeNull()
  expect(screen.getByText("completed")).toBeTruthy()

  // A second turn's card appears above the first.
  fake.emit({ type: "turn.started", source: "camera", turn_id: "t2", session_id: "20260101T000100000000Z-t2" })
  fake.emit({ type: "transcript.partial", text: "second turn", source: "camera" })

  const cards = screen.getAllByText(/Listening…|second turn/)
  expect(cards.length).toBeGreaterThan(0)
  expect(screen.getByText("second turn")).toBeTruthy()
  // The first turn's own heard text is still in the feed, further down.
  expect(screen.getByText("turn the lights on")).toBeTruthy()
})

test("empty_transcript renders the fixed outcome copy, not the raw outcome string", () => {
  const fake = fakeConnection()
  renderLive(fake.connect)

  fake.emit({ type: "observer.opened", wake_phrase: "hey spire", sources: ["camera"] })
  fake.emit({ type: "turn.started", source: "camera", turn_id: "t1", session_id: "20260101T000000000000Z-t1" })
  fake.emit({ type: "turn.timing", turn_outcome: "empty_transcript", source: "camera" })

  expect(screen.getByText("Understood no speech.")).toBeTruthy()
})

test("unmounting closes the underlying connection", () => {
  const fake = fakeConnection()
  const { unmount } = render(
    <MemoryRouter>
      <LiveRoute connect={fake.connect} />
    </MemoryRouter>,
  )
  expect(fake.isClosed()).toBe(false)
  unmount()
  expect(fake.isClosed()).toBe(true)
})
