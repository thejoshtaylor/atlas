// `observerSocket.ts`'s frame handling, driven against a stubbed
// `WebSocket` -- the module builds its own socket from `window.location`,
// so the constructor is what a test has to stand in for.
import { afterEach, expect, test } from "bun:test"
import { connect, type ObserverConnectionState, type ObserverMessage } from "./observerSocket"

const originalWebSocket = globalThis.WebSocket

afterEach(() => {
  globalThis.WebSocket = originalWebSocket
})

/** Captures the instance the module constructs, so a test can fire the
 * handlers it registered. */
function stubWebSocket() {
  const instances: Array<Record<string, unknown>> = []
  class FakeWebSocket {
    onopen: (() => void) | null = null
    onmessage: ((event: { data: unknown }) => void) | null = null
    onclose: (() => void) | null = null
    close() {}
    constructor(public url: string) {
      instances.push(this as unknown as Record<string, unknown>)
    }
  }
  globalThis.WebSocket = FakeWebSocket as unknown as typeof WebSocket
  return instances
}

function connectWithRecording() {
  const instances = stubWebSocket()
  const messages: ObserverMessage[] = []
  const states: ObserverConnectionState[] = []
  const connection = connect({
    onMessage: (message) => messages.push(message),
    onStateChange: (state) => states.push(state),
  })
  const socket = instances[0] as unknown as {
    onmessage: ((event: { data: unknown }) => void) | null
  }
  return { connection, socket, messages, states }
}

test("a JSON text frame reaches the handler parsed", () => {
  const { connection, socket, messages } = connectWithRecording()

  socket.onmessage?.({ data: JSON.stringify({ type: "reply.text", text: "hello" }) })

  expect(messages).toEqual([{ type: "reply.text", text: "hello" }])
  connection.close()
})

// IN-08: `JSON.parse` was unguarded, so an unparseable text frame threw a
// `SyntaxError` out of the `onmessage` handler -- where the binary branch
// beside it already dropped a frame it could not read.
test("an unparseable text frame is dropped, not thrown out of the handler", () => {
  const { connection, socket, messages } = connectWithRecording()

  expect(() => socket.onmessage?.({ data: "this is not JSON at all" })).not.toThrow()
  expect(messages).toEqual([])

  // And the connection is still live: the next good frame still arrives.
  socket.onmessage?.({ data: JSON.stringify({ type: "reply.text", text: "still here" }) })
  expect(messages).toEqual([{ type: "reply.text", text: "still here" }])
  connection.close()
})

test("a binary frame is dropped the same way -- an observer never receives one (D-05)", () => {
  const { connection, socket, messages } = connectWithRecording()

  socket.onmessage?.({ data: new ArrayBuffer(8) })

  expect(messages).toEqual([])
  connection.close()
})
