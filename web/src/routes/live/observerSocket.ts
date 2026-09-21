// The read-only observer connection for `/live` (WEB-06, D-05). Follows
// `dev-mic/transports.ts`'s own scheme-selection and host-construction
// shape, with everything that file needs for a *participant* turn removed:
// no `binaryType`, no audio worklet, no `getUserMedia` call at all. D-05's
// whole point is that watching what a source heard never opens a
// microphone of its own -- this module has no code path that could.
//
// Every frame this socket ever sends is JSON text -- parsed the same way
// `transports.ts`'s own `handleTransportMessage` parses its string branch,
// and only that branch; there is no binary message this connection is ever
// meant to receive.

export type ObserverConnectionState = "connecting" | "connected" | "disconnected"

export interface ObserverMessage {
  type: string
  [key: string]: unknown
}

export interface ObserverHandlers {
  onMessage: (message: ObserverMessage) => void
  onStateChange: (state: ObserverConnectionState) => void
}

export interface ObserverConnection {
  close: () => void
}

const RECONNECT_BASE_DELAY_MS = 500
const RECONNECT_MAX_DELAY_MS = 8000

/**
 * Opens `/ws/sessions/live` and reconnects on close with a bounded
 * exponential backoff, until `close()` is called. Synchronous, unlike
 * `dev-mic/transports.ts`'s own `start*Listening` functions -- there is no
 * permission prompt to await here, so the connection can start on the very
 * first render.
 *
 * The badge this connection drives describes the *current* fact, not the
 * history of how it got there (08-UI-SPEC.md's own reasoning): a socket
 * that never opened at all and one that dropped after connecting both read
 * as `"disconnected"`, and both retry the same way.
 */
export function connect(handlers: ObserverHandlers): ObserverConnection {
  let closed = false
  let socket: WebSocket | null = null
  let attempt = 0
  let reconnectTimer: ReturnType<typeof setTimeout> | null = null

  function open(): void {
    if (closed) return
    handlers.onStateChange("connecting")

    const scheme = window.location.protocol === "https:" ? "wss" : "ws"
    socket = new WebSocket(`${scheme}://${window.location.host}/ws/sessions/live`)

    socket.onopen = () => {
      attempt = 0
      handlers.onStateChange("connected")
    }

    socket.onmessage = (event: MessageEvent) => {
      if (typeof event.data !== "string") {
        // The observer socket never sends binary frames (D-05) -- if one
        // somehow arrived, there is nothing meaningful to parse it as.
        return
      }
      handlers.onMessage(JSON.parse(event.data) as ObserverMessage)
    }

    socket.onclose = () => {
      if (closed) return
      handlers.onStateChange("disconnected")
      scheduleReconnect()
    }
  }

  function scheduleReconnect(): void {
    if (closed) return
    const delay = Math.min(RECONNECT_BASE_DELAY_MS * 2 ** attempt, RECONNECT_MAX_DELAY_MS)
    attempt += 1
    reconnectTimer = setTimeout(open, delay)
  }

  open()

  return {
    close: () => {
      closed = true
      if (reconnectTimer !== null) {
        clearTimeout(reconnectTimer)
      }
      socket?.close()
    },
  }
}
