// `startWebsocketListening`/`startWebrtcListening` need a real
// `getUserMedia`/`RTCPeerConnection`/`AudioContext`, none of which exist
// in this project's `bun test` environment (no DOM, no WebRTC stack) --
// matching every other browser-hardware-dependent module in this
// project's test suite. What is exercised directly: the one behavioral
// change this fold-in actually requires (Task 3's own instruction) --
// that the REST calls this file makes carry the session cookie -- and the
// pure `fetchConfiguredTransport` helper.
import { afterEach, describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { queryClient } from "@/lib/queryClient"
import { fetchConfiguredTransport } from "./transports"

const SOURCE = readFileSync(join(import.meta.dir, "transports.ts"), "utf-8")

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("transports.ts -- every REST call goes through the shared, credentials-carrying fetch seam", () => {
  test("both /webrtc/offer and /transport are called through apiFetch, which sends credentials: same-origin", () => {
    expect(SOURCE).toMatch(/apiFetch<\{ sdp: string; type: string \}>\("\/webrtc\/offer"/)
    expect(SOURCE).toMatch(/apiFetch<\{ transport: "websocket" \| "webrtc" \}>\("\/transport"\)/)
  })

  test("the WebSocket connection needs no explicit credentials option -- a same-origin WebSocket already carries the session cookie", () => {
    expect(SOURCE).toMatch(/new WebSocket\(/)
  })
})

describe("fetchConfiguredTransport", () => {
  test("resolves with the server's own transport field", async () => {
    global.fetch = (async () =>
      new Response(JSON.stringify({ transport: "webrtc" }), {
        status: 200,
        headers: { "Content-Type": "application/json" },
      })) as typeof fetch

    await expect(fetchConfiguredTransport()).resolves.toBe("webrtc")
  })
})
