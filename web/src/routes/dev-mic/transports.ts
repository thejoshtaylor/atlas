// The two transports the dev harness can drive, ported from
// `web/public/dev-mic/{index.html,webrtc.js}` (D-19's fold-in: one
// surface, authenticated, not two). Behavior is unchanged -- the same
// microphone constraints, the same message shapes, the same gapless
// playback -- only the language (TypeScript) and how a request reaches
// the network changed.
//
// The turn routes this file calls (`/webrtc/offer`, `/ws/turn`) now sit
// behind `require_role(Role.OPERATOR)` (plan 03-05, T-03-32) -- this is
// the one change to carried-forward code this fold-in actually requires
// (Task 3's own instruction): `apiFetch` sends `credentials: "same-origin"`
// on every request, where the original vanilla `fetch("/webrtc/offer", ...)`
// relied on the browser's own same-origin default. A `WebSocket` connection
// already carries the session cookie automatically for a same-origin
// address -- nothing to change there.
import { apiFetch } from "@/lib/api"
import { GaplessPlayer } from "./gaplessPlayer"

export interface TurnTimingMessage {
  type: "turn.timing"
  end_of_speech_to_first_audio_ms: number | null
  end_of_speech_to_answer_audio_ms: number | null
  turn_outcome: string
  stage_durations_ms: Record<string, number | null>
}

export type TransportMessage =
  | { type: "transcript.partial"; text: string }
  | { type: "reply.text"; text: string }
  | TurnTimingMessage

export interface TransportHandlers {
  onMessage: (message: TransportMessage) => void
}

export interface ActiveTransport {
  close: () => void
}

const MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  channelCount: 1,
  sampleRate: 16000,
}

// The AudioWorklet module -- moved to `web/public/pcm-worklet.js` (a
// top-level static asset, still unauthenticated at the HTTP layer like
// every static file this application serves; the turn it could help
// start is what plan 03-05 already gated, not the inert script itself).
const PCM_WORKLET_URL = "/pcm-worklet.js"

function handleTransportMessage(
  event: MessageEvent,
  player: GaplessPlayer,
  handlers: TransportHandlers,
): void {
  if (typeof event.data === "string") {
    handlers.onMessage(JSON.parse(event.data) as TransportMessage)
    return
  }
  player.playChunk(new Int16Array(event.data as ArrayBuffer))
}

export async function startWebsocketListening(handlers: TransportHandlers): Promise<ActiveTransport> {
  const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS })

  const audioContext = new AudioContext({ sampleRate: 16000 })
  await audioContext.audioWorklet.addModule(PCM_WORKLET_URL)
  const source = audioContext.createMediaStreamSource(mediaStream)
  const worklet = new AudioWorkletNode(audioContext, "pcm-worklet")

  // 24kHz matches `tts.browser_sample_rate` in `config.example.yaml`.
  const player = new GaplessPlayer(audioContext, 24000)

  const scheme = window.location.protocol === "https:" ? "wss" : "ws"
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/turn`)
  ws.binaryType = "arraybuffer"
  ws.onmessage = (event) => handleTransportMessage(event, player, handlers)

  await new Promise<void>((resolve) => {
    ws.onopen = () => {
      worklet.port.onmessage = (event: MessageEvent) => {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(event.data as ArrayBuffer)
        }
      }
      source.connect(worklet)
      resolve()
    }
  })

  return {
    close: () => {
      ws.close()
      mediaStream.getTracks().forEach((track) => track.stop())
      void audioContext.close()
    },
  }
}

function waitForIceGatheringComplete(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") {
    return Promise.resolve()
  }
  return new Promise((resolve) => {
    function checkState() {
      if (pc.iceGatheringState === "complete") {
        pc.removeEventListener("icegatheringstatechange", checkState)
        resolve()
      }
    }
    pc.addEventListener("icegatheringstatechange", checkState)
  })
}

export async function startWebrtcListening(handlers: TransportHandlers): Promise<ActiveTransport> {
  const mediaStream = await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS })

  const pc = new RTCPeerConnection({ iceServers: [] })
  const dataChannel = pc.createDataChannel("events")
  dataChannel.binaryType = "arraybuffer"
  const player = new GaplessPlayer(new AudioContext(), 24000)
  dataChannel.onmessage = (event) => handleTransportMessage(event, player, handlers)

  for (const track of mediaStream.getAudioTracks()) {
    pc.addTrack(track, mediaStream)
  }

  const offer = await pc.createOffer()
  await pc.setLocalDescription(offer)
  await waitForIceGatheringComplete(pc)

  const answer = await apiFetch<{ sdp: string; type: string }>("/webrtc/offer", {
    method: "POST",
    body: { sdp: pc.localDescription!.sdp, type: pc.localDescription!.type },
  })
  await pc.setRemoteDescription(answer as RTCSessionDescriptionInit)

  return {
    close: () => {
      dataChannel.close()
      pc.close()
      mediaStream.getTracks().forEach((track) => track.stop())
    },
  }
}

/** The page learns which transport is configured from the server rather
 * than guessing (unchanged from the original page's own comment) --
 * routed through `apiFetch` rather than a bare `fetch` for the same
 * credentials-explicitness reason as the two calls above. */
export function fetchConfiguredTransport(): Promise<"websocket" | "webrtc"> {
  return apiFetch<{ transport: "websocket" | "webrtc" }>("/transport").then((body) => body.transport)
}
