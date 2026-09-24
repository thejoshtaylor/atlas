// The always-on browser listener. One per tab, kept at module scope so it
// survives navigation: an operator starts it on Home and it keeps
// listening while they move around the app.
//
// The wake word runs on the ATLAS server (`/ws/listen`), the same engine
// the camera uses. Speech-to-text opens only after the server hears the
// wake phrase, and this page sends audio only while the server is reading
// it -- never while ATLAS is thinking or speaking.
import { create } from "zustand"
import { GaplessPlayer } from "@/routes/dev-mic/gaplessPlayer"

export type ListenStatus =
  | "off"
  | "connecting"
  | "armed"
  | "hearing"
  | "thinking"
  | "speaking"
  | "denied"
  | "error"

export interface ListenState {
  status: ListenStatus
  wakePhrase: string | null
  heard: string
  reply: string
  error: string | null
  startedAt: number | null
  turns: number
  lastReplyMs: number | null
}

const INITIAL: ListenState = {
  status: "off",
  wakePhrase: null,
  heard: "",
  reply: "",
  error: null,
  startedAt: null,
  turns: 0,
  lastReplyMs: null,
}

export const useListenStore = create<ListenState>(() => INITIAL)
const set = useListenStore.setState

/** Statuses in which the microphone is open and the listener is running. */
export function isListening(status: ListenStatus): boolean {
  return status !== "off" && status !== "denied" && status !== "error"
}

const MIC_CONSTRAINTS: MediaTrackConstraints = {
  echoCancellation: true,
  noiseSuppression: true,
  autoGainControl: true,
  channelCount: 1,
  sampleRate: 16000,
}
const RECONNECT_MS = 3000
// A turn that goes quiet this long without any event is over, whatever
// the server failed to say. The listener must never stay deaf.
const TURN_WATCHDOG_MS = 45_000

interface Rig {
  stream: MediaStream
  ctx: AudioContext
  mic: AnalyserNode
  out: AnalyserNode
  player: GaplessPlayer
  ws: WebSocket | null
  wakeLock: WakeLockSentinel | null
  retry: ReturnType<typeof setTimeout> | null
  watchdog: ReturnType<typeof setTimeout> | null
  settle: ReturnType<typeof setTimeout> | null
}

let rig: Rig | null = null

/** The live analysers the dial draws from, or null while off. */
export function listenAnalysers(): { mic: AnalyserNode; out: AnalyserNode } | null {
  return rig ? { mic: rig.mic, out: rig.out } : null
}

function sendsAudio(status: ListenStatus): boolean {
  return status === "armed" || status === "hearing"
}

function armWatchdog() {
  if (!rig) return
  if (rig.watchdog) clearTimeout(rig.watchdog)
  rig.watchdog = setTimeout(() => set({ status: "armed" }), TURN_WATCHDOG_MS)
}

function playWakeCue(r: Rig) {
  const now = r.ctx.currentTime
  const osc = r.ctx.createOscillator()
  const gain = r.ctx.createGain()
  osc.frequency.setValueAtTime(660, now)
  osc.frequency.exponentialRampToValueAtTime(990, now + 0.12)
  gain.gain.setValueAtTime(0.0001, now)
  gain.gain.exponentialRampToValueAtTime(0.08, now + 0.02)
  gain.gain.exponentialRampToValueAtTime(0.0001, now + 0.18)
  osc.connect(gain).connect(r.out)
  osc.start(now)
  osc.stop(now + 0.2)
}

function handleEvent(r: Rig, event: { type: string; [key: string]: unknown }) {
  switch (event.type) {
    case "listen.ready":
      set({ status: "armed", wakePhrase: (event.wake_phrase as string) || null, error: null })
      return
    case "wake.heard":
      playWakeCue(r)
      set({ status: "hearing", heard: "", reply: "" })
      armWatchdog()
      return
    case "transcript.partial":
      set({ heard: String(event.text ?? "") })
      armWatchdog()
      return
    case "transcript.final":
      set({ status: "thinking", heard: String(event.text ?? "") })
      armWatchdog()
      return
    case "reply.text":
      set({ reply: String(event.text ?? "") })
      armWatchdog()
      return
    case "turn.timing": {
      if (r.watchdog) clearTimeout(r.watchdog)
      const ms = event.end_of_speech_to_first_audio_ms
      set((state) => ({ turns: state.turns + 1, lastReplyMs: typeof ms === "number" ? ms : state.lastReplyMs }))
      // Back to armed only once the reply has finished playing here.
      r.settle = setTimeout(() => set({ status: "armed" }), r.player.endsIn() * 1000 + 150)
      return
    }
  }
}

function connect() {
  const r = rig
  if (!r) return
  r.retry = null
  const scheme = window.location.protocol === "https:" ? "wss" : "ws"
  const ws = new WebSocket(`${scheme}://${window.location.host}/ws/listen`)
  ws.binaryType = "arraybuffer"
  r.ws = ws
  let ready = false
  let lastReply = ""
  set({ status: "connecting" })

  ws.onmessage = (message: MessageEvent) => {
    if (typeof message.data !== "string") {
      r.player.playChunk(new Int16Array(message.data as ArrayBuffer))
      if (useListenStore.getState().status !== "speaking") set({ status: "speaking" })
      armWatchdog()
      return
    }
    const event = JSON.parse(message.data) as { type: string; [key: string]: unknown }
    if (event.type === "listen.ready") ready = true
    if (event.type === "reply.text") lastReply = String(event.text ?? "")
    handleEvent(r, event)
  }

  ws.onclose = () => {
    if (rig !== r || r.ws !== ws) return
    r.ws = null
    if (!ready) {
      // Refused before it ever listened (a degraded provider, a lost
      // session): retrying would only repeat the refusal.
      stopListening()
      set({ status: "error", error: lastReply || "ATLAS closed the connection before it started listening." })
      return
    }
    set({ status: "connecting" })
    r.retry = setTimeout(connect, RECONNECT_MS)
  }
}

async function holdScreenAwake(r: Rig) {
  try {
    r.wakeLock = (await navigator.wakeLock?.request("screen")) ?? null
  } catch {
    r.wakeLock = null // not allowed here, or not supported: listening still works
  }
}

function onVisibility() {
  if (rig && document.visibilityState === "visible") void holdScreenAwake(rig)
}

export async function startListening(): Promise<void> {
  if (rig) return
  set({ ...INITIAL, status: "connecting" })
  let stream: MediaStream
  try {
    stream = await navigator.mediaDevices.getUserMedia({ audio: MIC_CONSTRAINTS })
  } catch (caught) {
    const denied = caught instanceof DOMException && caught.name === "NotAllowedError"
    set({
      status: denied ? "denied" : "error",
      error: denied ? null : "This browser could not open a microphone.",
    })
    return
  }

  const ctx = new AudioContext({ sampleRate: 16000 })
  await ctx.audioWorklet.addModule("/pcm-worklet.js")
  const source = ctx.createMediaStreamSource(stream)
  const worklet = new AudioWorkletNode(ctx, "pcm-worklet")
  const mic = ctx.createAnalyser()
  const out = ctx.createAnalyser()
  for (const analyser of [mic, out]) {
    analyser.fftSize = 256
    analyser.smoothingTimeConstant = 0.6
  }
  out.connect(ctx.destination)
  source.connect(mic)
  source.connect(worklet)

  const r: Rig = {
    stream,
    ctx,
    mic,
    out,
    player: new GaplessPlayer(ctx, 24000, out),
    ws: null,
    wakeLock: null,
    retry: null,
    watchdog: null,
    settle: null,
  }
  rig = r
  worklet.port.onmessage = (event: MessageEvent) => {
    const ws = r.ws
    if (ws?.readyState === WebSocket.OPEN && sendsAudio(useListenStore.getState().status)) {
      ws.send(event.data as ArrayBuffer)
    }
  }

  set({ startedAt: Date.now() })
  void holdScreenAwake(r)
  document.addEventListener("visibilitychange", onVisibility)
  connect()
}

export function stopListening(): void {
  const r = rig
  if (!r) return
  rig = null
  for (const timer of [r.retry, r.watchdog, r.settle]) if (timer) clearTimeout(timer)
  if (r.ws) {
    r.ws.onclose = null
    r.ws.close()
  }
  r.stream.getTracks().forEach((track) => track.stop())
  void r.ctx.close()
  void r.wakeLock?.release().catch(() => {})
  document.removeEventListener("visibilitychange", onVisibility)
  set({ ...INITIAL })
}
