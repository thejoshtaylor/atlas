import * as React from "react"
import { Button } from "@/components/ui/button"
import { ErrorState } from "@/components/state/ErrorState"
import { formatMs } from "@/lib/format"
import {
  fetchConfiguredTransport,
  startWebrtcListening,
  startWebsocketListening,
  type ActiveTransport,
  type TransportMessage,
  type TurnTimingMessage,
} from "./transports"

// The eight stages `atlas.timing.TurnTimings` actually records
// (`timing.py`'s own module docstring: "eight timestamps and their
// derived durations, nothing else") -- 03-UI-SPEC.md's own overflow row
// and this plan's own task text both say "seven"; the real dataclass has
// eight fields (`turn_started_at` through `answer_audio_at`), confirmed
// directly against the source rather than the plan's recollection of it.
// Rendered here against the real field count, not the stated one -- see
// this plan's own SUMMARY for the discrepancy, named rather than quietly
// dropping a real stage to match a miscounted spec.
const STAGE_LABELS: Record<string, string> = {
  turn_started_at: "Turn started",
  stt_socket_open_at: "STT socket open",
  first_partial_at: "First partial",
  stt_final_at: "STT final (end of speech)",
  brain_first_token_at: "Brain first token",
  tool_rounds_done_at: "Tool rounds done",
  first_audio_at: "First audio",
  answer_audio_at: "Answer audio",
}
const STAGE_ORDER = Object.keys(STAGE_LABELS)

/**
 * The developer browser-microphone page, folded into the application as
 * an authenticated route (D-19, T-03-62) -- ported from
 * `web/public/dev-mic/{index.html,webrtc.js,pcm-worklet.js}`, which this
 * task deletes once this route serves the same behavior. It is a
 * development and test tool; this plan's own objective decides exactly
 * two things are owed to it and implements them here:
 *
 *  - A denied microphone permission states what happened (`micDenied`
 *    below) rather than leaving the toggle idle with no explanation.
 *  - A turn that returned no transcript (`turn_outcome ===
 *    "empty_transcript"`, `atlas/turn/controller.py`) states that,
 *    too, rather than a blank reply the operator has to interpret.
 *
 * Everything else about the page is carried forward as it stood --
 * neither redesigned nor restyled beyond the shared page shell every
 * other authenticated screen already uses.
 */
export function DevMicRoute() {
  const [listening, setListening] = React.useState(false)
  const [partial, setPartial] = React.useState("")
  const [reply, setReply] = React.useState("")
  const [timingHistory, setTimingHistory] = React.useState<TurnTimingMessage[]>([])
  const [micDenied, setMicDenied] = React.useState(false)
  const [emptyTranscript, setEmptyTranscript] = React.useState(false)
  const [transport, setTransport] = React.useState<"websocket" | "webrtc" | null>(null)
  const activeRef = React.useRef<ActiveTransport | null>(null)

  React.useEffect(() => {
    let cancelled = false
    // The page learns which transport is configured from the server
    // rather than guessing -- the toggle stays disabled until this
    // resolves, matching the original page's own comment and behavior.
    void fetchConfiguredTransport().then((value) => {
      if (!cancelled) setTransport(value)
    })
    return () => {
      cancelled = true
    }
  }, [])

  const handleMessage = (message: TransportMessage) => {
    if (message.type === "transcript.partial") {
      setPartial(message.text)
      return
    }
    if (message.type === "reply.text") {
      setReply(message.text)
      setPartial("")
      return
    }
    setTimingHistory((history) => [message, ...history])
    setEmptyTranscript(message.turn_outcome === "empty_transcript")
  }

  const start = async () => {
    setMicDenied(false)
    try {
      activeRef.current =
        transport === "webrtc"
          ? await startWebrtcListening({ onMessage: handleMessage })
          : await startWebsocketListening({ onMessage: handleMessage })
      setListening(true)
    } catch (caught) {
      // getUserMedia rejects with a DOMException named NotAllowedError
      // when the operator (or the browser's own permission policy)
      // denies microphone access -- the one failure this page must state
      // rather than leave the toggle looking like nothing happened.
      if (caught instanceof DOMException && caught.name === "NotAllowedError") {
        setMicDenied(true)
        return
      }
      throw caught
    }
  }

  const stop = () => {
    activeRef.current?.close()
    activeRef.current = null
    setListening(false)
  }

  const handleToggle = () => {
    if (listening) {
      stop()
    } else {
      void start()
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Developer microphone</h1>
        <p className="text-body text-muted-foreground">
          A dev harness for one voice turn: no camera, no wake word. Press the button,
          speak one command, and the reply plays back through this page.
        </p>
      </div>

      <Button type="button" onClick={handleToggle} disabled={transport === null}>
        {listening ? "Stop listening" : "Start listening"}
      </Button>

      {micDenied ? (
        <ErrorState
          message="Microphone access was denied."
          detail="Allow microphone access for this site in your browser's settings, then try again."
          onRetry={() => void start()}
        />
      ) : null}

      {partial ? <p className="text-body text-muted-foreground">{partial}</p> : null}
      {reply ? <p className="text-body text-foreground">{reply}</p> : null}
      {emptyTranscript ? (
        <p className="text-body text-destructive">
          The last turn understood no speech. Try again, closer to the microphone.
        </p>
      ) : null}

      <div className="flex flex-col gap-3">
        <h2 className="text-heading font-semibold">Stage timings</h2>
        <p className="text-label text-muted-foreground">
          Newest turn first. Every number here came from the server -- this page computes
          no duration of its own.
        </p>
        {timingHistory.length === 0 ? (
          <p className="text-body text-muted-foreground">No turns yet.</p>
        ) : (
          <div className="flex flex-col gap-4">
            {timingHistory.map((entry, index) => (
              <div key={index} className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
                <p className="text-body text-foreground">
                  End of speech to first audio: {formatMs(entry.end_of_speech_to_first_audio_ms)}
                </p>
                <p className="text-body text-foreground">
                  End of speech to answer audio: {formatMs(entry.end_of_speech_to_answer_audio_ms)} (
                  {entry.turn_outcome})
                </p>
                {/* Every stage as a stacked label/value list, never a
                 * table -- this plan's fourth held-out visual check
                 * (03-UI-SPEC.md's overflow row for this page: fits at
                 * 375px with no horizontal scroll). */}
                {STAGE_ORDER.map((stage) => (
                  <div key={stage} className="flex items-baseline justify-between gap-2">
                    <span className="text-label text-muted-foreground">{STAGE_LABELS[stage]}</span>
                    <span className="readout text-label text-foreground">
                      {formatMs(entry.stage_durations_ms[stage])}
                    </span>
                  </div>
                ))}
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
