import * as React from "react"
import { Mic, Square } from "lucide-react"
import { Button } from "@/components/ui/button"
import { ErrorState } from "@/components/state/ErrorState"
import { formatMs } from "@/lib/format"
import {
  isListening,
  startListening,
  stopListening,
  useListenStore,
  type ListenStatus,
} from "@/lib/listener"
import { ListenDial } from "./ListenDial"

function statusLine(status: ListenStatus, wakePhrase: string | null): { title: string; detail: string } {
  const phrase = wakePhrase ? `“${wakePhrase}”` : "the wake phrase"
  switch (status) {
    case "connecting":
      return { title: "Connecting", detail: "Opening a line to ATLAS." }
    case "armed":
      return { title: `Listening for ${phrase}`, detail: "Say it, then your command." }
    case "hearing":
      return { title: "Hearing you", detail: "Keep going. ATLAS stops listening when you pause." }
    case "thinking":
      return { title: "Working on it", detail: "Deciding what to do." }
    case "speaking":
      return { title: "Replying", detail: "Playing on this device." }
    default:
      return { title: "Off", detail: "ATLAS is not listening on this device." }
  }
}

function formatUptime(ms: number): string {
  const total = Math.floor(ms / 1000)
  const h = Math.floor(total / 3600)
  const m = Math.floor((total % 3600) / 60)
  const s = total % 60
  const pad = (n: number) => String(n).padStart(2, "0")
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${pad(m)}:${pad(s)}`
}

function Uptime({ since }: { since: number | null }) {
  const [now, setNow] = React.useState(() => Date.now())
  React.useEffect(() => {
    if (since === null) return
    const timer = setInterval(() => setNow(Date.now()), 1000)
    return () => clearInterval(timer)
  }, [since])
  return <>{since === null ? "—" : formatUptime(now - since)}</>
}

/**
 * The always-on listener. The dial shows what ATLAS is doing right now;
 * the button keeps this device listening until it is pressed again, on
 * any page.
 */
export function ListenRoute() {
  const state = useListenStore()
  const on = isListening(state.status)
  const line = statusLine(state.status, state.wakePhrase)

  const toggle = (
    <Button
      type="button"
      size="lg"
      variant={on ? "outline" : "default"}
      className="h-11 w-full gap-2 sm:w-auto sm:px-6"
      onClick={() => (on ? stopListening() : void startListening())}
    >
      {on ? <Square className="size-4" aria-hidden /> : <Mic className="size-4" aria-hidden />}
      {on ? "Stop listening" : "Listen on this device"}
    </Button>
  )

  return (
    <div className="flex flex-col gap-8">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Listen</h1>
        <p className="max-w-prose text-body text-muted-foreground">
          This device becomes a microphone and speaker for ATLAS. It keeps listening while you use the rest of the
          app.
        </p>
      </div>

      <section className="grid items-center gap-8 md:grid-cols-[minmax(0,28rem)_1fr] md:gap-12">
        <ListenDial className="mx-auto w-full max-w-[22rem] md:max-w-none" />

        <div className="flex min-w-0 flex-col gap-6">
          <div aria-live="polite" className="flex flex-col gap-1">
            <p className="text-heading font-semibold text-foreground">{line.title}</p>
            <p className="text-body text-muted-foreground">{line.detail}</p>
          </div>

          {state.heard || state.reply ? (
            <dl className="flex flex-col gap-3 border-l border-border pl-4">
              {state.heard ? (
                <div className="flex flex-col gap-0.5">
                  <dt className="text-label text-muted-foreground">You said</dt>
                  <dd className="text-body text-foreground">{state.heard}</dd>
                </div>
              ) : null}
              {state.reply ? (
                <div className="flex flex-col gap-0.5">
                  <dt className="text-label text-muted-foreground">ATLAS</dt>
                  <dd className="text-body text-foreground">{state.reply}</dd>
                </div>
              ) : null}
            </dl>
          ) : null}

          <dl className="grid grid-cols-3 border-y border-border py-3">
            <div className="flex flex-col gap-0.5">
              <dt className="text-label text-muted-foreground">On for</dt>
              <dd className="readout text-body text-foreground">
                <Uptime since={on ? state.startedAt : null} />
              </dd>
            </div>
            <div className="flex flex-col gap-0.5 border-l border-border pl-4">
              <dt className="text-label text-muted-foreground">Turns</dt>
              <dd className="readout text-body text-foreground">{on ? state.turns : "—"}</dd>
            </div>
            <div className="flex flex-col gap-0.5 border-l border-border pl-4">
              <dt className="text-label text-muted-foreground">First reply</dt>
              <dd className="readout text-body text-foreground">
                {on && state.lastReplyMs !== null ? formatMs(state.lastReplyMs) : "—"}
              </dd>
            </div>
          </dl>

          {state.status === "denied" ? (
            <ErrorState
              message="Microphone access was denied."
              detail="Allow microphone access for this site in your browser's settings, then try again."
              onRetry={() => void startListening()}
            />
          ) : null}
          {state.status === "error" && state.error ? (
            <ErrorState message={state.error} onRetry={() => void startListening()} />
          ) : null}

          {state.status === "denied" || state.status === "error" ? null : toggle}

          <p className="max-w-prose text-label text-muted-foreground">
            Your ATLAS server listens for the wake phrase. No audio goes to speech-to-text until ATLAS hears it. Keep
            this screen on: phones turn the microphone off when they lock.
          </p>
        </div>
      </section>
    </div>
  )
}
