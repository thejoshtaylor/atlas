import * as React from "react"
import { AtlasGlobe } from "@/components/brand/AtlasGlobe"
import { listenAnalysers, useListenStore, type ListenStatus } from "@/lib/listener"
import { cn } from "@/lib/utils"

// The listener's dial: a graduated ring of ticks around the globe, the
// same tick scale an armillary carries. Two waveforms pulse round it:
//
//   gold   incoming audio -- the microphone, as the server reads it
//   blue   outgoing audio -- ATLAS's reply playing on this device
//
// Between them sits the resting dial in porcelain. While ATLAS is working
// (no audio either way) a porcelain sweep runs the dial and the globe
// turns fast. Every pass is one batched stroke per color, plus the sweep's
// short trail.

const TICKS = 120
const MAJOR_EVERY = 10

interface Look {
  /** How far incoming (mic) audio moves the ticks, 0..1. */
  incoming: number
  /** Opacity of the resting dial. */
  alpha: number
}

const LOOKS: Record<ListenStatus, Look> = {
  off: { incoming: 0, alpha: 0.32 },
  denied: { incoming: 0, alpha: 0.32 },
  error: { incoming: 0, alpha: 0.32 },
  connecting: { incoming: 0, alpha: 0.45 },
  // Armed: the room, quietly. Hearing: the voice, at full reach.
  armed: { incoming: 0.5, alpha: 0.55 },
  hearing: { incoming: 1, alpha: 0.55 },
  // The mic is not being sent while ATLAS works or replies, so it is not shown.
  thinking: { incoming: 0, alpha: 0.5 },
  speaking: { incoming: 0, alpha: 0.4 },
}

const SWEEP_TURN_MS = 1400
const SWEEP_TRAIL = 30

function cssVar(name: string): string {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim()
}

export function ListenDial({ className }: { className?: string }) {
  const status = useListenStore((state) => state.status)
  const canvasRef = React.useRef<HTMLCanvasElement>(null)

  React.useEffect(() => {
    const canvas = canvasRef.current
    const g = canvas?.getContext("2d")
    if (!canvas || !g) return

    const ink = cssVar("--foreground") || "#f4f1ea"
    const gold = cssVar("--primary") || "#ecc063"
    const blue = cssVar("--voice") || "#7fb8e0"
    const reduced = window.matchMedia("(prefers-reduced-motion: reduce)")
    const heard = new Float32Array(TICKS)
    const said = new Float32Array(TICKS)
    const bins = new Uint8Array(128)
    let incoming = 0
    let alpha = LOOKS.off.alpha
    let frame = 0

    const resize = () => {
      const dpr = window.devicePixelRatio || 1
      canvas.width = Math.round(canvas.clientWidth * dpr)
      canvas.height = Math.round(canvas.clientHeight * dpr)
    }
    resize()
    const observer = new ResizeObserver(resize)
    observer.observe(canvas)

    // Reduced motion keeps the level (it is feedback) but settles slower.
    const follow = (levels: Float32Array, analyser: AnalyserNode | null, gain: number) => {
      if (analyser) analyser.getByteFrequencyData(bins)
      const ease = reduced.matches ? 0.12 : 0.3
      for (let i = 0; i < TICKS; i++) {
        // Mirrored left/right from the top, low frequencies at 12 o'clock.
        const k = i <= TICKS / 2 ? i : TICKS - i
        const bin = 2 + Math.floor(k * 0.9)
        const target = analyser && gain > 0 ? Math.pow(bins[bin]! / 255, 1.5) * gain : 0
        levels[i]! += (target - levels[i]!) * ease
      }
    }

    const draw = (now: number) => {
      frame = requestAnimationFrame(draw)
      const status = useListenStore.getState().status
      const look = LOOKS[status]
      const analysers = listenAnalysers()
      incoming += (look.incoming - incoming) * 0.15
      alpha += (look.alpha - alpha) * 0.12
      follow(heard, analysers?.mic ?? null, incoming)
      follow(said, analysers?.out ?? null, 1)

      const w = canvas.width
      const c = w / 2
      const inner = w * 0.335
      const reach = w * 0.125
      g.clearRect(0, 0, w, w)
      g.lineCap = "round"
      g.lineWidth = Math.max(1.5, w * 0.0055)

      const tick = (i: number, level: number) => {
        const angle = (i / TICKS) * Math.PI * 2 - Math.PI / 2
        const base = i % MAJOR_EVERY === 0 ? w * 0.034 : w * 0.016
        const length = base + level * reach
        const cos = Math.cos(angle)
        const sin = Math.sin(angle)
        g.moveTo(c + cos * inner, c + sin * inner)
        g.lineTo(c + cos * (inner + length), c + sin * (inner + length))
      }
      // One waveform as one stroke: only the ticks it actually moves, at
      // an opacity that rises with its own loudness.
      const pulse = (levels: Float32Array, color: string) => {
        let peak = 0
        g.beginPath()
        for (let i = 0; i < TICKS; i++) {
          if (levels[i]! < 0.02) continue
          peak = Math.max(peak, levels[i]!)
          tick(i, levels[i]!)
        }
        if (peak === 0) return
        g.globalAlpha = 0.55 + 0.45 * Math.min(1, peak * 2)
        g.strokeStyle = color
        g.stroke()
      }

      g.globalAlpha = alpha
      g.strokeStyle = ink
      g.beginPath()
      for (let i = 0; i < TICKS; i++) tick(i, 0)
      g.stroke()

      pulse(heard, gold)
      pulse(said, blue)

      if (status === "thinking") {
        g.strokeStyle = ink
        if (reduced.matches) {
          g.globalAlpha = 0.9
          g.beginPath()
          for (let i = 0; i < TICKS; i++) tick(i, 0.12)
          g.stroke()
        } else {
          const head = Math.floor(((now % SWEEP_TURN_MS) / SWEEP_TURN_MS) * TICKS)
          for (let t = 0; t < SWEEP_TRAIL; t++) {
            const fade = Math.pow(1 - t / SWEEP_TRAIL, 2)
            g.globalAlpha = fade
            g.beginPath()
            tick((head - t + TICKS) % TICKS, fade * 0.45)
            g.stroke()
          }
        }
      }
      g.globalAlpha = 1
    }
    frame = requestAnimationFrame(draw)

    return () => {
      cancelAnimationFrame(frame)
      observer.disconnect()
    }
  }, [])

  return (
    <div className={cn("listen-dial relative aspect-square", className)} data-status={status}>
      <canvas ref={canvasRef} aria-hidden className="absolute inset-0 size-full" />
      <AtlasGlobe
        spinning={status !== "off" && status !== "denied" && status !== "error"}
        className="absolute inset-[22%] size-[56%] text-foreground/75"
      />
    </div>
  )
}
