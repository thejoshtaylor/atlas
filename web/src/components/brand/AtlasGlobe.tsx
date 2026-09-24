import { cn } from "@/lib/utils"

const MERIDIANS = 6
const LATITUDES = [-60, -30, 0, 30, 60]
const R = 40

export interface AtlasGlobeProps {
  className?: string
  /** Turn the meridians -- the "listening" state. Static otherwise, and
   * always static under prefers-reduced-motion (index.css). */
  spinning?: boolean
  /** Decorative by default; pass a label when the globe is the only
   * thing naming the product in its context. */
  label?: string
}

/**
 * The ATLAS mark: a wireframe globe of meridians and parallels with a
 * gold ecliptic ring across it. The meridians turn by scaling each
 * ellipse's x-axis through cos(t) (a CSS keyframe with sine easing,
 * staggered by phase), so the rotation costs no JavaScript.
 */
export function AtlasGlobe({ className, spinning = false, label }: AtlasGlobeProps) {
  return (
    <svg
      viewBox="0 0 100 100"
      className={cn("atlas-globe", spinning && "atlas-globe--spinning", className)}
      role={label ? "img" : undefined}
      aria-label={label}
      aria-hidden={label ? undefined : true}
      fill="none"
    >
      <g transform="rotate(-18 50 50)" stroke="currentColor" strokeWidth={1.1} vectorEffect="non-scaling-stroke">
        <circle cx={50} cy={50} r={R} />
        {LATITUDES.map((lat) => {
          const phi = (lat * Math.PI) / 180
          const rx = R * Math.cos(phi)
          return (
            <ellipse
              key={lat}
              cx={50}
              cy={50 - R * Math.sin(phi)}
              rx={rx}
              ry={rx * 0.16}
              opacity={lat === 0 ? 0.9 : 0.45}
            />
          )
        })}
        {Array.from({ length: MERIDIANS }, (_, i) => (
          <ellipse
            key={i}
            className="atlas-globe__meridian"
            cx={50}
            cy={50}
            rx={R}
            ry={R}
            opacity={0.55}
            style={{
              // Static pose: meridians fanned at 30 degree steps.
              ["--static-scale" as string]: Math.cos((i * Math.PI) / MERIDIANS).toFixed(3),
              // Spinning: same fan, as a phase offset into one turn.
              animationDelay: `${(-i / MERIDIANS) * 12}s`,
            }}
          />
        ))}
      </g>
      <ellipse
        className="atlas-globe__ecliptic"
        cx={50}
        cy={50}
        rx={R * 1.2}
        ry={R * 0.3}
        transform="rotate(8 50 50)"
        stroke="var(--primary)"
        strokeWidth={1.6}
        vectorEffect="non-scaling-stroke"
      />
    </svg>
  )
}
