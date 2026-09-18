/**
 * WEB-08's 44px minimum touch target (03-UI-SPEC.md's Spacing Scale
 * exception), verified against the real compiled Tailwind output rather
 * than against a jsdom/happy-dom render -- neither of those emulators
 * performs CSS layout (`getBoundingClientRect`/`offsetHeight` always
 * return 0 in both), so a "rendered" assertion against them would be
 * trivially true regardless of whether the real page is actually 44px.
 * This suite instead asserts against `web/dist`'s real generated CSS,
 * which Vite/Tailwind v4 produce identically whether a browser ever
 * opens the page or not -- the same reasoning the plan's own held-out
 * `<human-check>` step exists for: a real headless-browser layout
 * assertion needs Playwright (or an equivalent), a new dependency this
 * plan does not install without the package-legitimacy checkpoint every
 * prior batch in this project has gone through (see 03-03-SUMMARY.md's
 * Deviations section). The real-browser walk-through at 375px is the
 * held-out human verification step this suite does not attempt to
 * replace.
 */
import { afterAll, beforeAll, describe, expect, test } from "bun:test"
import { spawnSync } from "node:child_process"
import { readFileSync, readdirSync } from "node:fs"
import { join } from "node:path"

const VIEWPORT_WIDTH_PX = 375

const WEB_ROOT = join(import.meta.dir)
const DIST_ASSETS_DIR = join(WEB_ROOT, "..", "dist", "assets")

let builtCss = ""

beforeAll(() => {
  const result = spawnSync("bun", ["run", "build"], {
    cwd: join(WEB_ROOT, ".."),
    encoding: "utf-8",
  })
  if (result.status !== 0) {
    throw new Error(`bun run build failed:\n${result.stdout}\n${result.stderr}`)
  }
  const cssFile = readdirSync(DIST_ASSETS_DIR).find((name) => name.endsWith(".css"))
  if (!cssFile) throw new Error(`no built CSS file found under ${DIST_ASSETS_DIR}`)
  builtCss = readFileSync(join(DIST_ASSETS_DIR, cssFile), "utf-8")
})

afterAll(() => {
  builtCss = ""
})

function pxValue(remExpression: string): number {
  // Tailwind v4 emits sizes as `calc(var(--spacing) * N)` against a
  // `--spacing: .25rem` root token (both asserted below), or as a
  // literal `Xrem` for hand-authored custom properties (e.g.
  // `.touch-target`). Either resolves to pixels at the browser's default
  // 16px root font size, which this project never overrides.
  const spacingMultiple = remExpression.match(/calc\(var\(--spacing\) \* (\d+(?:\.\d+)?)\)/)
  if (spacingMultiple) return Number(spacingMultiple[1]) * 0.25 * 16
  const rem = remExpression.match(/^(\d*\.?\d+)rem$/)
  if (rem) return Number(rem[1]) * 16
  throw new Error(`unrecognized size expression: ${remExpression}`)
}

describe("44px minimum touch target (WEB-08, verified at the compiled-CSS level)", () => {
  test("--spacing resolves to 4px, the scale every h-*/size-* class below is measured against", () => {
    const match = builtCss.match(/--spacing:\s*([^;]+);/)
    expect(match).not.toBeNull()
    expect(pxValue(match![1].trim())).toBe(4)
  })

  test.each([
    ["h-11", "the Button/Input default height (SubmitButton, Input, SignInRoute)"],
    ["size-11", "the AppShell header's icon-only nav-toggle button"],
    ["touch-target", "the reusable class every AppShell nav item composes"],
  ])("%s (%s) resolves to at least 44px", (className) => {
    const rule = builtCss.match(new RegExp(`\\.${className}\\{([^}]*)\\}`))
    expect(rule).not.toBeNull()

    const heightDeclaration = rule![1].match(/(?:^|;)(?:height|min-height):([^;]+)/)
    expect(heightDeclaration).not.toBeNull()
    expect(pxValue(heightDeclaration![1].trim())).toBeGreaterThanOrEqual(44)
  })
})

describe("no fixed width in the shell's own CSS exceeds the 375px viewport (WEB-08, D-17)", () => {
  // A genuine "render at 375px and read the resulting scrollWidth" check
  // needs a real browser layout engine (Playwright or equivalent) --
  // neither happy-dom nor jsdom compute layout, so that assertion cannot
  // be made honestly without a new dependency this plan does not install
  // without the package-legitimacy checkpoint every prior batch in this
  // project went through (03-03-SUMMARY.md's Deviations section flags
  // this as a follow-up). This check instead statically measures every
  // fixed-pixel width this project's own CSS declares -- the one way a
  // *static* declaration (as opposed to unpredictable runtime content
  // such as a long entity name) could force horizontal overflow -- and
  // is a real, if partial, measured assertion rather than a tautology.
  // The held-out `<human-check>` step in this task's own `<verify>`
  // block is the authoritative check for content-driven overflow.
  test("no `width`/`min-width` declaration in the compiled CSS exceeds 375px", () => {
    const widthDeclarations = [...builtCss.matchAll(/(?:^|[;{])(?:width|min-width):([^;}]+)[;}]/g)]
    expect(widthDeclarations.length).toBeGreaterThan(0)

    const overflowing = widthDeclarations
      .map((match) => match[1].trim())
      .filter((value) => /^\d*\.?\d+px$/.test(value))
      .filter((value) => Number.parseFloat(value) > VIEWPORT_WIDTH_PX)

    expect(overflowing).toEqual([])
  })
})
