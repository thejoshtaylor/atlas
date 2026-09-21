/**
 * Every `derive*ScreenState.ts` module in this project opens with the same
 * claim: "Imports nothing from React: what belongs here is only ever a
 * fact about data, never a fact about a render." Until this file, nothing
 * checked it -- and `routes/sessions/deriveSessionsScreenState.ts` had
 * been importing `formatMs` from `routes/dev-mic/DevMicRoute.tsx` since
 * Phase 8 shipped (WR-06 review), which put React, the transports module
 * and the audio-worklet plumbing in the Sessions screen's own module graph.
 *
 * A source-text assertion, deliberately: what is being checked is a fact
 * about the module *graph*, which a render cannot observe. Mounting the
 * component proves the screen works; it does not prove which files came
 * along.
 */
import { describe, expect, test } from "bun:test"
import { existsSync, readFileSync, readdirSync } from "node:fs"
import { join } from "node:path"

const SRC = join(import.meta.dir)

function deriveModules(directory: string): string[] {
  const found: string[] = []
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = join(directory, entry.name)
    if (entry.isDirectory()) {
      found.push(...deriveModules(path))
    } else if (/^derive.*\.ts$/.test(entry.name) && !entry.name.endsWith(".test.ts")) {
      found.push(path)
    }
  }
  return found
}

function importSpecifiers(source: string): string[] {
  return [...source.matchAll(/^import[^"']*["']([^"']+)["']/gm)].map((match) => match[1]!)
}

describe("derive*ScreenState modules import nothing from React and nothing from a route", () => {
  const modules = deriveModules(SRC)

  test("there are derive modules to check at all", () => {
    expect(modules.length).toBeGreaterThan(3)
  })

  for (const path of modules) {
    test(`${path.slice(SRC.length + 1)} keeps its own stated rule`, () => {
      const specifiers = importSpecifiers(readFileSync(path, "utf-8"))
      expect(specifiers.filter((specifier) => specifier === "react" || specifier.startsWith("react/"))).toEqual([])
      // A route *component* is a render, and its graph is a render's
      // graph. Another pure module under `routes/` is fine -- what is
      // forbidden is reaching into a `.tsx`. Share a helper through
      // `lib/` instead; that is what `lib/` is for.
      const componentImports = specifiers.filter(
        (specifier) =>
          specifier.startsWith("@/") && existsSync(join(SRC, `${specifier.slice(2)}.tsx`)),
      )
      expect(componentImports).toEqual([])
    })
  }
})
