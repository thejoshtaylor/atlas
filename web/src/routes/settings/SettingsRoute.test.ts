import { afterEach, describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { createSubmitGuard } from "@/lib/submitGuard"
import { saveCredentialMutationOptions } from "@/lib/credentials"
import { queryClient } from "@/lib/queryClient"

// `web/` has no rendered-DOM test infrastructure (no Playwright, no
// @testing-library/*) -- this file asserts the source text carries the
// wiring 03/07-UI-SPEC.md require. It cannot prove the page actually
// renders correctly in a browser, the same limitation every prior
// phase's screen test in this repository carries (five phases now).
const SOURCE = readFileSync(join(import.meta.dir, "SettingsRoute.tsx"), "utf-8")

describe("SettingsRoute -- content is exactly what the server's listing returns", () => {
  test("credential sections are rendered from query.data.map, with no hardcoded slot allowlist filtering the results", () => {
    expect(SOURCE).toMatch(/query\.data\.map\(\(entry\) => <CredentialSection key=\{entry\.slot\} entry=\{entry\} \/>\)/)
    // No `.filter(` narrowing the listing down to a known set of slots.
    expect(SOURCE).not.toMatch(/query\.data\.filter\(/)
  })
})

describe("SettingsRoute -- badges come from the server's own classification field", () => {
  test("the badge text is entry.applies_live ? \"Live\" : \"Needs restart\", never a hardcoded value", () => {
    expect(SOURCE).toMatch(/\{entry\.applies_live \? "Live" : "Needs restart"\}/)
  })
  test("badges use the secondary variant -- neither the accent (default/primary) nor destructive colour", () => {
    const badgeSites = [...SOURCE.matchAll(/<Badge variant="(\w+)"/g)].map((m) => m[1])
    expect(badgeSites.length).toBeGreaterThan(0)
    expect(badgeSites).not.toContain("default")
    expect(badgeSites).not.toContain("destructive")
    expect(badgeSites.every((v) => v === "secondary")).toBe(true)
  })
})

describe("SettingsRoute -- an unset credential is a placeholdered input; a set one is a mask plus a date and an Update control", () => {
  test("unset/editing branch renders an Input with placeholder \"Not set\"", () => {
    expect(SOURCE).toMatch(/placeholder="Not set"/)
  })
  test("the set-and-not-editing branch renders the mask literal and an Update button", () => {
    expect(SOURCE).toMatch(/•{8} saved/)
    expect(SOURCE).toMatch(/onClick=\{\(\) => setEditing\(true\)\}[\s\S]{0,40}>\s*Update/)
  })
})

describe("SettingsRoute -- the update control opens a fresh, blank input, never pre-filled", () => {
  test("the input's value is bound to local `value` state, initialized to \"\", never to entry.value or any server-supplied credential content", () => {
    expect(SOURCE).toMatch(/useState\(""\)/)
    expect(SOURCE).toMatch(/value=\{value\}/)
    // The entry type structurally cannot carry a credential value at all
    // (CredentialEntry has no such field) -- so there is nothing here
    // that *could* pre-fill the input even by mistake.
    expect(SOURCE).not.toMatch(/entry\.value/)
  })
})

describe("SettingsRoute -- a failed save leaves the field in its prior state", () => {
  test("the catch branch resets editing to false and clears the draft value -- returning a previously-set field to its mask and a previously-unset field to empty", () => {
    const catchBlock = SOURCE.slice(SOURCE.indexOf("} catch {"), SOURCE.indexOf("Nothing was stored"))
    expect(catchBlock).toMatch(/setEditing\(false\)/)
    expect(catchBlock).toMatch(/setValue\(""\)/)
  })
})

describe("SettingsRoute -- loading never renders as an empty list", () => {
  test("SkeletonList renders while query.status === \"pending\"", () => {
    expect(SOURCE).toMatch(/query\.status === "pending" \? <SkeletonList/)
  })
})

describe("SettingsRoute -- the Providers row links out, in Safety policy's exact shape", () => {
  test("the row reads \"Providers\" and links to /providers", () => {
    expect(SOURCE).toMatch(/<p className="text-heading font-semibold text-foreground">Providers<\/p>/)
    expect(SOURCE).toMatch(/<Link to="\/providers"/)
  })

  test("the badge is present only when some slot needs a restart, absent otherwise -- never a count", () => {
    expect(SOURCE).toMatch(/providersNeedRestart \? <Badge variant="secondary">Needs restart<\/Badge> : null/)
    // Not a "{n} slots pending" count line -- nothing else in this
    // product counts pending things in a summary row (07-UI-SPEC.md).
    expect(SOURCE).not.toMatch(/slots? pending/)
  })

  test("the divergence fact is the shared needsRestart function, not a re-derived boolean", () => {
    expect(SOURCE).toMatch(/from "@\/routes\/providers\/deriveProvidersScreenState"/)
    expect(SOURCE).toMatch(/providersQuery\.data\?\.slots\.some\(\(slot\) => needsRestart\(slot\)\)/)
  })

  test("provider selection does not grow a credential section of its own -- the row only links out", () => {
    // No provider-specific Input/config-value editing on this screen --
    // a slot's provider and its credential are edited in two different
    // places (this file's own Providers row vs. CredentialSection).
    expect(SOURCE).not.toMatch(/provider_name/)
    expect(SOURCE).not.toMatch(/ProviderSlot/)
  })
})

const originalFetch = global.fetch
afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("SettingsRoute -- a second save cannot be issued from a pending control", () => {
  test("the shared re-entrancy guard allows exactly one underlying PUT for two back-to-back saves", async () => {
    let calls = 0
    let resolveFirst: (() => void) | undefined
    global.fetch = (async () => {
      calls += 1
      await new Promise<void>((resolve) => {
        resolveFirst = resolve
      })
      return new Response(
        JSON.stringify({
          slot: "stt",
          label: "Speech to text",
          is_set: true,
          updated_at: "2026-09-18T00:00:00Z",
          source: "database",
          applies_live: false,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    }) as typeof fetch

    const guard = createSubmitGuard(() =>
      saveCredentialMutationOptions.mutationFn!({ slot: "stt", value: "secret" }, {} as never),
    )

    const first = guard.run()
    const second = guard.run()
    expect(guard.isPending()).toBe(true)
    expect(calls).toBe(1)

    resolveFirst?.()
    await Promise.all([first, second])
    expect(calls).toBe(1)
  })
})
