import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "PluginsRoute.tsx"), "utf-8")

describe("PluginsRoute -- the list has no empty state at all (06-UI-SPEC.md's vocabulary lock)", () => {
  test("no EmptyState import or usage -- Home Assistant and weather are always-seeded builtin rows", () => {
    expect(SOURCE).not.toMatch(/EmptyState/)
  })

  test("plugins render directly from screen.plugins with no length-zero branch", () => {
    expect(SOURCE).toMatch(/screen\.plugins\.map/)
  })
})

describe("PluginsRoute -- loading never renders as an empty list", () => {
  test("SkeletonList renders for screen.kind === \"loading\"", () => {
    expect(SOURCE).toMatch(/screen\.kind === "loading" \? <SkeletonList/)
  })
})

describe("PluginsRoute -- a failed load disables the new-plugin action and every row control", () => {
  test("controlsDisabled derives from screen.kind !== \"ready\" and gates New plugin", () => {
    expect(SOURCE).toMatch(/const controlsDisabled = screen\.kind !== "ready"/)
    expect(SOURCE).toMatch(/<Button asChild disabled=\{controlsDisabled\}>/)
  })

  test("ErrorState renders with a retry for screen.kind === \"error\"", () => {
    expect(SOURCE).toMatch(/screen\.kind === "error" \? \(/)
    expect(SOURCE).toMatch(/onRetry=\{\(\) => void query\.refetch\(\)\}/)
  })

  test("row controls (enable/disable, delete) are disabled by the same list-unknown fact", () => {
    expect(SOURCE).toMatch(/const rowControlsDisabled = disabled \|\| setEnabled\.isPending \|\| remove\.isPending/)
  })
})

describe("PluginsRoute -- disable/enable act with no confirmation; delete always confirms", () => {
  test("the enable/disable button calls the mutation directly on click, no AlertDialog wraps it", () => {
    expect(SOURCE).toMatch(/setEnabled\.mutate\(\s*\{ pluginId: plugin\.id, enabled: !plugin\.enabled \}/)
  })

  // IN-02 (code review): both row writes used to have no error path at
  // all -- a 502 from the server's own "saved, but the running assistant
  // could not be updated" refusal produced an unhandled promise rejection
  // in the console and nothing on screen, so the admin saw the row simply
  // not change.
  test("a failed enable/disable names itself on the row", () => {
    expect(SOURCE).toMatch(/onError: \(\) => setRowError\(ROW_ENABLE_FAILED\)/)
  })

  test("a failed delete names itself on the row", () => {
    expect(SOURCE).toMatch(/onError: \(\) => setRowError\(ROW_DELETE_FAILED\)/)
  })

  test("the row renders whichever of the two failures happened", () => {
    expect(SOURCE).toMatch(/\{rowError \? <p className="text-body text-destructive">\{rowError\}<\/p> : null\}/)
  })

  test("delete is a three-part destructive confirmation with the contract's exact text", () => {
    expect(SOURCE).toMatch(
      /Delete "\$\{plugin\.display_name\}"\? Its tools will no longer reach the assistant, and any macro or workflow step that used one will be flagged as unresolved\./,
    )
    expect(SOURCE).toMatch(/variant="destructive"[\s\S]{0,500}Delete plugin/)
  })

  test("the button reads Disable when enabled, Enable when not", () => {
    expect(SOURCE).toMatch(/\{plugin\.enabled \? "Disable" : "Enable"\}/)
  })
})

describe("PluginsRoute -- a builtin row renders no delete control at all", () => {
  test("the delete control is gated by showDeleteControl(plugin), absent rather than disabled", () => {
    expect(SOURCE).toMatch(/\{showDeleteControl\(plugin\) \? \(/)
  })
})

describe("PluginsRoute -- every row shows its runtime badge, transport, builtin flag, and collision flag", () => {
  test("runtimeStatusDisplay drives the badge variant and text", () => {
    expect(SOURCE).toMatch(/const status = runtimeStatusDisplay\(plugin\)/)
    expect(SOURCE).toMatch(/<Badge variant=\{status\.badgeVariant\} className="shrink-0">\s*\{status\.badgeText\}/)
  })

  test("a builtin row shows the Built-in outline badge", () => {
    expect(SOURCE).toMatch(/\{plugin\.builtin \? <Badge variant="outline">Built-in<\/Badge> : null\}/)
  })

  test("a collision count shows the denied summary badge, only when collisions > 0", () => {
    expect(SOURCE).toMatch(/const collisions = collidingToolCount\(plugin\)/)
    expect(SOURCE).toMatch(/\{collisions > 0 \? <Badge variant="denied">\{formatCollisionSummary\(collisions\)\}<\/Badge> : null\}/)
  })
})

describe("PluginsRoute -- every interactive row meets the touch-target floor", () => {
  test("the enable/disable and delete buttons both carry touch-target", () => {
    const matches = SOURCE.match(/touch-target/g) ?? []
    expect(matches.length).toBeGreaterThanOrEqual(2)
  })
})

describe("PluginsRoute -- built from the enumerated inventory only, no data table", () => {
  test("cards are a hand-styled <li>, matching MacroRow's own precedent, not the Card primitive", () => {
    expect(SOURCE).toMatch(/<li className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">/)
  })

  test("no @/components/ui/table import", () => {
    expect(SOURCE).not.toMatch(/@\/components\/ui\/table/)
  })
})
