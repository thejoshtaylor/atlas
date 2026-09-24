import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { Plugin } from "@/lib/plugins"
import {
  collidingToolCount,
  derivePluginsScreenState,
  formatCollisionSummary,
  formatToolCount,
  runtimeStatusDisplay,
  showDeleteControl,
  type QueryLike,
} from "./derivePluginsScreenState"

function pending(): QueryLike<Plugin[]> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored(error: unknown): QueryLike<Plugin[]> {
  return { status: "error", data: undefined, error }
}
function success(data: Plugin[]): QueryLike<Plugin[]> {
  return { status: "success", data, error: undefined }
}

function plugin(overrides: Partial<Plugin> = {}): Plugin {
  return {
    id: 1,
    slug: "home-assistant",
    display_name: "Home Assistant",
    transport: "stdio",
    args: ["-m", "atlas_mcp.ha"],
    url: null,
    enabled: true,
    builtin: true,
    timeout_ms: 5000,
    state: "running",
    reason: null,
    tools: [],
    config_values: [],
    applies_live: true,
    ...overrides,
  }
}

describe("derivePluginsScreenState", () => {
  test("pending renders loading -- the list must show skeleton rows, never an empty flash", () => {
    expect(derivePluginsScreenState(pending())).toEqual({ kind: "loading" })
  })

  test("a failed load carries the Copywriting Contract's exact fallback text when the server gives none more specific", () => {
    expect(derivePluginsScreenState(errored(new Error("network down")))).toEqual({
      kind: "error",
      message: "Couldn't load plugins. Try again.",
    })
  })

  test("a failed load with a named server reason surfaces it unmodified", () => {
    expect(derivePluginsScreenState(errored(new ApiError(500, "database unreachable")))).toEqual({
      kind: "error",
      message: "database unreachable",
    })
  })

  test("a successful load renders ready with every plugin -- including a ready state with zero rows, which the type permits even though it never happens in practice", () => {
    const plugins = [plugin({ id: 1 }), plugin({ id: 2, builtin: false })]
    expect(derivePluginsScreenState(success(plugins))).toEqual({ kind: "ready", plugins })
  })
})

describe("formatToolCount -- zero-one-many", () => {
  test("1 reads as singular", () => {
    expect(formatToolCount(1)).toBe("1 tool")
  })
  test("0 and >1 read as plural", () => {
    expect(formatToolCount(0)).toBe("0 tools")
    expect(formatToolCount(3)).toBe("3 tools")
  })
})

describe("collidingToolCount -- only a tool with a non-empty collides_with counts", () => {
  test("counts tools with at least one co-owner", () => {
    const p = plugin({
      tools: [
        { name: "forecast", description: "", collides_with: ["Weather"] },
        { name: "turn_on", description: "", collides_with: [] },
      ],
    })
    expect(collidingToolCount(p)).toBe(1)
  })

  test("a plugin with no colliding tools counts zero", () => {
    const p = plugin({ tools: [{ name: "turn_on", description: "", collides_with: [] }] })
    expect(collidingToolCount(p)).toBe(0)
  })
})

describe("formatCollisionSummary -- zero-one-many, verbatim Copywriting Contract text", () => {
  test("1 reads as singular", () => {
    expect(formatCollisionSummary(1)).toBe("1 tool shares a name with another plugin")
  })
  test("many reads as plural", () => {
    expect(formatCollisionSummary(3)).toBe("3 tools share names with other plugins")
  })
})

describe("runtimeStatusDisplay -- one badge and one caption per state, verbatim Copywriting Contract text", () => {
  test("running -- secondary badge, no caption", () => {
    expect(runtimeStatusDisplay({ state: "running", reason: null })).toEqual({
      badgeVariant: "secondary",
      badgeText: "Running",
      caption: null,
      showRetry: false,
    })
  })

  test("starting -- secondary badge, no caption", () => {
    expect(runtimeStatusDisplay({ state: "starting", reason: null })).toEqual({
      badgeVariant: "secondary",
      badgeText: "Starting…",
      caption: null,
      showRetry: false,
    })
  })

  test("disabled -- outline badge, no caption", () => {
    expect(runtimeStatusDisplay({ state: "disabled", reason: null })).toEqual({
      badgeVariant: "outline",
      badgeText: "Disabled",
      caption: null,
      showRetry: false,
    })
  })

  test("degraded -- denied badge, the server's own reason carried through unchanged, retry offered", () => {
    expect(runtimeStatusDisplay({ state: "degraded", reason: "connection refused at 192.168.1.50:8123" })).toEqual({
      badgeVariant: "denied",
      badgeText: "Degraded — won't start",
      caption: "connection refused at 192.168.1.50:8123",
      showRetry: true,
    })
  })

  test("crashed_retrying -- denied badge, fixed caption, no retry control (the supervisor already owns recovery)", () => {
    expect(runtimeStatusDisplay({ state: "crashed_retrying", reason: null })).toEqual({
      badgeVariant: "denied",
      badgeText: "Crashed — retrying",
      caption: "Restarting automatically.",
      showRetry: false,
    })
  })
})

describe("showDeleteControl -- absent, not disabled, for a builtin plugin", () => {
  test("a builtin plugin has no delete control", () => {
    expect(showDeleteControl({ builtin: true })).toBe(false)
  })
  test("a non-builtin plugin has one", () => {
    expect(showDeleteControl({ builtin: false })).toBe(true)
  })
})
