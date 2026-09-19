import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { CatalogEntry, Plugin } from "@/lib/plugins"
import {
  BLANK_DISPLAY_NAME_REASON,
  MISSING_COMMAND_REASON,
  MISSING_URL_REASON,
  NO_CATALOG_SELECTION_REASON,
  deriveCatalogPaneState,
  derivePluginEditorState,
  formatConfigKeyCount,
  remoteConfigCaption,
  REMOTE_PLAIN_KEYS_UNUSED,
  saveBlockedByBlankDisplayName,
  saveBlockedByMissingCustomSource,
  saveBlockedByNoCatalogSelection,
  toolCollisionText,
  type QueryLike,
} from "./derivePluginEditorState"

function pending<T>(): QueryLike<T> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored<T>(error: unknown): QueryLike<T> {
  return { status: "error", data: undefined, error }
}
function success<T>(data: T): QueryLike<T> {
  return { status: "success", data, error: undefined }
}

function plugin(overrides: Partial<Plugin> = {}): Plugin {
  return {
    id: 1,
    slug: "home-assistant",
    display_name: "Home Assistant",
    transport: "stdio",
    args: ["-m", "spire_mcp.ha"],
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

describe("derivePluginEditorState", () => {
  test("null query (new plugin, /plugins/new) is always ready with plugin: null", () => {
    expect(derivePluginEditorState(null)).toEqual({ kind: "ready", plugin: null })
  })

  test("pending renders loading", () => {
    expect(derivePluginEditorState(pending<Plugin>())).toEqual({ kind: "loading" })
  })

  test("a failed load carries the Copywriting Contract's exact fallback text when the server gives none more specific", () => {
    expect(derivePluginEditorState(errored<Plugin>(new Error("network down")))).toEqual({
      kind: "error",
      message: "Couldn't load this plugin. It may have been removed.",
    })
  })

  test("a failed load with a named server reason surfaces it unmodified", () => {
    expect(derivePluginEditorState(errored<Plugin>(new ApiError(404, "no plugin with id 5")))).toEqual({
      kind: "error",
      message: "no plugin with id 5",
    })
  })

  test("a successful load renders ready with the plugin", () => {
    const p = plugin()
    expect(derivePluginEditorState(success<Plugin>(p))).toEqual({ kind: "ready", plugin: p })
  })
})

function catalogEntry(overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    name: "Home Assistant",
    description: "Control and query devices through a Home Assistant instance on the house network.",
    transport: "stdio",
    config_keys: [],
    ...overrides,
  }
}

describe("deriveCatalogPaneState -- the catalog pane, unlike the plugins list, has a genuine empty state", () => {
  test("pending renders loading", () => {
    expect(deriveCatalogPaneState(pending<CatalogEntry[]>())).toEqual({ kind: "loading" })
  })

  test("an error renders with a fallback message", () => {
    expect(deriveCatalogPaneState(errored<CatalogEntry[]>(new Error("boom")))).toEqual({
      kind: "error",
      message: "Couldn't load the plugin catalog. Try again.",
    })
  })

  test("a zero-entry catalog file renders empty, distinct from loading or error", () => {
    expect(deriveCatalogPaneState(success<CatalogEntry[]>([]))).toEqual({ kind: "empty" })
  })

  test("a populated catalog renders ready with every entry", () => {
    const entries = [catalogEntry(), catalogEntry({ name: "Weather" })]
    expect(deriveCatalogPaneState(success<CatalogEntry[]>(entries))).toEqual({ kind: "ready", entries })
  })
})

describe("formatConfigKeyCount -- zero-one-many", () => {
  test("1 reads as singular", () => {
    expect(formatConfigKeyCount(1)).toBe("1 key")
  })
  test("0 and >1 read as plural", () => {
    expect(formatConfigKeyCount(0)).toBe("0 keys")
    expect(formatConfigKeyCount(3)).toBe("3 keys")
  })
})

describe("saveBlockedByNoCatalogSelection -- catalog-mode install needs an entry chosen", () => {
  test("no selection blocks save with the named reason", () => {
    expect(saveBlockedByNoCatalogSelection(null)).toBe(true)
  })
  test("a selection unblocks it", () => {
    expect(saveBlockedByNoCatalogSelection("Home Assistant")).toBe(false)
  })
  test("the reason string is stable", () => {
    expect(NO_CATALOG_SELECTION_REASON).toBe("Select a plugin from the catalog before installing.")
  })
})

describe("saveBlockedByMissingCustomSource -- a hand-entered install needs whichever of command/URL its transport picked", () => {
  test("command mode blocks on a blank command", () => {
    expect(saveBlockedByMissingCustomSource("command", "", "")).toBe(MISSING_COMMAND_REASON)
    expect(saveBlockedByMissingCustomSource("command", "  ", "")).toBe(MISSING_COMMAND_REASON)
  })
  test("command mode is unblocked once a command is typed", () => {
    expect(saveBlockedByMissingCustomSource("command", "-m spire_mcp.custom", "")).toBeNull()
  })
  test("url mode blocks on a blank URL, ignoring a stray command value", () => {
    expect(saveBlockedByMissingCustomSource("url", "-m ignored", "")).toBe(MISSING_URL_REASON)
  })
  test("url mode is unblocked once a URL is typed", () => {
    expect(saveBlockedByMissingCustomSource("url", "", "https://example.local/mcp")).toBeNull()
  })
})

describe("saveBlockedByBlankDisplayName -- a hand-added plugin needs a name (install_plugin's own refusal)", () => {
  test("blank or whitespace-only blocks", () => {
    expect(saveBlockedByBlankDisplayName("")).toBe(true)
    expect(saveBlockedByBlankDisplayName("   ")).toBe(true)
  })
  test("a real name unblocks", () => {
    expect(saveBlockedByBlankDisplayName("My Plugin")).toBe(false)
  })
  test("the reason string is stable", () => {
    expect(BLANK_DISPLAY_NAME_REASON).toBe("Add a name before installing.")
  })
})

describe("toolCollisionText -- per-tool collision display in the editor", () => {
  test("an uncontested tool has no collision text", () => {
    expect(toolCollisionText([])).toBeNull()
  })
  test("names the other owning plugin", () => {
    expect(toolCollisionText(["Weather"])).toBe("Shares this name with Weather")
  })
  test("joins more than one co-owner", () => {
    expect(toolCollisionText(["Weather", "Custom Forecast"])).toBe("Shares this name with Weather, Custom Forecast")
  })
})

describe("remoteConfigCaption -- WR-07: a URL plugin's plain values are stored and unused", () => {
  test("a URL plugin says so, in the editor, where the keys are entered", () => {
    expect(remoteConfigCaption("remote")).toBe(REMOTE_PLAIN_KEYS_UNUSED)
    expect(remoteConfigCaption("remote")).toMatch(/bearer token/)
  })

  test("a stdio plugin says nothing -- every key it carries reaches its child", () => {
    expect(remoteConfigCaption("stdio")).toBeNull()
  })
})
