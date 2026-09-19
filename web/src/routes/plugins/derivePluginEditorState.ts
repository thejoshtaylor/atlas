// PluginEditorRoute's own logic, kept apart from its JSX -- copied in
// shape from `routes/macros/deriveMacroEditorState.ts`'s `QueryLike`
// interface and discriminated union. Imports nothing from React, for the
// same stated reason.
import { ApiError } from "@/lib/api"
import type { CatalogEntry, Plugin } from "@/lib/plugins"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

// `plugin: null` is the "new, unsaved plugin" ready state -- there is no
// id to fetch yet, so `PluginEditorRoute` never runs a query for it at
// all (see `query: null` below), the exact shape
// `deriveMacroEditorState.ts` already established.
export type PluginEditorScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; plugin: Plugin | null }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load this plugin. It may have been removed."
}

/**
 * `query` is `null` for a new plugin (`/plugins/new`, no id to fetch) --
 * that route never contacts `/api/plugins/{id}` at all, so it is always
 * ready with `plugin: null` rather than a query that stays pending
 * forever.
 */
export function derivePluginEditorState(query: QueryLike<Plugin> | null): PluginEditorScreenState {
  if (query === null) return { kind: "ready", plugin: null }
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  if (!query.data) return { kind: "loading" }
  return { kind: "ready", plugin: query.data }
}

// The catalog pane, unlike the plugins list itself, has a genuine empty
// state -- a catalog file with zero entries is a real, possible state
// (`plugins/catalog.py`'s own docstring), not a hypothetical one.
export type CatalogPaneState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "empty" }
  | { kind: "ready"; entries: CatalogEntry[] }

function catalogMessageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load the plugin catalog. Try again."
}

export function deriveCatalogPaneState(query: QueryLike<CatalogEntry[]>): CatalogPaneState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: catalogMessageFor(query.error) }
  const entries = query.data
  if (!entries) return { kind: "loading" }
  if (entries.length === 0) return { kind: "empty" }
  return { kind: "ready", entries }
}

/** "1 key" / "{n} keys" -- the configuration section's own heading
 * caption, the same zero-one-many rule every count line in this product
 * follows. */
export function formatConfigKeyCount(count: number): string {
  return `${count} ${count === 1 ? "key" : "keys"}`
}

/** Install is blocked, with a named reason, until the admin has picked a
 * catalog entry (catalog mode) -- one of the two "neither a command nor a
 * URL" shapes Task 1's own behaviour names, the catalog-mode half. */
export const NO_CATALOG_SELECTION_REASON = "Select a plugin from the catalog before installing."

export function saveBlockedByNoCatalogSelection(selectedCatalogEntry: string | null): boolean {
  return selectedCatalogEntry === null
}

export const MISSING_COMMAND_REASON = "Add a command before installing."
export const MISSING_URL_REASON = "Add a server URL before installing."

/** The custom-mode half of the same rule: a hand-entered install needs
 * whichever of command/URL its own transport picker selected. */
export function saveBlockedByMissingCustomSource(
  transport: "command" | "url",
  command: string,
  url: string,
): string | null {
  if (transport === "command") return command.trim() === "" ? MISSING_COMMAND_REASON : null
  return url.trim() === "" ? MISSING_URL_REASON : null
}

export const BLANK_DISPLAY_NAME_REASON = "Add a name before installing."

/** `install_plugin`'s own `_missing_display_name_error` (a hand-added
 * plugin has no catalog entry to fall back a name from) -- checked
 * client-side first for the same advisory reason
 * `saveBlockedByBlankPhrase` is: the route refuses it regardless. */
export function saveBlockedByBlankDisplayName(displayName: string): boolean {
  return displayName.trim() === ""
}

/**
 * "Shares this name with {other plugin's display name}" -- 06-UI-SPEC.md's
 * Copywriting Contract, verbatim, for the per-tool row in the plugin
 * editor (the list's own summary badge is `formatCollisionSummary`
 * instead). `null` for an uncontested tool -- the caller is responsible
 * for that guard.
 */
export function toolCollisionText(collidesWith: string[]): string | null {
  if (collidesWith.length === 0) return null
  return `Shares this name with ${collidesWith.join(", ")}`
}
