// PluginsRoute's own logic, kept apart from its JSX -- copied in shape
// from `routes/macros/deriveMacrosScreenState.ts`. Imports nothing from
// React, for the same stated reason: what belongs here is only ever a
// fact about data, never a fact about a render.
import { ApiError } from "@/lib/api"
import type { Plugin } from "@/lib/plugins"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

// There is no "empty" member on purpose (06-UI-SPEC.md's vocabulary
// lock): Home Assistant and weather are always-seeded `builtin` rows, so
// a `ready` state with zero plugins cannot occur on a real deployment.
// Adding an empty variant here would be code for a state that never
// happens -- `PluginsRoute.tsx` renders `screen.plugins` directly with no
// `EmptyState` branch at all.
export type PluginsScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "ready"; plugins: Plugin[] }

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load plugins. Try again."
}

export function derivePluginsScreenState(query: QueryLike<Plugin[]>): PluginsScreenState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  const plugins = query.data
  if (!plugins) return { kind: "loading" }
  return { kind: "ready", plugins }
}

/** "1 tool" / "{n} tools" -- the same zero-one-many rule every count line
 * in this product follows (`formatActionCount`'s own precedent). Shared
 * by the list row and the editor's tools-section heading. */
export function formatToolCount(count: number): string {
  return `${count} ${count === 1 ? "tool" : "tools"}`
}

/** How many of a plugin's own tools currently share a bare name with
 * another plugin (D-09) -- a tool's own `collides_with` is non-empty
 * exactly when it does. */
export function collidingToolCount(plugin: Pick<Plugin, "tools">): number {
  return plugin.tools.filter((tool) => tool.collides_with.length > 0).length
}

/** "1 tool shares a name with another plugin" / "{n} tools share names
 * with other plugins" -- 06-UI-SPEC.md's Copywriting Contract, verbatim.
 * Only ever rendered when `collidingToolCount(plugin) > 0`, matching
 * `formatConflictSummary`'s own never-called-for-zero precedent. */
export function formatCollisionSummary(count: number): string {
  return count === 1 ? "1 tool shares a name with another plugin" : `${count} tools share names with other plugins`
}

export interface RuntimeStatusDisplay {
  badgeVariant: "secondary" | "outline" | "denied"
  badgeText: string
  /** The server's own verbatim degraded reason, or a fixed caption for
   * `crashed_retrying`, or `null` for a state with nothing to say beneath
   * the badge. Never paraphrased (the CMD-07/VOICE-02 lesson this
   * project keeps relearning, cited directly in 06-UI-SPEC.md). */
  caption: string | null
  /** "Retry now" only for a plugin the supervisor is not already
   * retrying on its own (D-05) -- `crashed_retrying` never shows it. */
  showRetry: boolean
}

/**
 * A plugin's runtime status maps to exactly one badge and one caption
 * (Task 1's own behaviour) -- 06-UI-SPEC.md's Copywriting Contract,
 * verbatim, for every one of `PluginState`'s five values.
 */
export function runtimeStatusDisplay(plugin: Pick<Plugin, "state" | "reason">): RuntimeStatusDisplay {
  switch (plugin.state) {
    case "running":
      return { badgeVariant: "secondary", badgeText: "Running", caption: null, showRetry: false }
    case "starting":
      return { badgeVariant: "secondary", badgeText: "Starting…", caption: null, showRetry: false }
    case "disabled":
      return { badgeVariant: "outline", badgeText: "Disabled", caption: null, showRetry: false }
    case "degraded":
      return {
        badgeVariant: "denied",
        badgeText: "Degraded — won't start",
        caption: plugin.reason ?? "",
        showRetry: true,
      }
    case "crashed_retrying":
      return {
        badgeVariant: "denied",
        badgeText: "Crashed — retrying",
        caption: "Restarting automatically.",
        showRetry: false,
      }
  }
}

/** A `builtin` plugin renders no Delete control at all -- absent, never
 * disabled (D-04, 06-UI-SPEC.md's vocabulary lock). */
export function showDeleteControl(plugin: Pick<Plugin, "builtin">): boolean {
  return !plugin.builtin
}
