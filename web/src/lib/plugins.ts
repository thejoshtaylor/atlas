// The plugins admin surface's fetch layer (PLUG-03, PLUG-08, D-14). Every
// shape here matches `src/spire_voice/routes/plugins.py`'s own response
// models field for field, the same discipline `lib/macros.ts` states at
// its own top of file -- a renamed or reshaped field here is a silent
// drift from what the server actually sends.
import type { UseMutationOptions } from "@tanstack/react-query"
import { apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `PluginState`'s exact five values (`plugins/manager.py`). `starting` is
 * transient; `running`/`degraded` are the two boot-time outcomes (D-07);
 * `crashed_retrying` is a plugin the supervisor is currently respawning
 * (D-05); `disabled` is a row nobody has attempted to start. */
export type PluginState = "starting" | "running" | "degraded" | "crashed_retrying" | "disabled"

/** `CatalogConfigKeyResponse`'s exact shape. */
export interface CatalogConfigKey {
  key: string
  label: string
  secret: boolean
}

/** `CatalogEntryResponse`'s exact shape. No icon/logo field -- confirmed,
 * not contradicted, by 06-06-SUMMARY.md's own key-decisions. */
export interface CatalogEntry {
  name: string
  description: string
  transport: string
  config_keys: CatalogConfigKey[]
}

/** `PluginToolResponse`'s exact shape. `collides_with` is `[]` for an
 * uncontested tool, never omitted (D-09, D-10). */
export interface PluginTool {
  name: string
  description: string
  collides_with: string[]
}

/** `PluginConfigValueResponse`'s exact shape. `value` is always `null`
 * for a secret key (D-03, T-06-27) -- this module never receives a
 * decrypted secret to accidentally render. */
export interface PluginConfigValue {
  key: string
  secret: boolean
  value: string | null
  is_set: boolean
}

/** `PluginResponse`'s exact shape. */
export interface Plugin {
  id: number
  slug: string
  display_name: string
  transport: string
  args: string[]
  url: string | null
  enabled: boolean
  builtin: boolean
  timeout_ms: number
  state: PluginState
  reason: string | null
  tools: PluginTool[]
  config_values: PluginConfigValue[]
  applies_live: boolean
}

export const PLUGINS_QUERY_KEY = ["plugins"] as const

export function pluginQueryKey(pluginId: number) {
  return ["plugins", pluginId] as const
}

export const PLUGIN_CATALOG_QUERY_KEY = ["plugins", "catalog"] as const

export function fetchPlugins(): Promise<Plugin[]> {
  return apiFetch<Plugin[]>("/api/plugins")
}

export function fetchPlugin(pluginId: number): Promise<Plugin> {
  return apiFetch<Plugin>(`/api/plugins/${pluginId}`)
}

export function fetchPluginCatalog(): Promise<CatalogEntry[]> {
  return apiFetch<CatalogEntry[]>("/api/plugins/catalog")
}

/** `ConfigValueInput`'s exact shape -- what an install or config-save
 * request sends per key. A blank `value` on an already-set secret key
 * means "leave it alone" server-side (`_prepare_config_value`'s own
 * contract); this module does not enforce that here, the draft store
 * does (`pluginDraftStore.ts::draftConfigValuesToInput`). */
export interface ConfigValueInput {
  value: string
  secret: boolean
}

/** `InstallPluginRequest`'s exact shape. Exactly one of `catalog_entry`
 * or (`transport` + `command`/`url`) is given -- the server enforces the
 * exclusivity (`_resolve_install_source`), this module only carries the
 * shape. */
export interface InstallPluginInput {
  catalog_entry?: string | null
  display_name?: string | null
  transport?: "command" | "url" | null
  command?: string | null
  url?: string | null
  timeout_ms: number
  config_values: Record<string, ConfigValueInput>
}

/**
 * No optimistic update, same reasoning `macros.ts` records for its own
 * write mutations: an install is a real side effect on a running process
 * that can fail (D-15's own live-reconcile-before-return rule), so a row
 * shown before the server confirmed it would be a row the admin believes
 * is running when it is not. `onSuccess` seeds the cache from the
 * confirmed response.
 */
export const installPluginMutationOptions: UseMutationOptions<Plugin, unknown, InstallPluginInput> = {
  mutationFn: (input) => apiFetch<Plugin>("/api/plugins", { method: "POST", body: input }),
  onSuccess: (plugin) => {
    void queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEY })
    queryClient.setQueryData(pluginQueryKey(plugin.id), plugin)
  },
}

export interface SetPluginEnabledInput {
  pluginId: number
  enabled: boolean
}

export const setPluginEnabledMutationOptions: UseMutationOptions<Plugin, unknown, SetPluginEnabledInput> = {
  mutationFn: ({ pluginId, enabled }) =>
    apiFetch<Plugin>(`/api/plugins/${pluginId}/enabled`, { method: "PUT", body: { enabled } }),
  onSuccess: (plugin) => {
    void queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEY })
    queryClient.setQueryData(pluginQueryKey(plugin.id), plugin)
  },
}

export interface SavePluginConfigInput {
  pluginId: number
  values: Record<string, ConfigValueInput>
  // IN-04 (code review): D-06 calls the per-plugin deadline
  // admin-editable, and until this field existed no route could change it
  // after install -- the editor loaded `timeout_ms` into its draft and had
  // nowhere to send it. Optional: omitted means "leave it alone".
  timeout_ms?: number
}

export const savePluginConfigMutationOptions: UseMutationOptions<Plugin, unknown, SavePluginConfigInput> = {
  mutationFn: ({ pluginId, values, timeout_ms }) =>
    apiFetch<Plugin>(`/api/plugins/${pluginId}/config`, {
      method: "PUT",
      body: timeout_ms === undefined ? { values } : { values, timeout_ms },
    }),
  onSuccess: (plugin) => {
    void queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEY })
    queryClient.setQueryData(pluginQueryKey(plugin.id), plugin)
  },
}

export interface DeletePluginInput {
  pluginId: number
}

export const deletePluginMutationOptions: UseMutationOptions<void, unknown, DeletePluginInput> = {
  mutationFn: ({ pluginId }) => apiFetch<void>(`/api/plugins/${pluginId}`, { method: "DELETE" }),
  onSuccess: (_data, { pluginId }) => {
    void queryClient.invalidateQueries({ queryKey: PLUGINS_QUERY_KEY })
    queryClient.removeQueries({ queryKey: pluginQueryKey(pluginId) })
  },
}
