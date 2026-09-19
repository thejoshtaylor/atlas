import { create } from "zustand"
import type { CatalogEntry, ConfigValueInput, Plugin } from "@/lib/plugins"

// The plugin editor's unsaved draft -- the install-flow picker and its
// form (relevant only on `/plugins/new`), and the configuration
// key/value editor shared by both the install form and an existing
// plugin's "Save configuration" section. Built on `macroDraftStore.ts`'s
// shape (Task 1's own action text): the loaded plugin itself lives in a
// query cache, never here -- `loadPlugin`/`loadBlank` are the only two
// ways this store's fields are ever set from outside an admin's own
// typing.

export type InstallMode = "catalog" | "custom"
export type CustomTransport = "command" | "url"

export interface ConfigValueDraft {
  value: string
  secret: boolean
  /**
   * `false` only for an already-set secret key on an existing plugin,
   * before the admin taps "Update" -- the mask/"Set" branch renders while
   * this is `false` (`SettingsRoute.tsx`'s own per-field `editing` flag,
   * one instance per key here instead of per screen). A plain value or a
   * freshly added key is always `true`: there is nothing to mask, so the
   * field is always the live editable one.
   */
  editing: boolean
  /** The server's own `is_set` fact for this key, carried alongside the
   * draft so the mask branch can render "Set" without re-deriving it from
   * `value` (which the server never populates for a secret, set or
   * not). Meaningless for a plain key. */
  isSet: boolean
}

interface PluginDraftState {
  installMode: InstallMode
  selectedCatalogEntry: string | null
  displayName: string
  customTransport: CustomTransport
  command: string
  url: string
  timeoutMs: number

  configValues: Record<string, ConfigValueDraft>

  newKeyName: string
  newKeyValue: string
  newKeySecret: boolean

  setInstallMode: (mode: InstallMode) => void
  selectCatalogEntry: (entry: CatalogEntry) => void
  clearCatalogSelection: () => void
  setDisplayName: (name: string) => void
  setCustomTransport: (transport: CustomTransport) => void
  setCommand: (command: string) => void
  setUrl: (url: string) => void
  setTimeoutMs: (timeoutMs: number) => void

  setConfigValue: (key: string, value: string) => void
  startEditingSecret: (key: string) => void
  addConfigKey: (key: string, value: string, secret: boolean) => void

  setNewKeyName: (name: string) => void
  setNewKeyValue: (value: string) => void
  setNewKeySecret: (secret: boolean) => void

  loadPlugin: (plugin: Plugin) => void
  loadBlank: () => void
}

/** A secret value never arrives from the server (D-03) -- its draft
 * starts blank regardless of whether it is set, and `editing` is `true`
 * only when it is genuinely unset (there is no mask to show yet, so the
 * live field is the only option). A plain value's draft is initialized
 * to the server's own current value, always `editing: true` (there is
 * nothing to mask, per 06-UI-SPEC.md's Copywriting Contract). */
function configValuesFromPlugin(plugin: Plugin): Record<string, ConfigValueDraft> {
  const entries: Record<string, ConfigValueDraft> = {}
  for (const value of plugin.config_values) {
    entries[value.key] = {
      value: value.secret ? "" : (value.value ?? ""),
      secret: value.secret,
      editing: !value.secret || !value.is_set,
      isSet: value.is_set,
    }
  }
  return entries
}

const BLANK_DRAFT_FIELDS = {
  installMode: "catalog" as InstallMode,
  selectedCatalogEntry: null as string | null,
  displayName: "",
  customTransport: "command" as CustomTransport,
  command: "",
  url: "",
  timeoutMs: 5000,
  configValues: {} as Record<string, ConfigValueDraft>,
  newKeyName: "",
  newKeyValue: "",
  newKeySecret: false,
}

export const usePluginDraftStore = create<PluginDraftState>((set) => ({
  ...BLANK_DRAFT_FIELDS,

  setInstallMode: (installMode) => set({ installMode }),

  // Selecting a catalog entry replaces whatever configuration draft was
  // there before -- switching entries mid-pick must not leave a stale
  // key from the previously-selected entry behind (06-UI-SPEC.md: "a
  // link back to reopen it" implies a fresh form each time one is
  // chosen).
  selectCatalogEntry: (entry) =>
    set({
      selectedCatalogEntry: entry.name,
      displayName: entry.name,
      configValues: Object.fromEntries(
        entry.config_keys.map((configKey) => [
          configKey.key,
          { value: "", secret: configKey.secret, editing: true, isSet: false },
        ]),
      ),
    }),

  clearCatalogSelection: () => set({ selectedCatalogEntry: null, displayName: "", configValues: {} }),

  setDisplayName: (displayName) => set({ displayName }),
  setCustomTransport: (customTransport) => set({ customTransport }),
  setCommand: (command) => set({ command }),
  setUrl: (url) => set({ url }),
  setTimeoutMs: (timeoutMs) => set({ timeoutMs }),

  setConfigValue: (key, value) =>
    set((state) => ({
      configValues: { ...state.configValues, [key]: { ...state.configValues[key]!, value } },
    })),

  startEditingSecret: (key) =>
    set((state) => ({
      configValues: { ...state.configValues, [key]: { ...state.configValues[key]!, value: "", editing: true } },
    })),

  // A no-op when `key` already has a row -- adding a key this plugin
  // already declares would silently discard whatever draft edit was
  // already there.
  addConfigKey: (key, value, secret) =>
    set((state) => {
      if (state.configValues[key]) return state
      return {
        configValues: { ...state.configValues, [key]: { value, secret, editing: true, isSet: false } },
        newKeyName: "",
        newKeyValue: "",
        newKeySecret: false,
      }
    }),

  setNewKeyName: (newKeyName) => set({ newKeyName }),
  setNewKeyValue: (newKeyValue) => set({ newKeyValue }),
  setNewKeySecret: (newKeySecret) => set({ newKeySecret }),

  loadPlugin: (plugin) =>
    set({
      ...BLANK_DRAFT_FIELDS,
      displayName: plugin.display_name,
      timeoutMs: plugin.timeout_ms,
      configValues: configValuesFromPlugin(plugin),
    }),

  loadBlank: () => set({ ...BLANK_DRAFT_FIELDS }),
}))

/**
 * The draft's configuration values, shaped for a save/install request
 * body. An already-set secret whose mask is still showing (`editing:
 * false`) is omitted entirely -- sending it as a blank value would still
 * mean "leave it alone" server-side (`_prepare_config_value`'s own
 * contract), but omitting it here keeps the request body exactly the set
 * of keys the admin actually touched, the same discipline
 * `draftActionsToInput` uses for the macro editor's own save shape.
 */
export function draftConfigValuesToInput(
  configValues: Record<string, ConfigValueDraft>,
): Record<string, ConfigValueInput> {
  const input: Record<string, ConfigValueInput> = {}
  for (const [key, draft] of Object.entries(configValues)) {
    if (draft.secret && !draft.editing) continue
    input[key] = { value: draft.value, secret: draft.secret }
  }
  return input
}

/** UI-SPEC's "Add configuration key" affordance names no exact
 * blocked-save copy -- this project's own house style for a blocked
 * inline action (`ZERO_ACTIONS_SAVE_BLOCKED_REASON`'s own phrasing). */
export const BLANK_CONFIG_KEY_NAME_REASON = "Add a key name before adding it."

/** The draft store reports "Add key" as blocked when the key name is
 * blank -- a pure function over the field rather than a redundant
 * boolean, matching `saveBlockedByEmptyActions`'s own precedent. */
export function saveBlockedByBlankConfigKeyName(key: string): boolean {
  return key.trim() === ""
}
