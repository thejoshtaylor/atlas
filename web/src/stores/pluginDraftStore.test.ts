import { beforeEach, describe, expect, test } from "bun:test"
import type { CatalogEntry, Plugin } from "@/lib/plugins"
import {
  BLANK_CONFIG_KEY_NAME_REASON,
  draftConfigValuesToInput,
  saveBlockedByBlankConfigKeyName,
  usePluginDraftStore,
  type ConfigValueDraft,
} from "./pluginDraftStore"

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

function catalogEntry(overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    name: "Home Assistant",
    description: "Control and query devices through a Home Assistant instance on the house network.",
    transport: "stdio",
    config_keys: [
      { key: "HA_URL", label: "Home Assistant URL", secret: false },
      { key: "HA_TOKEN", label: "Home Assistant long-lived access token", secret: true },
    ],
    ...overrides,
  }
}

beforeEach(() => {
  usePluginDraftStore.getState().loadBlank()
})

describe("loadBlank -- the new-plugin/install starting point", () => {
  test("resets every field, including a config draft left by a prior selection", () => {
    usePluginDraftStore.getState().selectCatalogEntry(catalogEntry())
    usePluginDraftStore.getState().loadBlank()
    const state = usePluginDraftStore.getState()
    expect(state.installMode).toBe("catalog")
    expect(state.selectedCatalogEntry).toBeNull()
    expect(state.displayName).toBe("")
    expect(state.configValues).toEqual({})
  })
})

describe("selectCatalogEntry -- pre-fills the configuration draft from the entry's own declared keys", () => {
  test("a plain key gets a blank editable draft, a secret key gets a blank draft with isSet: false", () => {
    usePluginDraftStore.getState().selectCatalogEntry(catalogEntry())
    const state = usePluginDraftStore.getState()
    expect(state.selectedCatalogEntry).toBe("Home Assistant")
    expect(state.displayName).toBe("Home Assistant")
    expect(state.configValues["HA_URL"]).toEqual({ value: "", secret: false, editing: true, isSet: false })
    expect(state.configValues["HA_TOKEN"]).toEqual({ value: "", secret: true, editing: true, isSet: false })
  })

  test("choosing a different entry replaces the prior draft, not merges with it", () => {
    usePluginDraftStore.getState().selectCatalogEntry(catalogEntry())
    usePluginDraftStore.getState().setConfigValue("HA_URL", "http://ha.local")
    usePluginDraftStore
      .getState()
      .selectCatalogEntry(
        catalogEntry({ name: "Weather", config_keys: [{ key: "WEATHER_LATITUDE", label: "Latitude", secret: false }] }),
      )
    const state = usePluginDraftStore.getState()
    expect(state.configValues["HA_URL"]).toBeUndefined()
    expect(state.configValues["WEATHER_LATITUDE"]).toEqual({ value: "", secret: false, editing: true, isSet: false })
  })
})

describe("clearCatalogSelection -- reopens the catalog list (06-UI-SPEC.md's 'Change selection' link)", () => {
  test("clears the selection, the name, and the configuration draft together", () => {
    usePluginDraftStore.getState().selectCatalogEntry(catalogEntry())
    usePluginDraftStore.getState().clearCatalogSelection()
    const state = usePluginDraftStore.getState()
    expect(state.selectedCatalogEntry).toBeNull()
    expect(state.displayName).toBe("")
    expect(state.configValues).toEqual({})
  })
})

describe("loadPlugin -- an existing plugin's configuration values become the draft's starting point", () => {
  test("a plain value shows its current value, always editing", () => {
    usePluginDraftStore.getState().loadPlugin(
      plugin({ config_values: [{ key: "HA_URL", secret: false, value: "http://ha.local", is_set: true }] }),
    )
    expect(usePluginDraftStore.getState().configValues["HA_URL"]).toEqual({
      value: "http://ha.local",
      secret: false,
      editing: true,
      isSet: true,
    })
  })

  test("an already-set secret starts blank and not editing -- the mask branch renders, never a decrypted value", () => {
    usePluginDraftStore
      .getState()
      .loadPlugin(plugin({ config_values: [{ key: "HA_TOKEN", secret: true, value: null, is_set: true }] }))
    expect(usePluginDraftStore.getState().configValues["HA_TOKEN"]).toEqual({
      value: "",
      secret: true,
      editing: false,
      isSet: true,
    })
  })

  test("a genuinely unset secret starts blank and editing -- there is no mask to show", () => {
    usePluginDraftStore
      .getState()
      .loadPlugin(plugin({ config_values: [{ key: "HA_TOKEN", secret: true, value: null, is_set: false }] }))
    expect(usePluginDraftStore.getState().configValues["HA_TOKEN"]).toEqual({
      value: "",
      secret: true,
      editing: true,
      isSet: false,
    })
  })

  test("loading a plugin also seeds the display name and timeout, and resets the install-flow fields", () => {
    usePluginDraftStore.getState().setInstallMode("custom")
    usePluginDraftStore.getState().loadPlugin(plugin({ display_name: "My HA", timeout_ms: 8000 }))
    const state = usePluginDraftStore.getState()
    expect(state.displayName).toBe("My HA")
    expect(state.timeoutMs).toBe(8000)
    expect(state.installMode).toBe("catalog")
  })
})

describe("startEditingSecret -- opens a blank input for an already-set secret, matching SettingsRoute.tsx's Update control", () => {
  test("flips editing to true and clears the value", () => {
    usePluginDraftStore
      .getState()
      .loadPlugin(plugin({ config_values: [{ key: "HA_TOKEN", secret: true, value: null, is_set: true }] }))
    usePluginDraftStore.getState().startEditingSecret("HA_TOKEN")
    expect(usePluginDraftStore.getState().configValues["HA_TOKEN"]).toEqual({
      value: "",
      secret: true,
      editing: true,
      isSet: true,
    })
  })
})

describe("addConfigKey -- the 'Add configuration key' affordance", () => {
  test("adds a new row and clears the add-key form", () => {
    usePluginDraftStore.getState().setNewKeyName("EXTRA_KEY")
    usePluginDraftStore.getState().setNewKeyValue("extra")
    usePluginDraftStore.getState().addConfigKey("EXTRA_KEY", "extra", false)
    const state = usePluginDraftStore.getState()
    expect(state.configValues["EXTRA_KEY"]).toEqual({ value: "extra", secret: false, editing: true, isSet: false })
    expect(state.newKeyName).toBe("")
    expect(state.newKeyValue).toBe("")
  })

  test("adding a key that already exists is a no-op -- it does not overwrite the existing draft", () => {
    usePluginDraftStore.getState().addConfigKey("EXTRA_KEY", "first", false)
    usePluginDraftStore.getState().addConfigKey("EXTRA_KEY", "second", true)
    expect(usePluginDraftStore.getState().configValues["EXTRA_KEY"]).toEqual({
      value: "first",
      secret: false,
      editing: true,
      isSet: false,
    })
  })
})

describe("draftConfigValuesToInput -- the save/install request shape", () => {
  test("a plain value and an actively-edited secret are both sent", () => {
    const draft: Record<string, ConfigValueDraft> = {
      HA_URL: { value: "http://ha.local", secret: false, editing: true, isSet: true },
      HA_TOKEN: { value: "new-token", secret: true, editing: true, isSet: false },
    }
    expect(draftConfigValuesToInput(draft)).toEqual({
      HA_URL: { value: "http://ha.local", secret: false },
      HA_TOKEN: { value: "new-token", secret: true },
    })
  })

  test("an already-set secret still showing its mask (editing: false) is omitted entirely", () => {
    const draft: Record<string, ConfigValueDraft> = {
      HA_TOKEN: { value: "", secret: true, editing: false, isSet: true },
    }
    expect(draftConfigValuesToInput(draft)).toEqual({})
  })
})

describe("saveBlockedByBlankConfigKeyName -- 'Add key' needs a non-blank key name", () => {
  test("blank or whitespace-only blocks", () => {
    expect(saveBlockedByBlankConfigKeyName("")).toBe(true)
    expect(saveBlockedByBlankConfigKeyName("   ")).toBe(true)
  })
  test("a real key name unblocks", () => {
    expect(saveBlockedByBlankConfigKeyName("EXTRA_KEY")).toBe(false)
  })
  test("the reason string is stable", () => {
    expect(BLANK_CONFIG_KEY_NAME_REASON).toBe("Add a key name before adding it.")
  })
})
