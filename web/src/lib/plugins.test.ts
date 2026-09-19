import { afterEach, describe, expect, test } from "bun:test"
import { queryClient } from "./queryClient"
import {
  PLUGIN_CATALOG_QUERY_KEY,
  PLUGINS_QUERY_KEY,
  deletePluginMutationOptions,
  fetchPlugin,
  fetchPluginCatalog,
  fetchPlugins,
  installPluginMutationOptions,
  pluginQueryKey,
  savePluginConfigMutationOptions,
  setPluginEnabledMutationOptions,
  type CatalogEntry,
  type Plugin,
} from "./plugins"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

function samplePlugin(overrides: Partial<Plugin> = {}): Plugin {
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

function sampleCatalogEntry(overrides: Partial<CatalogEntry> = {}): CatalogEntry {
  return {
    name: "Home Assistant",
    description: "Control and query devices through a Home Assistant instance on the house network.",
    transport: "stdio",
    config_keys: [{ key: "HA_URL", label: "Home Assistant URL", secret: false }],
    ...overrides,
  }
}

describe("plugins.ts -- calls the real routes/plugins.py paths", () => {
  test("fetchPlugins calls GET /api/plugins", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [samplePlugin()])
    }) as typeof fetch
    await fetchPlugins()
    expect(calledUrl).toBe("/api/plugins")
  })

  test("fetchPlugin calls GET /api/plugins/{id}", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, samplePlugin({ id: 5 }))
    }) as typeof fetch
    await fetchPlugin(5)
    expect(calledUrl).toBe("/api/plugins/5")
  })

  test("fetchPluginCatalog calls GET /api/plugins/catalog", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [sampleCatalogEntry()])
    }) as typeof fetch
    await fetchPluginCatalog()
    expect(calledUrl).toBe("/api/plugins/catalog")
  })

  test("installPluginMutationOptions POSTs the install shape to /api/plugins", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(201, samplePlugin())
    }) as typeof fetch
    await installPluginMutationOptions.mutationFn!(
      { catalog_entry: "Home Assistant", timeout_ms: 5000, config_values: {} },
      {} as never,
    )
    expect(calledUrl).toBe("/api/plugins")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ catalog_entry: "Home Assistant", timeout_ms: 5000, config_values: {} })
  })

  test("setPluginEnabledMutationOptions PUTs to /api/plugins/{id}/enabled", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, samplePlugin({ id: 3, enabled: false }))
    }) as typeof fetch
    await setPluginEnabledMutationOptions.mutationFn!({ pluginId: 3, enabled: false }, {} as never)
    expect(calledUrl).toBe("/api/plugins/3/enabled")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({ enabled: false })
  })

  test("savePluginConfigMutationOptions PUTs to /api/plugins/{id}/config", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, samplePlugin({ id: 4 }))
    }) as typeof fetch
    await savePluginConfigMutationOptions.mutationFn!(
      { pluginId: 4, values: { HA_URL: { value: "http://ha.local", secret: false } } },
      {} as never,
    )
    expect(calledUrl).toBe("/api/plugins/4/config")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({ values: { HA_URL: { value: "http://ha.local", secret: false } } })
  })

  test("deletePluginMutationOptions DELETEs /api/plugins/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await deletePluginMutationOptions.mutationFn!({ pluginId: 9 }, {} as never)
    expect(calledUrl).toBe("/api/plugins/9")
    expect(calledMethod).toBe("DELETE")
  })
})

describe("installPluginMutationOptions / setPluginEnabledMutationOptions / savePluginConfigMutationOptions -- onSuccess seeds the single-plugin cache directly", () => {
  test("install seeds pluginQueryKey(id) with the response plugin", () => {
    const installed = samplePlugin({ id: 42, builtin: false })
    installPluginMutationOptions.onSuccess?.(
      installed,
      { timeout_ms: 5000, config_values: {} },
      undefined,
      {} as never,
    )
    expect(queryClient.getQueryData(pluginQueryKey(42))).toEqual(installed)
  })

  test("enable/disable seeds pluginQueryKey(id) with the response plugin", () => {
    const updated = samplePlugin({ id: 7, enabled: false, state: "disabled" })
    setPluginEnabledMutationOptions.onSuccess?.(updated, { pluginId: 7, enabled: false }, undefined, {} as never)
    expect(queryClient.getQueryData(pluginQueryKey(7))).toEqual(updated)
  })

  test("config save seeds pluginQueryKey(id) with the response plugin", () => {
    const updated = samplePlugin({ id: 8 })
    savePluginConfigMutationOptions.onSuccess?.(updated, { pluginId: 8, values: {} }, undefined, {} as never)
    expect(queryClient.getQueryData(pluginQueryKey(8))).toEqual(updated)
  })
})

describe("deletePluginMutationOptions -- onSuccess evicts the deleted plugin's own single-item cache entry", () => {
  test("removeQueries targets exactly pluginQueryKey(pluginId), not the whole plugins cache tree", () => {
    queryClient.setQueryData(pluginQueryKey(9), samplePlugin({ id: 9 }))
    queryClient.setQueryData(PLUGINS_QUERY_KEY, [samplePlugin({ id: 9 })])
    deletePluginMutationOptions.onSuccess?.(undefined, { pluginId: 9 }, undefined, {} as never)
    expect(queryClient.getQueryData(pluginQueryKey(9))).toBeUndefined()
  })
})

describe("pluginQueryKey / PLUGINS_QUERY_KEY / PLUGIN_CATALOG_QUERY_KEY -- distinct, stable keys", () => {
  test("the list key, the catalog key, and a single-plugin key never collide", () => {
    expect(pluginQueryKey(1)).not.toEqual(PLUGINS_QUERY_KEY)
    expect(pluginQueryKey(1)).toEqual(["plugins", 1])
    expect(PLUGIN_CATALOG_QUERY_KEY).toEqual(["plugins", "catalog"])
  })
})
