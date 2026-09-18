import { afterEach, describe, expect, test } from "bun:test"
import { queryClient } from "./queryClient"
import {
  MACROS_QUERY_KEY,
  createMacroMutationOptions,
  deleteMacroMutationOptions,
  fetchMacro,
  fetchMacros,
  macroQueryKey,
  updateMacroMutationOptions,
  type Macro,
} from "./macros"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

function sampleMacro(overrides: Partial<Macro> = {}): Macro {
  return {
    id: 1,
    phrase: "goodnight",
    aliases: ["night night"],
    reply: "Good night.",
    actions: [],
    created_at: "2026-09-18T00:00:00Z",
    updated_at: "2026-09-18T00:00:00Z",
    created_by_user_id: 1,
    reply_cached: true,
    reply_synthesis_degraded: false,
    reply_synthesis_message: null,
    ...overrides,
  }
}

describe("macros.ts -- calls the real routes/macros.py paths", () => {
  test("fetchMacros calls GET /api/macros", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [sampleMacro()])
    }) as typeof fetch
    await fetchMacros()
    expect(calledUrl).toBe("/api/macros")
  })

  test("fetchMacro calls GET /api/macros/{id}", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, sampleMacro({ id: 5 }))
    }) as typeof fetch
    await fetchMacro(5)
    expect(calledUrl).toBe("/api/macros/5")
  })

  test("createMacroMutationOptions POSTs the save shape to /api/macros", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(201, sampleMacro())
    }) as typeof fetch
    await createMacroMutationOptions.mutationFn!(
      { phrase: "goodnight", aliases: [], reply: "Good night.", actions: [] },
      {} as never,
    )
    expect(calledUrl).toBe("/api/macros")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ phrase: "goodnight", aliases: [], reply: "Good night.", actions: [] })
  })

  test("updateMacroMutationOptions PUTs to /api/macros/{id}, with macroId stripped from the body", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, sampleMacro({ id: 7 }))
    }) as typeof fetch
    await updateMacroMutationOptions.mutationFn!(
      { macroId: 7, phrase: "goodnight", aliases: [], reply: "Good night.", actions: [] },
      {} as never,
    )
    expect(calledUrl).toBe("/api/macros/7")
    expect(calledMethod).toBe("PUT")
    expect(JSON.parse(calledBody!)).toEqual({ phrase: "goodnight", aliases: [], reply: "Good night.", actions: [] })
  })

  test("deleteMacroMutationOptions DELETEs /api/macros/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await deleteMacroMutationOptions.mutationFn!({ macroId: 9 }, {} as never)
    expect(calledUrl).toBe("/api/macros/9")
    expect(calledMethod).toBe("DELETE")
  })
})

describe("createMacroMutationOptions / updateMacroMutationOptions -- onSuccess seeds the single-macro cache directly, no second fetch needed to see it", () => {
  test("create seeds macroQueryKey(id) with the response macro", () => {
    const created = sampleMacro({ id: 42 })
    createMacroMutationOptions.onSuccess?.(
      created,
      { phrase: "x", aliases: [], reply: "y", actions: [] },
      undefined,
      {} as never,
    )
    expect(queryClient.getQueryData(macroQueryKey(42))).toEqual(created)
  })

  test("update seeds macroQueryKey(id) with the response macro", () => {
    const updated = sampleMacro({ id: 7, phrase: "goodnight (updated)" })
    updateMacroMutationOptions.onSuccess?.(
      updated,
      { macroId: 7, phrase: "goodnight (updated)", aliases: [], reply: "Good night.", actions: [] },
      undefined,
      {} as never,
    )
    expect(queryClient.getQueryData(macroQueryKey(7))).toEqual(updated)
  })
})

describe("deleteMacroMutationOptions -- onSuccess evicts the deleted macro's own single-item cache entry", () => {
  test("removeQueries targets exactly macroQueryKey(macroId), not the whole macros cache tree", () => {
    queryClient.setQueryData(macroQueryKey(9), sampleMacro({ id: 9 }))
    queryClient.setQueryData(MACROS_QUERY_KEY, [sampleMacro({ id: 9 })])
    deleteMacroMutationOptions.onSuccess?.(undefined, { macroId: 9 }, undefined, {} as never)
    expect(queryClient.getQueryData(macroQueryKey(9))).toBeUndefined()
  })
})

describe("macroQueryKey / MACROS_QUERY_KEY -- distinct, stable keys", () => {
  test("the list key and a single-macro key never collide", () => {
    expect(macroQueryKey(1)).not.toEqual(MACROS_QUERY_KEY)
    expect(macroQueryKey(1)).toEqual(["macros", 1])
  })
})
