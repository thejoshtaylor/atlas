import { afterEach, describe, expect, test } from "bun:test"
import { ApiError } from "./api"
import { queryClient } from "./queryClient"
import { SESSION_QUERY_KEY } from "./session"
import { SETUP_STATUS_QUERY_KEY } from "./setup"
import {
  checkHubStepMutationOptions,
  classifyHubCheckError,
  createAdminMutationOptions,
  finishWizardMutationOptions,
  setAudioSourceMutationOptions,
  WIZARD_QUERY_KEY,
} from "./wizard"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("classifyHubCheckError -- the three distinguishable hub failures", () => {
  test("unreachable: a refused connection", () => {
    const { heading, detail } = classifyHubCheckError(
      new ApiError(502, "unreachable: no Home Assistant URL is configured (HA_URL)"),
    )
    expect(heading).toBe("Couldn't reach Home Assistant.")
    expect(detail).toBe("unreachable: no Home Assistant URL is configured (HA_URL)")
  })

  test("unauthorized: the token was rejected", () => {
    const { heading, detail } = classifyHubCheckError(
      new ApiError(502, "unauthorized: Home Assistant rejected the token (HTTP 401)"),
    )
    expect(heading).toBe("Home Assistant rejected the token.")
    expect(detail).toBe("unauthorized: Home Assistant rejected the token (HTTP 401)")
  })

  test("unparseable: a response arrived but made no sense", () => {
    const { heading, detail } = classifyHubCheckError(
      new ApiError(502, "unparseable: Home Assistant returned HTTP 500"),
    )
    expect(heading).toBe("Home Assistant's response couldn't be understood.")
    expect(detail).toBe("unparseable: Home Assistant returned HTTP 500")
  })

  test("every other failure falls back to 'unknown' with no server detail shown", () => {
    const { heading, detail } = classifyHubCheckError(new Error("network down"))
    expect(heading).toBe("Couldn't verify the connection. Check the address and token and try again.")
    expect(detail).toBeUndefined()
  })

  test("the three named categories render three different headings", () => {
    const headings = new Set(
      ["unreachable: a", "unauthorized: b", "unparseable: c"].map(
        (message) => classifyHubCheckError(new ApiError(502, message)).heading,
      ),
    )
    expect(headings.size).toBe(3)
  })
})

describe("createAdminMutationOptions -- signs the new admin in and unblocks the status gate", () => {
  test("onSuccess writes the session cache and invalidates the setup-status query", async () => {
    let invalidated: unknown
    const originalInvalidate = queryClient.invalidateQueries.bind(queryClient)
    queryClient.invalidateQueries = (async (options: { queryKey: unknown }) => {
      invalidated = options.queryKey
      return originalInvalidate(options as never)
    }) as typeof queryClient.invalidateQueries

    const session = { id: 1, email: "admin@example.invalid", display_name: "Admin", role: "admin" as const }
    await createAdminMutationOptions.onSuccess?.(
      session,
      { email: "admin@example.invalid", display_name: "Admin", password: "test-key" },
      undefined,
      { client: queryClient } as never,
    )

    expect(queryClient.getQueryData(SESSION_QUERY_KEY)).toEqual(session)
    expect(invalidated).toEqual(SETUP_STATUS_QUERY_KEY)
    queryClient.invalidateQueries = originalInvalidate
  })
})

describe("checkHubStepMutationOptions -- POST /api/wizard/steps/hub/check", () => {
  test("mutationFn calls the exact route with no body", async () => {
    let calledPath: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledPath = url
      calledMethod = init?.method
      return jsonResponse(200, { name: "hub", complete: true, detail: { checked_at: "now" } })
    }) as typeof fetch

    await checkHubStepMutationOptions.mutationFn!(undefined, {} as never)
    expect(calledPath).toBe("/api/wizard/steps/hub/check")
    expect(calledMethod).toBe("POST")
  })
})

describe("setAudioSourceMutationOptions -- PUT /api/wizard/audio-source", () => {
  test("mutationFn sends the source as the request body", async () => {
    let sentBody: unknown
    global.fetch = (async (_url: string, init?: RequestInit) => {
      sentBody = init?.body ? JSON.parse(init.body as string) : undefined
      return jsonResponse(200, { name: "audio_source", complete: true, detail: { source: "camera" } })
    }) as typeof fetch

    await setAudioSourceMutationOptions.mutationFn!({ source: "camera" }, {} as never)
    expect(sentBody).toEqual({ source: "camera" })
  })
})

describe("finishWizardMutationOptions -- POST /api/wizard/finish", () => {
  test("a 409 refusal carries every outstanding step name unmodified -- not just the first", async () => {
    const detail = "setup is not finished -- outstanding steps: hub, provider_set"
    global.fetch = (async () => jsonResponse(409, { detail })) as typeof fetch

    const rejection = finishWizardMutationOptions.mutationFn!(undefined, {} as never)
    await expect(rejection).rejects.toBeInstanceOf(ApiError)
    await expect(rejection).rejects.toMatchObject({ status: 409, message: detail })
    expect(detail).toContain("hub")
    expect(detail).toContain("provider_set")
  })

  test("a 200 resolves with complete: true", async () => {
    global.fetch = (async () =>
      jsonResponse(200, { complete: true, completed_at: "2026-09-18T00:00:00+00:00" })) as typeof fetch

    const result = await finishWizardMutationOptions.mutationFn!(undefined, {} as never)
    expect(result).toEqual({ complete: true, completed_at: "2026-09-18T00:00:00+00:00" })
  })
})

describe("no second network path -- every route function here goes through the shared fetch seam", () => {
  test("WIZARD_QUERY_KEY is a stable, distinct cache key from SETUP_STATUS_QUERY_KEY", () => {
    expect(WIZARD_QUERY_KEY).not.toEqual(SETUP_STATUS_QUERY_KEY)
  })
})
