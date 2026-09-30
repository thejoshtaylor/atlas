import { afterEach, describe, expect, test } from "bun:test"
import {
  createSpeakerMutationOptions,
  deleteSpeakerMutationOptions,
  enrollPhraseMutationOptions,
  fetchEnrollmentPhrases,
  fetchSpeakers,
  updateSpeakerMutationOptions,
} from "./speakers"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("speakers.ts -- calls the real routes/speakers.py paths", () => {
  test("fetchSpeakers calls GET /api/speakers", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, [])
    }) as typeof fetch
    await fetchSpeakers()
    expect(calledUrl).toBe("/api/speakers")
  })

  test("createSpeakerMutationOptions POSTs {display_name, linked_user_id} to /api/speakers", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(201, {
        id: 1,
        display_name: "Member B",
        linked_user_id: null,
        created_at: "2026-09-28T00:00:00Z",
        enrolled_phrases: 0,
        required_phrases: 5,
        model_id: null,
      })
    }) as typeof fetch
    const created = await createSpeakerMutationOptions.mutationFn!(
      { displayName: "Member B", linkedUserId: null },
      {} as never,
    )
    expect(calledUrl).toBe("/api/speakers")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ display_name: "Member B", linked_user_id: null })
    expect(created.display_name).toBe("Member B")
  })

  test("deleteSpeakerMutationOptions DELETEs /api/speakers/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await deleteSpeakerMutationOptions.mutationFn!({ speakerId: 7 }, {} as never)
    expect(calledUrl).toBe("/api/speakers/7")
    expect(calledMethod).toBe("DELETE")
  })

  test("updateSpeakerMutationOptions PATCHes {can_control_home} to /api/speakers/{id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, {
        id: 7,
        display_name: "Member B",
        linked_user_id: null,
        created_at: "2026-09-28T00:00:00Z",
        enrolled_phrases: 0,
        required_phrases: 5,
        model_id: null,
        retroactive_clips: 0,
        can_control_home: false,
      })
    }) as typeof fetch
    const updated = await updateSpeakerMutationOptions.mutationFn!(
      { speakerId: 7, canControlHome: false },
      {} as never,
    )
    expect(calledUrl).toBe("/api/speakers/7")
    expect(calledMethod).toBe("PATCH")
    expect(JSON.parse(calledBody!)).toEqual({ can_control_home: false })
    expect(updated.can_control_home).toBe(false)
  })

  test("fetchEnrollmentPhrases calls GET /api/speakers/enrollment-phrases", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, { phrases: ["one", "two"] })
    }) as typeof fetch
    const result = await fetchEnrollmentPhrases()
    expect(calledUrl).toBe("/api/speakers/enrollment-phrases")
    expect(result.phrases).toEqual(["one", "two"])
  })

  test("enrollPhraseMutationOptions POSTs {device_id} to /api/speakers/{id}/enrollment/{index}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, {
        speaker_id: 1,
        phrase_index: 0,
        speech_ms: 1234,
        enrolled_phrases: 1,
        required_phrases: 5,
      })
    }) as typeof fetch
    const result = await enrollPhraseMutationOptions.mutationFn!(
      { speakerId: 1, phraseIndex: 0, deviceId: 7 },
      {} as never,
    )
    expect(calledUrl).toBe("/api/speakers/1/enrollment/0")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ device_id: 7 })
    expect(result.speech_ms).toBe(1234)
  })
})
