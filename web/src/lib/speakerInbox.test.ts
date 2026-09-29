import { afterEach, describe, expect, test } from "bun:test"
import {
  assignVoiceMutationOptions,
  deleteRetroactiveClipMutationOptions,
  fetchVoiceInbox,
  retroactiveClipAudioUrl,
  voiceInboxAudioUrl,
} from "./speakerInbox"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("speakerInbox.ts -- calls the real routes/speaker_inbox.py paths", () => {
  test("fetchVoiceInbox calls GET /api/speakers/voice-inbox", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, { items: [] })
    }) as typeof fetch
    await fetchVoiceInbox()
    expect(calledUrl).toBe("/api/speakers/voice-inbox")
  })

  test("assigning to an existing member POSTs {speaker_id}", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    let calledBody: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      calledBody = init?.body as string
      return jsonResponse(200, { speaker_id: 1, phrase_index: 100, speech_ms: 2000, dropped_phrase_indices: [] })
    }) as typeof fetch
    await assignVoiceMutationOptions.mutationFn!(
      { sessionId: "a b", target: { kind: "existing", speakerId: 1 } },
      {} as never,
    )
    expect(calledUrl).toBe("/api/speakers/voice-inbox/a%20b/assign")
    expect(calledMethod).toBe("POST")
    expect(JSON.parse(calledBody!)).toEqual({ speaker_id: 1 })
  })

  test("assigning to a new member POSTs {new_speaker_name}", async () => {
    let calledBody: string | undefined
    global.fetch = (async (_url: string, init?: RequestInit) => {
      calledBody = init?.body as string
      return jsonResponse(200, { speaker_id: 2, phrase_index: 100, speech_ms: 2000, dropped_phrase_indices: [] })
    }) as typeof fetch
    await assignVoiceMutationOptions.mutationFn!(
      { sessionId: "s1", target: { kind: "new", displayName: "Guest" } },
      {} as never,
    )
    expect(JSON.parse(calledBody!)).toEqual({ new_speaker_name: "Guest" })
  })

  test("deleting a clip calls DELETE /api/speakers/1/retroactive-clips/100", async () => {
    let calledUrl: string | undefined
    let calledMethod: string | undefined
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledUrl = url
      calledMethod = init?.method
      return new Response(null, { status: 204 })
    }) as typeof fetch
    await deleteRetroactiveClipMutationOptions.mutationFn!({ speakerId: 1, phraseIndex: 100 }, {} as never)
    expect(calledUrl).toBe("/api/speakers/1/retroactive-clips/100")
    expect(calledMethod).toBe("DELETE")
  })

  test("the audio URL helpers are plain, encoded paths", () => {
    expect(voiceInboxAudioUrl("a b")).toBe("/api/speakers/voice-inbox/a%20b/audio")
    expect(retroactiveClipAudioUrl(1, 100)).toBe("/api/speakers/1/retroactive-clips/100/audio")
  })
})
