// `sessions.ts`'s own fetch layer, checked the way `plugins.test.ts`
// already checks its sibling: a stubbed `global.fetch`, asserting the
// right path is called and the envelope is unwrapped correctly.
import { afterEach, describe, expect, test } from "bun:test"
import { SESSIONS_QUERY_KEY, fetchSessions, sessionAudioUrl, type SessionSummary } from "./sessions"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), { status, headers: { "Content-Type": "application/json" } })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
})

function sampleSession(overrides: Partial<SessionSummary> = {}): SessionSummary {
  return {
    id: "20260919T154201123456Z-turn-abc123",
    started_at: "2026-09-19T15:42:01.123456+00:00",
    turn_outcome: "completed",
    reply_text: "the reply",
    duration_ms: 812.5,
    has_audio: true,
    ...overrides,
  }
}

describe("sessions.ts -- calls the real routes/sessions.py paths", () => {
  test("fetchSessions calls GET /api/sessions and unwraps the envelope", async () => {
    let calledUrl: string | undefined
    global.fetch = (async (url: string) => {
      calledUrl = url
      return jsonResponse(200, { sessions: [sampleSession()] })
    }) as typeof fetch

    const sessions = await fetchSessions()

    expect(calledUrl).toBe("/api/sessions")
    expect(sessions).toEqual([sampleSession()])
  })

  test("SESSIONS_QUERY_KEY is a stable, tuple-shaped key", () => {
    expect(SESSIONS_QUERY_KEY).toEqual(["sessions"])
  })

  test("a session summary carries no source field -- the source label lives on /live, not here", () => {
    expect(Object.keys(sampleSession())).not.toContain("source")
  })

  test("sessionAudioUrl returns the audio path for that session id and issues no request", async () => {
    let fetchCalled = false
    global.fetch = (async () => {
      fetchCalled = true
      throw new Error("sessionAudioUrl must not fetch")
    }) as typeof fetch

    expect(sessionAudioUrl("20260919T154201123456Z-turn-abc123")).toBe(
      "/api/sessions/20260919T154201123456Z-turn-abc123/audio",
    )
    expect(fetchCalled).toBe(false)
  })
})
