import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { SessionSummary } from "@/lib/sessions"
import { deriveSessionsScreenState } from "./deriveSessionsScreenState"

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

describe("deriveSessionsScreenState", () => {
  test("a pending query is loading", () => {
    expect(deriveSessionsScreenState({ status: "pending", data: undefined, error: undefined })).toEqual({
      kind: "loading",
    })
  })

  test("an errored query carries the server's own message", () => {
    const state = deriveSessionsScreenState({
      status: "error",
      data: undefined,
      error: new ApiError(500, "the server named this reason"),
    })
    expect(state).toEqual({ kind: "error", message: "the server named this reason" })
  })

  test("a non-ApiError error falls back to the generic copy", () => {
    const state = deriveSessionsScreenState({ status: "error", data: undefined, error: new Error("network") })
    expect(state).toEqual({ kind: "error", message: "Couldn't load sessions. Try again." })
  })

  test("a successful query with sessions is ready", () => {
    const sessions = [sampleSession()]
    expect(deriveSessionsScreenState({ status: "success", data: sessions, error: undefined })).toEqual({
      kind: "ready",
      sessions,
    })
  })
})
