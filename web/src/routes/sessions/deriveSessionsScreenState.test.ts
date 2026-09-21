import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { SessionSummary } from "@/lib/sessions"
import { deriveSessionsScreenState, formatSessionDuration, summarizeSessionOutcome } from "./deriveSessionsScreenState"

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

  // Task 3 (TDD, RED): the sessions list genuinely can be empty (a fresh
  // install before the first turn) -- `derivePluginsScreenState.ts` has
  // no empty branch because the plugin list never is; this diverges
  // deliberately.
  test("a successful query with zero sessions is empty", () => {
    expect(deriveSessionsScreenState({ status: "success", data: [], error: undefined })).toEqual({
      kind: "empty",
    })
  })
})

describe("summarizeSessionOutcome", () => {
  test("a session with a reply summarises as the reply text", () => {
    expect(summarizeSessionOutcome(sampleSession({ reply_text: "Turning the lights on." }))).toBe(
      "Turning the lights on.",
    )
  })

  test("empty_transcript summarises as the fixed copy, even with no reply text", () => {
    expect(
      summarizeSessionOutcome(sampleSession({ reply_text: null, turn_outcome: "empty_transcript" })),
    ).toBe("Understood no speech.")
  })

  test("empty_reply summarises as the fixed copy", () => {
    expect(summarizeSessionOutcome(sampleSession({ reply_text: null, turn_outcome: "empty_reply" }))).toBe(
      "No reply was given.",
    )
  })

  test("an outcome this contract does not name renders verbatim, never a fabricated label", () => {
    expect(
      summarizeSessionOutcome(sampleSession({ reply_text: null, turn_outcome: "brain_provider_unavailable" })),
    ).toBe("brain_provider_unavailable")
  })

  // `turn/controller.py` sets at least nine distinct outcome values today
  // (this plan's own action text) -- enumerated here so a tenth added
  // later shows up as a raw string on screen, never silently mapped to a
  // wrong label.
  test("every outcome turn/controller.py is known to set today resolves to something -- reply text, a named mapping, or the raw value verbatim", () => {
    const knownOutcomes = [
      "completed",
      "empty_transcript",
      "empty_reply",
      "timeout",
      "round_cap",
      "wake_gate_blocked",
      "mic_denied",
      "stt_error",
      "brain_error",
    ]
    for (const outcome of knownOutcomes) {
      const summary = summarizeSessionOutcome(sampleSession({ reply_text: null, turn_outcome: outcome }))
      expect(typeof summary).toBe("string")
      expect(summary.length).toBeGreaterThan(0)
    }
  })
})

describe("formatSessionDuration", () => {
  test("formats a known duration with the DevMicRoute millisecond formatter, suffixed 'end to end'", () => {
    expect(formatSessionDuration(812.5)).toBe("812.5 ms end to end")
  })

  test("a null duration is not reached, end to end", () => {
    expect(formatSessionDuration(null)).toBe("not reached end to end")
  })
})
