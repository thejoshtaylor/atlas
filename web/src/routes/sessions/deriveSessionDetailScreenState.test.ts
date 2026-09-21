// Task 2 (TDD), RED: `./deriveSessionDetailScreenState` does not exist yet
// -- this file targets it before it is written, following
// `deriveCalibrationScreenState.ts`'s own multi-branch shape.
import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { SessionDetail, TimelineEntry } from "@/lib/sessions"
import { activeTimelineIndexAt, deriveSessionDetailScreenState } from "./deriveSessionDetailScreenState"

function sampleDetail(overrides: Partial<SessionDetail> = {}): SessionDetail {
  return {
    id: "20260919T154201123456Z-turn-abc123",
    started_at: "2026-09-19T15:42:01.123456+00:00",
    turn_outcome: "completed",
    transcript: "turn the kitchen lights on",
    reply_text: "Turning the kitchen lights on.",
    stage_durations_ms: { turn_started_at: null, stt_final_at: 120.0 },
    end_of_speech_to_first_audio_ms: 400.0,
    end_of_speech_to_answer_audio_ms: 600.0,
    audio_format: { encoding: "pcm", sample_rate: 8000 },
    has_audio: true,
    timeline: [{ ts: 1.0, kind: "stage", offset_s: 0.0, stage: "turn_started_at" }],
    ...overrides,
  }
}

describe("deriveSessionDetailScreenState", () => {
  test("a pending query is loading", () => {
    const state = deriveSessionDetailScreenState({
      query: { status: "pending", data: undefined, error: undefined },
      hasEverLoaded: false,
    })
    expect(state).toEqual({ kind: "loading" })
  })

  test("a successful query with a session is ready", () => {
    const session = sampleDetail()
    const state = deriveSessionDetailScreenState({
      query: { status: "success", data: session, error: undefined },
      hasEverLoaded: false,
    })
    expect(state).toEqual({ kind: "ready", session })
  })

  test("a retention refusal, never previously loaded, is removed", () => {
    const state = deriveSessionDetailScreenState({
      query: {
        status: "error",
        data: undefined,
        error: new ApiError(404, "session '20260101T000000000000Z-x' has been removed by the retention sweep"),
      },
      hasEverLoaded: false,
    })
    expect(state).toEqual({ kind: "removed" })
  })

  test("an unrecognised or never-existed id, never previously loaded, is not_found", () => {
    const state = deriveSessionDetailScreenState({
      query: { status: "error", data: undefined, error: new ApiError(404, "no session with id 'x'") },
      hasEverLoaded: false,
    })
    expect(state).toEqual({ kind: "not_found" })
  })

  test("a session that loaded a moment ago and now fails to fetch is removed, regardless of the error text (D-04)", () => {
    const state = deriveSessionDetailScreenState({
      query: { status: "error", data: undefined, error: new ApiError(404, "no session with id 'x'") },
      hasEverLoaded: true,
    })
    expect(state).toEqual({ kind: "removed" })
  })
})

// Task 2 of 08-05 (TDD): activeTimelineIndexAt -- "which row is active at
// time t," as arithmetic with no DOM in it. See deriveSessionDetailScreenState.ts's
// own comment for the boundary rule this covers.

function entryAt(offset: number | null): TimelineEntry {
  return { ts: (offset ?? 0) + 1, kind: "stage", offset_s: offset }
}

describe("activeTimelineIndexAt", () => {
  test("a single entry at offset zero is active at time zero", () => {
    expect(activeTimelineIndexAt([entryAt(0)], 0)).toBe(0)
  })

  test("the inclusive-at-offset boundary and the gap between two later entries", () => {
    const entries = [entryAt(0), entryAt(1.5), entryAt(4.0)]
    expect(activeTimelineIndexAt(entries, 1.4)).toBe(0)
    expect(activeTimelineIndexAt(entries, 1.5)).toBe(1)
    expect(activeTimelineIndexAt(entries, 3.9)).toBe(1)
  })

  test("a time beyond the last entry's offset returns the last index", () => {
    const entries = [entryAt(0), entryAt(1.5), entryAt(4.0)]
    expect(activeTimelineIndexAt(entries, 99)).toBe(2)
  })

  test("a time before the first entry's offset returns null", () => {
    const entries = [entryAt(1.0), entryAt(2.0)]
    expect(activeTimelineIndexAt(entries, 0.5)).toBeNull()
  })

  test("an empty timeline returns null at any time", () => {
    expect(activeTimelineIndexAt([], 0)).toBeNull()
    expect(activeTimelineIndexAt([], 999)).toBeNull()
  })

  test("two entries sharing an offset resolve to the later of the two, matching append order", () => {
    const entries = [entryAt(0), entryAt(2.0), entryAt(2.0)]
    expect(activeTimelineIndexAt(entries, 2.0)).toBe(2)
  })
})
