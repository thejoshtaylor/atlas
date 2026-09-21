import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { WakeEvent, WakeEventsResponse } from "@/lib/wakeTuning"
import { clearsThreshold, deriveWakeTuningScreenState, type QueryLike } from "./deriveWakeTuningScreenState"

function pending(): QueryLike<WakeEventsResponse> {
  return { status: "pending", data: undefined, error: undefined }
}
function errored(error: unknown): QueryLike<WakeEventsResponse> {
  return { status: "error", data: undefined, error }
}
function success(data: WakeEventsResponse): QueryLike<WakeEventsResponse> {
  return { status: "success", data, error: undefined }
}

function event(overrides: Partial<WakeEvent> = {}): WakeEvent {
  return {
    id: 1,
    source: "camera",
    engine: "openwakeword",
    score: 0.5,
    allowed: true,
    block_reason: null,
    recorded_at: "2026-09-21T00:00:00Z",
    ...overrides,
  }
}

function response(overrides: Partial<WakeEventsResponse> = {}): WakeEventsResponse {
  return {
    events: [],
    engine: "openwakeword",
    engine_grades: true,
    threshold: 0.5,
    not_scored_session_count: 0,
    capped: false,
    ...overrides,
  }
}

describe("clearsThreshold -- the boundary comparison, named by both cases", () => {
  test("a score exactly equal to the threshold clears it", () => {
    expect(clearsThreshold(0.5, 0.5)).toBe(true)
  })

  test("a score one hundredth below the threshold does not clear it", () => {
    expect(clearsThreshold(0.49, 0.5)).toBe(false)
  })

  test("a null score neither clears nor fails -- it returns false, and is excluded from both counts by the caller", () => {
    expect(clearsThreshold(null, 0.5)).toBe(false)
  })
})

describe("deriveWakeTuningScreenState -- loading, error, empty", () => {
  test("pending -> loading", () => {
    expect(deriveWakeTuningScreenState(pending(), 0.5)).toEqual({ kind: "loading" })
  })

  test("error with an ApiError -> the server's own message", () => {
    const state = deriveWakeTuningScreenState(errored(new ApiError(500, "boom")), 0.5)
    expect(state).toEqual({ kind: "error", message: "boom" })
  })

  test("error with a non-ApiError -> the generic message", () => {
    const state = deriveWakeTuningScreenState(errored(new Error("network down")), 0.5)
    expect(state).toEqual({ kind: "error", message: "Couldn't load wake history. Try again." })
  })

  test("zero recorded events yields the empty state, never a partition of nothing", () => {
    const state = deriveWakeTuningScreenState(success(response({ events: [] })), 0.5)
    expect(state).toEqual({ kind: "empty" })
  })
})

describe("deriveWakeTuningScreenState -- the partition", () => {
  test("the counts sum to the number of scored events, and not-scored is reported separately, never added into that total", () => {
    const state = deriveWakeTuningScreenState(
      success(
        response({
          events: [
            event({ id: 1, score: 0.9 }),
            event({ id: 2, score: 0.3 }),
            event({ id: 3, score: null }),
          ],
          not_scored_session_count: 4,
        }),
      ),
      0.5,
    )
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.clearingCount).toBe(1)
    expect(state.belowCount).toBe(1)
    expect(state.totalScored).toBe(2)
    expect(state.clearingCount + state.belowCount).toBe(state.totalScored)
    expect(state.notScoredCount).toBe(4)
    expect(state.capped).toBe(false)
  })

  test("a threshold change recomputes the counts from the already-held event list, taking no argument that could be a network result", () => {
    const query = success(
      response({
        events: [event({ id: 1, score: 0.9 }), event({ id: 2, score: 0.3 })],
      }),
    )
    const low = deriveWakeTuningScreenState(query, 0.1)
    const high = deriveWakeTuningScreenState(query, 0.95)
    if (low.kind !== "ready" || high.kind !== "ready") throw new Error("expected ready")
    expect(low.clearingCount).toBe(2)
    expect(high.clearingCount).toBe(0)
  })

  test("the histogram places every scored event in exactly one bucket, covering the whole range with no overlap", () => {
    const state = deriveWakeTuningScreenState(
      success(
        response({
          events: [
            event({ id: 1, score: 0.0 }),
            event({ id: 2, score: 0.05 }),
            event({ id: 3, score: 0.5 }),
            event({ id: 4, score: 0.99 }),
            event({ id: 5, score: 1.0 }),
            event({ id: 6, score: null }),
          ],
        }),
      ),
      0.5,
    )
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.histogram).toHaveLength(10)
    const totalBucketed = state.histogram.reduce((sum, bucket) => sum + bucket.count, 0)
    // 5 scored events land in a bucket; the null-scored one lands in none.
    expect(totalBucketed).toBe(5)
    // Contiguous, non-overlapping coverage of [0, 1].
    for (let index = 0; index < state.histogram.length - 1; index += 1) {
      expect(state.histogram[index].to).toBe(state.histogram[index + 1].from)
    }
    expect(state.histogram[0].from).toBe(0)
    expect(state.histogram[state.histogram.length - 1].to).toBe(1)
  })

  test("the no-gradient disclosure flag is present when the response says the engine does not grade", () => {
    const state = deriveWakeTuningScreenState(
      success(response({ events: [event()], engine: "vosk", engine_grades: false })),
      0.5,
    )
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.engineGrades).toBe(false)
    expect(state.engine).toBe("vosk")
  })

  test("the no-gradient disclosure flag is absent (true) when the response says the engine grades", () => {
    const state = deriveWakeTuningScreenState(
      success(response({ events: [event()], engine: "openwakeword", engine_grades: true })),
      0.5,
    )
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.engineGrades).toBe(true)
  })
})

describe("deriveWakeTuningScreenState -- the two kinds of fact stay separate", () => {
  test("an event whose score clears the previewed threshold but was historically blocked carries both facts, independently", () => {
    const state = deriveWakeTuningScreenState(
      success(
        response({
          events: [event({ id: 1, score: 0.9, allowed: false, block_reason: "refractory" })],
        }),
      ),
      0.5,
    )
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.events[0].clearsPreviewedThreshold).toBe(true)
    expect(state.events[0].historicalOutcome).toBe("blocked_refractory")
  })

  test("every BlockReason maps to a distinct historicalOutcome, and an allowed hit maps to started_turn", () => {
    const state = deriveWakeTuningScreenState(
      success(
        response({
          events: [
            event({ id: 1, allowed: true, block_reason: null }),
            event({ id: 2, allowed: false, block_reason: "below_threshold" }),
            event({ id: 3, allowed: false, block_reason: "refractory" }),
            event({ id: 4, allowed: false, block_reason: "media_playing" }),
          ],
        }),
      ),
      0.5,
    )
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.events.map((e) => e.historicalOutcome)).toEqual([
      "started_turn",
      "blocked_below_threshold",
      "blocked_refractory",
      "blocked_media_playing",
    ])
  })
})

describe("deriveWakeTuningScreenState -- a partial history says so (WR-06)", () => {
  test("the server's cap is carried onto the ready state, never dropped", () => {
    const state = deriveWakeTuningScreenState(
      success(response({ events: [event({ id: 1, score: 0.9 })], capped: true })),
      0.5,
    )
    if (state.kind !== "ready") throw new Error("expected ready")
    expect(state.capped).toBe(true)
  })
})

describe("buildHistogram -- the bucket the threshold falls inside (WR-07)", () => {
  test("a score of 0.87 against a threshold of 0.85 is a clearing event in its own bucket, matching the count line", () => {
    const state = deriveWakeTuningScreenState(
      success(response({ events: [event({ id: 1, score: 0.87 })] })),
      0.85,
    )
    if (state.kind !== "ready") throw new Error("expected ready")

    // The count line says it clears.
    expect(state.clearingCount).toBe(1)
    // And so does the bar it sits in -- [0.80, 0.90), whose lower bound
    // does not clear. Asking the lower bound was the defect.
    const straddling = state.histogram[8]
    expect(straddling.from).toBeCloseTo(0.8)
    expect(straddling.count).toBe(1)
    expect(straddling.clearingCount).toBe(1)
    expect(straddling.belowCount).toBe(0)
  })

  test("the straddling bucket is genuinely split, not shaded one way for both of its events", () => {
    const state = deriveWakeTuningScreenState(
      success(
        response({ events: [event({ id: 1, score: 0.87 }), event({ id: 2, score: 0.81 })] }),
      ),
      0.85,
    )
    if (state.kind !== "ready") throw new Error("expected ready")

    const straddling = state.histogram[8]
    expect(straddling.count).toBe(2)
    expect(straddling.clearingCount).toBe(1)
    expect(straddling.belowCount).toBe(1)
    expect(state.clearingCount).toBe(1)
    expect(state.belowCount).toBe(1)
  })

  test("every bucket's two parts sum to its own count, and the buckets' clearing parts sum to the headline count", () => {
    const state = deriveWakeTuningScreenState(
      success(
        response({
          events: [
            event({ id: 1, score: 0.05 }),
            event({ id: 2, score: 0.5 }),
            event({ id: 3, score: 0.55 }),
            event({ id: 4, score: 1.0 }),
            event({ id: 5, score: null }),
          ],
        }),
      ),
      0.52,
    )
    if (state.kind !== "ready") throw new Error("expected ready")

    for (const bucket of state.histogram) {
      expect(bucket.clearingCount + bucket.belowCount).toBe(bucket.count)
    }
    const clearingAcrossBuckets = state.histogram.reduce((sum, b) => sum + b.clearingCount, 0)
    expect(clearingAcrossBuckets).toBe(state.clearingCount)
    const belowAcrossBuckets = state.histogram.reduce((sum, b) => sum + b.belowCount, 0)
    expect(belowAcrossBuckets).toBe(state.belowCount)
  })
})
