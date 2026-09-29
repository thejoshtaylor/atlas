import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { Speaker } from "@/lib/speakers"
import { deriveSpeakersScreenState, formatEnrollmentProgress, type QueryLike } from "./deriveSpeakersScreenState"

function sampleSpeaker(overrides: Partial<Speaker> = {}): Speaker {
  return {
    id: 1,
    display_name: "Ann",
    linked_user_id: null,
    created_at: "2026-09-28T00:00:00Z",
    enrolled_phrases: 0,
    required_phrases: 5,
    model_id: "cam++",
    ...overrides,
  }
}

function pending<T>(): QueryLike<T> {
  return { status: "pending", data: undefined, error: undefined }
}

function errored<T>(error: unknown): QueryLike<T> {
  return { status: "error", data: undefined, error }
}

function success<T>(data: T): QueryLike<T> {
  return { status: "success", data, error: undefined }
}

describe("deriveSpeakersScreenState", () => {
  test("pending -> loading", () => {
    expect(deriveSpeakersScreenState({ speakers: pending() })).toEqual({ kind: "loading" })
  })

  test("error with an ApiError -> the ApiError's own message", () => {
    const state = deriveSpeakersScreenState({ speakers: errored(new ApiError(500, "boom")) })
    expect(state).toEqual({ kind: "error", message: "boom" })
  })

  test("error with a non-ApiError -> the generic connection message", () => {
    const state = deriveSpeakersScreenState({ speakers: errored(new TypeError("network down")) })
    expect(state).toEqual({ kind: "error", message: "Can't reach the server. Check your connection and try again." })
  })

  test("success -> ready with the speakers list", () => {
    const speakers = [sampleSpeaker()]
    expect(deriveSpeakersScreenState({ speakers: success(speakers) })).toEqual({ kind: "ready", speakers })
  })

  test("success with undefined data -> ready with an empty list", () => {
    expect(deriveSpeakersScreenState({ speakers: success(undefined as unknown as Speaker[]) })).toEqual({
      kind: "ready",
      speakers: [],
    })
  })
})

describe("formatEnrollmentProgress", () => {
  test("model_id null -> 'Speaker ID model not set', regardless of enrolled_phrases", () => {
    expect(formatEnrollmentProgress(sampleSpeaker({ model_id: null, enrolled_phrases: 3 }))).toBe(
      "Speaker ID model not set",
    )
  })

  test("partially enrolled -> 'N of M phrases'", () => {
    expect(formatEnrollmentProgress(sampleSpeaker({ enrolled_phrases: 3, required_phrases: 5 }))).toBe(
      "3 of 5 phrases",
    )
  })

  test("fully enrolled -> 'Enrolled'", () => {
    expect(formatEnrollmentProgress(sampleSpeaker({ enrolled_phrases: 5, required_phrases: 5 }))).toBe("Enrolled")
  })

  test("enrolled_phrases past required (shouldn't happen, but must not break) -> 'Enrolled'", () => {
    expect(formatEnrollmentProgress(sampleSpeaker({ enrolled_phrases: 6, required_phrases: 5 }))).toBe("Enrolled")
  })
})
