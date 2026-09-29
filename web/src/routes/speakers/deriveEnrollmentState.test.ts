import { describe, expect, test } from "bun:test"
import {
  deriveEnrollmentState,
  enrollmentReducer,
  initialEnrollmentState,
  type EnrollmentReducerState,
} from "./deriveEnrollmentState"

const PHRASES = ["one", "two", "three", "four", "five"]

describe("initialEnrollmentState / deriveEnrollmentState", () => {
  test("starts at choose_device", () => {
    const state = initialEnrollmentState(5)
    expect(deriveEnrollmentState(state, PHRASES)).toEqual({ kind: "choose_device" })
  })
})

describe("enrollmentReducer -- the full happy path from phrase 0 through phrase 4 to complete", () => {
  test("moves choose_device -> ready(0) -> recording -> recorded -> ready(1) -> ... -> complete", () => {
    let state: EnrollmentReducerState = initialEnrollmentState(5)

    state = enrollmentReducer(state, { type: "select_device", deviceId: 7 })
    expect(deriveEnrollmentState(state, PHRASES)).toEqual({ kind: "ready", phraseIndex: 0, phrase: "one" })

    for (let phraseIndex = 0; phraseIndex < 5; phraseIndex += 1) {
      state = enrollmentReducer(state, { type: "start_recording" })
      expect(deriveEnrollmentState(state, PHRASES)).toEqual({
        kind: "recording",
        phraseIndex,
        phrase: PHRASES[phraseIndex],
      })

      const enrolledPhrases = phraseIndex + 1
      state = enrollmentReducer(state, { type: "record_success", speechMs: 1000 + phraseIndex, enrolledPhrases })
      expect(deriveEnrollmentState(state, PHRASES)).toEqual({
        kind: "recorded",
        phraseIndex,
        speechMs: 1000 + phraseIndex,
        enrolledPhrases,
      })

      state = enrollmentReducer(state, { type: "next_phrase" })
      if (phraseIndex < 4) {
        expect(deriveEnrollmentState(state, PHRASES)).toEqual({
          kind: "ready",
          phraseIndex: phraseIndex + 1,
          phrase: PHRASES[phraseIndex + 1],
        })
      } else {
        expect(deriveEnrollmentState(state, PHRASES)).toEqual({ kind: "complete", enrolledPhrases: 5 })
      }
    }
  })
})

describe("enrollmentReducer -- a failure keeps the same phrase index, and a retry re-records it", () => {
  test("failed carries the same phraseIndex; start_recording from failed goes to recording on that index", () => {
    let state: EnrollmentReducerState = initialEnrollmentState(5)
    state = enrollmentReducer(state, { type: "select_device", deviceId: 7 })
    state = enrollmentReducer(state, { type: "start_recording" })
    state = enrollmentReducer(state, { type: "start_recording" }) // phrase 0, recording
    // Move to phrase 2 first, to prove a failure/retry doesn't reset to 0.
    state = enrollmentReducer(state, { type: "record_success", speechMs: 1000, enrolledPhrases: 1 })
    state = enrollmentReducer(state, { type: "next_phrase" })
    state = enrollmentReducer(state, { type: "start_recording" }) // phrase 1, recording
    state = enrollmentReducer(state, { type: "record_failure", message: "no speech was heard in time" })

    expect(deriveEnrollmentState(state, PHRASES)).toEqual({
      kind: "failed",
      phraseIndex: 1,
      message: "no speech was heard in time",
    })

    state = enrollmentReducer(state, { type: "start_recording" })
    expect(deriveEnrollmentState(state, PHRASES)).toEqual({ kind: "recording", phraseIndex: 1, phrase: "two" })
  })
})

describe("enrollmentReducer -- ignores actions that don't apply to the current status", () => {
  test("start_recording from choose_device is a no-op", () => {
    const state = initialEnrollmentState(5)
    expect(enrollmentReducer(state, { type: "start_recording" })).toEqual(state)
  })

  test("next_phrase from ready is a no-op", () => {
    let state = initialEnrollmentState(5)
    state = enrollmentReducer(state, { type: "select_device", deviceId: 7 })
    expect(enrollmentReducer(state, { type: "next_phrase" })).toEqual(state)
  })
})
