// EnrollmentPanel's own state machine, kept apart from its JSX -- copies
// `deriveSpeakersScreenState.ts`'s discipline. Imports nothing from
// React: what belongs here is only ever a fact about data, never a fact
// about a render.
//
// The public shape (`EnrollmentState`) is exactly the union
// 11-11-PLAN.md's `<interfaces>` names. `EnrollmentReducerState` is the
// internal shape the reducer actually carries (it needs `deviceId`,
// `requiredPhrases` and `enrolledPhrases` alongside the fields the public
// union exposes) -- `deriveEnrollmentState` is the seam between the two,
// the same split `deriveSpeakersScreenState`'s `QueryLike<T>` makes
// between a query's raw shape and the screen's own rendered shape.

export type EnrollmentState =
  | { kind: "choose_device" }
  | { kind: "ready"; phraseIndex: number; phrase: string }
  | { kind: "recording"; phraseIndex: number; phrase: string }
  | { kind: "recorded"; phraseIndex: number; speechMs: number; enrolledPhrases: number }
  | { kind: "failed"; phraseIndex: number; message: string }
  | { kind: "complete"; enrolledPhrases: number }

export interface EnrollmentReducerState {
  phraseIndex: number
  status: "choose_device" | "ready" | "recording" | "recorded" | "failed" | "complete"
  deviceId: number | null
  requiredPhrases: number
  enrolledPhrases: number
  speechMs: number | null
  message: string | null
}

export type EnrollmentAction =
  | { type: "select_device"; deviceId: number }
  | { type: "start_recording" }
  | { type: "record_success"; speechMs: number; enrolledPhrases: number }
  | { type: "record_failure"; message: string }
  | { type: "next_phrase" }

/**
 * `startPhraseIndex` defaults to 0: an "Enroll" and a "Re-record phrases"
 * open both start reading phrase 0 (D-02's "one at a time" applies to
 * both a first enrollment and a re-record -- there is no partial-resume
 * behavior this plan asks for).
 */
export function initialEnrollmentState(requiredPhrases: number, startPhraseIndex = 0): EnrollmentReducerState {
  return {
    phraseIndex: startPhraseIndex,
    status: "choose_device",
    deviceId: null,
    requiredPhrases,
    enrolledPhrases: 0,
    speechMs: null,
    message: null,
  }
}

export function enrollmentReducer(
  state: EnrollmentReducerState,
  action: EnrollmentAction,
): EnrollmentReducerState {
  switch (action.type) {
    case "select_device":
      return { ...state, deviceId: action.deviceId, status: "ready" }
    case "start_recording":
      // "Record phrase" (from ready) and "Try again" (from failed) both
      // land here -- a retry re-records the SAME phraseIndex, since
      // nothing below touches it.
      if (state.status !== "ready" && state.status !== "failed") return state
      return { ...state, status: "recording", message: null }
    case "record_success":
      return { ...state, status: "recorded", speechMs: action.speechMs, enrolledPhrases: action.enrolledPhrases }
    case "record_failure":
      return { ...state, status: "failed", message: action.message }
    case "next_phrase": {
      if (state.status !== "recorded") return state
      // Completion is decided by the server's own enrolled_phrases count,
      // never the local phraseIndex counter -- a re-record can complete
      // enrollment on a phrase index that isn't 4, and the server is the
      // one source of truth for how many phrases are actually stored.
      if (state.enrolledPhrases >= state.requiredPhrases) {
        return { ...state, status: "complete" }
      }
      return { ...state, status: "ready", phraseIndex: state.phraseIndex + 1, speechMs: null }
    }
    default:
      return state
  }
}

export function deriveEnrollmentState(state: EnrollmentReducerState, phrases: string[]): EnrollmentState {
  const phrase = phrases[state.phraseIndex] ?? ""
  switch (state.status) {
    case "choose_device":
      return { kind: "choose_device" }
    case "ready":
      return { kind: "ready", phraseIndex: state.phraseIndex, phrase }
    case "recording":
      return { kind: "recording", phraseIndex: state.phraseIndex, phrase }
    case "recorded":
      return {
        kind: "recorded",
        phraseIndex: state.phraseIndex,
        speechMs: state.speechMs ?? 0,
        enrolledPhrases: state.enrolledPhrases,
      }
    case "failed":
      return { kind: "failed", phraseIndex: state.phraseIndex, message: state.message ?? "" }
    case "complete":
      return { kind: "complete", enrolledPhrases: state.enrolledPhrases }
  }
}
