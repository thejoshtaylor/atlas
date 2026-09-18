import { describe, expect, test } from "bun:test"
import { ApiError } from "@/lib/api"
import type { EchoCalibrationResult } from "@/lib/calibration"
import {
  deriveCalibrationScreenState,
  type MutationLike,
  type QueryLike,
} from "./deriveCalibrationScreenState"

const RESULT: EchoCalibrationResult = {
  delay_s: 0.142,
  gain: 0.37,
  confidence: 0.91,
  agc_verdict: "present",
  source: "camera",
  placement_note: "kitchen counter, 2m from camera",
  taken_at: "2026-09-17T12:00:00+00:00",
  age_days: 0.01,
}

const idleRun: MutationLike<EchoCalibrationResult> = { status: "idle", data: undefined, error: undefined }
const pendingRun: MutationLike<EchoCalibrationResult> = { status: "pending", data: undefined, error: undefined }

function successQuery(data: EchoCalibrationResult | null): QueryLike<EchoCalibrationResult | null> {
  return { status: "success", data, error: undefined }
}

function pendingQuery(): QueryLike<EchoCalibrationResult | null> {
  return { status: "pending", data: undefined, error: undefined }
}

function errorQuery(error: unknown): QueryLike<EchoCalibrationResult | null> {
  return { status: "error", data: undefined, error }
}

describe("deriveCalibrationScreenState -- before a run", () => {
  test("the initial GET in flight is the loading state", () => {
    expect(deriveCalibrationScreenState({ latest: pendingQuery(), run: idleRun })).toEqual({
      kind: "loading",
    })
  })

  test("a 404 (null) is the empty state, not a failure", () => {
    expect(deriveCalibrationScreenState({ latest: successQuery(null), run: idleRun })).toEqual({
      kind: "empty",
    })
  })

  test("a stored calibration renders the stored state carrying that result", () => {
    expect(deriveCalibrationScreenState({ latest: successQuery(RESULT), run: idleRun })).toEqual({
      kind: "stored",
      result: RESULT,
    })
  })
})

describe("deriveCalibrationScreenState -- running", () => {
  test("a pending run wins over whatever the query already knows", () => {
    expect(deriveCalibrationScreenState({ latest: successQuery(RESULT), run: pendingRun })).toEqual({
      kind: "running",
    })
  })
})

describe("deriveCalibrationScreenState -- after a run", () => {
  test("a successful run renders the success state carrying the fresh result", () => {
    const run: MutationLike<EchoCalibrationResult> = { status: "success", data: RESULT, error: undefined }
    expect(deriveCalibrationScreenState({ latest: successQuery(null), run })).toEqual({
      kind: "success",
      result: RESULT,
    })
  })
})

describe("deriveCalibrationScreenState -- failed, four named reasons", () => {
  test("a 403 on the run is 'disabled', carrying the server's detail, sourced from the run", () => {
    const detail = "echo-path calibration is disabled -- set calibration.route_enabled: true ..."
    const run: MutationLike<EchoCalibrationResult> = {
      status: "error",
      data: undefined,
      error: new ApiError(403, detail),
    }
    expect(deriveCalibrationScreenState({ latest: successQuery(null), run })).toEqual({
      kind: "failed",
      reason: "disabled",
      detail,
      source: "run",
    })
  })

  test("a 409 on the run is 'in-progress', carrying the server's detail", () => {
    const detail = "a calibration run is already in progress"
    const run: MutationLike<EchoCalibrationResult> = {
      status: "error",
      data: undefined,
      error: new ApiError(409, detail),
    }
    expect(deriveCalibrationScreenState({ latest: successQuery(null), run })).toMatchObject({
      kind: "failed",
      reason: "in-progress",
      detail,
    })
  })

  test("a 422 on the run is 'unmeasurable', carrying the run's own failure reason", () => {
    const detail = "no correlation peak above the usable confidence threshold"
    const run: MutationLike<EchoCalibrationResult> = {
      status: "error",
      data: undefined,
      error: new ApiError(422, detail),
    }
    expect(deriveCalibrationScreenState({ latest: successQuery(null), run })).toMatchObject({
      kind: "failed",
      reason: "unmeasurable",
      detail,
    })
  })

  test("a non-ApiError failure (a network error) is 'unknown', with no detail to show", () => {
    const run: MutationLike<EchoCalibrationResult> = {
      status: "error",
      data: undefined,
      error: new TypeError("Failed to fetch"),
    }
    expect(deriveCalibrationScreenState({ latest: successQuery(null), run })).toEqual({
      kind: "failed",
      reason: "unknown",
      detail: undefined,
      source: "run",
    })
  })

  test("a 403 on the initial GET is also 'disabled', sourced from 'latest' so the screen retries the GET", () => {
    const detail = "echo-path calibration is disabled -- set calibration.route_enabled: true ..."
    expect(
      deriveCalibrationScreenState({ latest: errorQuery(new ApiError(403, detail)), run: idleRun }),
    ).toEqual({
      kind: "failed",
      reason: "disabled",
      detail,
      source: "latest",
    })
  })
})
