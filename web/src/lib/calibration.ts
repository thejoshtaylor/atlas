// Typed access to the two routes Phase 2 already built --
// `GET /calibration/echo-path` and `POST /calibration/echo-path/run`
// (`src/spire_voice/app.py`, `_calibration_response`) -- through plan
// 03-03's fetch seam (`./api.ts`). There is one implementation of the
// echo-path measurement, and it is on the server; this module is a
// second caller of it, never a second measurement.
//
// Field names below are copied verbatim from `_calibration_response`'s
// dict keys, not renamed to a camelCase convention of this module's own
// invention -- a rename here would be a second contract to keep in sync
// with the route every time it changes.
//
// Every failure this module's two functions can produce carries the
// server's own `detail` string unmodified (`ApiError.message`, already
// un-paraphrased by `apiFetch` -- see `api.ts`'s own module docstring).
// This project writes those strings to be acted on: the disabled-route
// reason names the configuration key to set, and the run-failure reason
// is `run_echo_calibration`'s own account of why nothing could be
// measured (`calibration/runner.py`). Rewriting either in the browser
// throws away the thing that makes it actionable. The one caller-visible
// exception is `fetchLatestCalibration`'s 404: "no calibration has been
// taken yet" is not an error at all, so it is not raised as one -- it
// resolves to `null`, the state this project ships in.
import type { UseMutationOptions } from "@tanstack/react-query"
import { ApiError, apiFetch } from "./api"
import { queryClient } from "./queryClient"

/** `_calibration_response`'s exact shape (`app.py`). */
export interface EchoCalibrationResult {
  delay_s: number
  gain: number
  confidence: number
  agc_verdict: string
  source: string
  placement_note: string
  taken_at: string
  age_days: number
  echo_cancelled: boolean
}

export const CALIBRATION_QUERY_KEY = ["calibration", "echo-path"] as const

/**
 * The last stored calibration, or `null` when `GET /calibration/echo-path`
 * answers 404 -- "no echo-path calibration has been taken yet" is the
 * state this project ships in, not a failure, so it is not thrown.
 * Every other non-2xx (403 when `calibration.route_enabled` is off, a
 * network failure, ...) propagates as `ApiError`, carrying the server's
 * `detail` unmodified.
 */
export async function fetchLatestCalibration(): Promise<EchoCalibrationResult | null> {
  try {
    return await apiFetch<EchoCalibrationResult>("/calibration/echo-path")
  } catch (error) {
    if (error instanceof ApiError && error.status === 404) {
      return null
    }
    throw error
  }
}

export interface RunCalibrationInput {
  placementNote: string
}

/**
 * Runs a live echo-path calibration against the camera and speaker this
 * process already holds open -- `POST /calibration/echo-path/run`. Makes
 * a real house play a sound and record the room; never called except in
 * direct response to an explicit press (this module has no autorun of
 * its own).
 *
 * Three distinct non-2xx outcomes, each carrying the server's own
 * `detail` unmodified: 403 (the route is switched off, naming the
 * configuration key to set), 409 (a run is already in progress), 422
 * (the run completed and could not measure anything -- `detail` is
 * `run_echo_calibration`'s own `failure_reason`).
 */
export function runCalibration({ placementNote }: RunCalibrationInput): Promise<EchoCalibrationResult> {
  return apiFetch<EchoCalibrationResult>("/calibration/echo-path/run", {
    method: "POST",
    body: { placement_note: placementNote },
  })
}

/**
 * The run, as a TanStack Query mutation: retry stays off explicitly
 * (matching `queryClient.ts`'s own default for every mutation, restated
 * here rather than relied on implicitly, since a retried calibration run
 * would make the house play the probe sound a second time without a
 * second press) and a successful run replaces the cached "latest
 * calibration" query data in place, so the screen does not need a second
 * round trip to show what it just measured.
 */
export const runCalibrationMutationOptions: UseMutationOptions<
  EchoCalibrationResult,
  unknown,
  RunCalibrationInput
> = {
  mutationFn: runCalibration,
  retry: false,
  onSuccess: (result) => {
    queryClient.setQueryData(CALIBRATION_QUERY_KEY, result)
  },
}
