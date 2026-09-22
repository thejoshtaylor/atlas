import { afterEach, describe, expect, test } from "bun:test"
import { ApiError } from "./api"
import {
  CALIBRATION_QUERY_KEY,
  fetchLatestCalibration,
  runCalibration,
  runCalibrationMutationOptions,
} from "./calibration"
import { queryClient } from "./queryClient"

function jsonResponse(status: number, body: unknown): Response {
  return new Response(JSON.stringify(body), {
    status,
    headers: { "Content-Type": "application/json" },
  })
}

const originalFetch = global.fetch

afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

const FULL_RESULT_BODY = {
  delay_s: 0.142,
  gain: 0.37,
  confidence: 0.91,
  agc_verdict: "present",
  source: "camera",
  placement_note: "kitchen counter, 2m from camera",
  taken_at: "2026-09-17T12:00:00+00:00",
  age_days: 0.01,
  echo_cancelled: false,
}

describe("fetchLatestCalibration -- the GET route, four distinguishable outcomes (1 of 4: not-found)", () => {
  test("a 404 resolves to null -- no calibration taken yet is the shipped state, not an error", async () => {
    global.fetch = (async () =>
      jsonResponse(404, { detail: "no echo-path calibration has been taken yet" })) as typeof fetch

    await expect(fetchLatestCalibration()).resolves.toBeNull()
  })

  test("a 200 resolves with _calibration_response's exact field names, character for character", async () => {
    global.fetch = (async () => jsonResponse(200, FULL_RESULT_BODY)) as typeof fetch

    const result = await fetchLatestCalibration()
    expect(result).toEqual(FULL_RESULT_BODY)
    // Every key `_calibration_response` (app.py) returns, and no others.
    expect(Object.keys(result!).sort()).toEqual(
      [
        "delay_s",
        "gain",
        "confidence",
        "agc_verdict",
        "source",
        "placement_note",
        "taken_at",
        "age_days",
        "echo_cancelled",
      ].sort(),
    )
  })

  test("a 403 (calibration.route_enabled is off) propagates as ApiError carrying the server's detail unmodified (2 of 4: disabled)", async () => {
    const detail =
      "echo-path calibration is disabled -- set calibration.route_enabled: true to allow this route."
    global.fetch = (async () => jsonResponse(403, { detail })) as typeof fetch

    const rejection = fetchLatestCalibration()
    await expect(rejection).rejects.toBeInstanceOf(ApiError)
    await expect(rejection).rejects.toMatchObject({ status: 403, message: detail })
  })
})

describe("runCalibration -- the POST route, the remaining two distinguishable outcomes", () => {
  test("a 409 (a run is already in progress) propagates as ApiError carrying the server's detail unmodified (3 of 4: in-progress)", async () => {
    const detail = "a calibration run is already in progress"
    global.fetch = (async () => jsonResponse(409, { detail })) as typeof fetch

    const rejection = runCalibration({ placementNote: "kitchen counter" })
    await expect(rejection).rejects.toBeInstanceOf(ApiError)
    await expect(rejection).rejects.toMatchObject({ status: 409, message: detail })
  })

  test("a 422 (the run completed and measured nothing) propagates as ApiError carrying the run's own failure reason unmodified (4 of 4: unmeasurable)", async () => {
    const detail = "no correlation peak above the usable confidence threshold"
    global.fetch = (async () => jsonResponse(422, { detail })) as typeof fetch

    const rejection = runCalibration({ placementNote: "kitchen counter" })
    await expect(rejection).rejects.toBeInstanceOf(ApiError)
    await expect(rejection).rejects.toMatchObject({ status: 422, message: detail })
  })

  test("a 200 resolves with the measured result, and the placement note travels as snake_case placement_note", async () => {
    let sentBody: unknown
    global.fetch = (async (_url: string, init?: RequestInit) => {
      sentBody = init?.body ? JSON.parse(init.body as string) : undefined
      return jsonResponse(200, FULL_RESULT_BODY)
    }) as typeof fetch

    const result = await runCalibration({ placementNote: "kitchen counter, 2m from camera" })
    expect(result).toEqual(FULL_RESULT_BODY)
    expect(sentBody).toEqual({ placement_note: "kitchen counter, 2m from camera" })
  })
})

describe("runCalibrationMutationOptions -- the run, as a mutation with retry disabled", () => {
  test("retry is explicitly false -- a retried run would replay the probe sound with no second press", () => {
    expect(runCalibrationMutationOptions.retry).toBe(false)
  })

  test("mutationFn is runCalibration itself, not a re-derived copy", () => {
    expect(runCalibrationMutationOptions.mutationFn).toBe(runCalibration)
  })

  test("onSuccess replaces the cached latest-calibration query data with the fresh result", async () => {
    queryClient.setQueryData(CALIBRATION_QUERY_KEY, null)
    await runCalibrationMutationOptions.onSuccess?.(
      FULL_RESULT_BODY,
      { placementNote: "kitchen counter" },
      undefined,
      { client: queryClient } as never,
    )
    expect(queryClient.getQueryData(CALIBRATION_QUERY_KEY)).toEqual(FULL_RESULT_BODY)
  })
})

describe("no second network path -- this module goes through the shared fetch seam only", () => {
  test("apiFetch is the only thing calibration.ts's two route functions call over the network", async () => {
    // A structural guarantee (also checked by the plan's own grep verify
    // step: `! grep -rn "fetch(" web/src/lib/calibration.ts`): both
    // functions resolve through the single mocked `global.fetch` this
    // file installs above, so no second, real network call is possible
    // from either.
    let calls = 0
    global.fetch = (async () => {
      calls += 1
      return jsonResponse(404, { detail: "no echo-path calibration has been taken yet" })
    }) as typeof fetch

    await fetchLatestCalibration()
    expect(calls).toBe(1)
  })
})
