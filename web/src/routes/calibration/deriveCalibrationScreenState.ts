// The calibration screen's own logic, kept apart from its JSX so it is
// testable without a DOM (this project has no rendered-component test
// infrastructure yet -- see this plan's SUMMARY, Deviations). Given the
// two pieces of state the screen reads (the last stored calibration, and
// the in-flight run), this module decides which of the design contract's
// four states to show and, for a failed run, which of its four named
// reasons applies -- so `CalibrationRoute.tsx` never branches on a raw
// HTTP status itself.
import { ApiError } from "@/lib/api"
import type { EchoCalibrationResult } from "@/lib/calibration"

/** The shape this module needs from a TanStack Query `useQuery` result --
 * a subset, so this file depends on no React Query type directly and can
 * be driven by a hand-built object in a test. */
export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

/** The shape this module needs from a TanStack Query `useMutation` result. */
export interface MutationLike<T> {
  status: "idle" | "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

/**
 * The three failures that are not a failed measurement each get their
 * own treatment (this plan's Task 2 objective); "unknown" is the
 * fallback for a failure that named none of them (a network error, an
 * unexpected 5xx) and is the one case with no server-written reason to
 * show.
 */
export type CalibrationFailureReason = "disabled" | "in-progress" | "unmeasurable" | "unknown"

export type CalibrationScreenState =
  | { kind: "loading" }
  | { kind: "empty" }
  | { kind: "stored"; result: EchoCalibrationResult }
  | { kind: "running" }
  | { kind: "success"; result: EchoCalibrationResult }
  | {
      kind: "failed"
      reason: CalibrationFailureReason
      /** The server's own `detail` string, verbatim -- absent only for "unknown". */
      detail: string | undefined
      /** Which request failed, so the screen retries the right one. */
      source: "latest" | "run"
    }

const STATUS_TO_REASON: Partial<Record<number, CalibrationFailureReason>> = {
  403: "disabled",
  409: "in-progress",
  422: "unmeasurable",
}

function classify(error: unknown): { reason: CalibrationFailureReason; detail: string | undefined } {
  if (error instanceof ApiError) {
    const reason = STATUS_TO_REASON[error.status] ?? "unknown"
    return { reason, detail: reason === "unknown" ? undefined : error.message }
  }
  return { reason: "unknown", detail: undefined }
}

/**
 * The mutation (a run in flight, or one that just settled) always wins
 * over the query (the last stored calibration): a run replaces what was
 * known before it, and its own pending/error/success state is what the
 * operator is looking at while it is happening. Once the mutation is
 * back to `idle` (never run this visit, or after a fresh page load),
 * the screen falls back to whatever the query already knows.
 */
export function deriveCalibrationScreenState(input: {
  latest: QueryLike<EchoCalibrationResult | null>
  run: MutationLike<EchoCalibrationResult>
}): CalibrationScreenState {
  const { latest, run } = input

  if (run.status === "pending") return { kind: "running" }
  if (run.status === "error") {
    const { reason, detail } = classify(run.error)
    return { kind: "failed", reason, detail, source: "run" }
  }
  if (run.status === "success" && run.data !== undefined) {
    return { kind: "success", result: run.data }
  }

  if (latest.status === "pending") return { kind: "loading" }
  if (latest.status === "error") {
    const { reason, detail } = classify(latest.error)
    return { kind: "failed", reason, detail, source: "latest" }
  }
  if (latest.data) return { kind: "stored", result: latest.data }
  return { kind: "empty" }
}
