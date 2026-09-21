// Imports nothing from React (the same discipline every `derive*ScreenState`
// module in this project follows, `derivePluginsScreenState.ts`'s own module
// comment): what belongs here is only ever a fact about data, never a fact
// about a render. This module is DBG-05's central honesty requirement made
// concrete in the type system: `clearsPreviewedThreshold` (a hypothetical
// answer to "what would this threshold have done") and `historicalOutcome`
// (a fixed fact about what the gate actually decided at record time) are two
// separate fields on `WakeEventDisplay`, never collapsed into one status.
// Re-partitioning the events against a new threshold recomputes only the
// first field -- the second never changes, no matter where the control
// moves (T-08-34).
import { ApiError } from "@/lib/api"
import type { WakeEvent, WakeEventsResponse } from "@/lib/wakeTuning"

export interface QueryLike<T> {
  status: "pending" | "error" | "success"
  data: T | undefined
  error: unknown
}

/** What the gate actually decided at record time -- a historical fact no
 * re-partition can alter. `started_turn` is the one allowed outcome; the
 * three named blocks mirror `wake/gate.py::BlockReason`'s own three
 * reasons exactly, never a fourth invented here.
 *
 * `blocked_unknown_reason` is the fifth member and the point of this
 * type. `blocked_below_threshold` used to be a bare fallthrough, so a
 * fourth reason added to `BlockReason` later would have rendered as a
 * *wrong* third one, and a row written `allowed=false, block_reason=null`
 * (the column is nullable and nothing enforces the pairing) read as a
 * threshold decision that never happened. Fabricating a reason is exactly
 * what `deriveSessionsScreenState`'s own `summarizeSessionOutcome`
 * refuses to do two directories away. */
export type HistoricalOutcome =
  | "started_turn"
  | "blocked_below_threshold"
  | "blocked_refractory"
  | "blocked_media_playing"
  | "blocked_unknown_reason"

export interface WakeEventDisplay {
  id: number
  source: string
  recordedAt: string
  score: number | null
  /** Hypothetical: does this event's score clear the *previewed*
   * threshold right now. Recomputed on every threshold change. */
  clearsPreviewedThreshold: boolean
  /** Historical: what the gate actually decided. Fixed; moving the
   * control never changes this field (T-08-34). */
  historicalOutcome: HistoricalOutcome
  /** The server's own `block_reason` string, carried through unchanged so
   * the render can show an unrecognised one verbatim -- the same verbatim
   * fallback the Sessions list gives an unrecognised `turn_outcome`.
   * `null` for an allowed hit, and also for a blocked one the gate
   * recorded no reason against. */
  blockReason: string | null
}

export interface HistogramBucket {
  /** Inclusive lower bound of this bucket's score range, in [0, 1). */
  from: number
  /** Exclusive upper bound (1.0 for the last bucket, which is closed on
   * both ends so a score of exactly 1.0 has a bucket to land in). */
  to: number
  count: number
  /** How many of `count` clear the previewed threshold, and how many do
   * not. Two numbers, not one `clears` boolean: the threshold almost
   * always falls *inside* a bucket rather than on a boundary, and no
   * single flag is true of that bucket. The old flag asked whether the
   * bucket's lower bound cleared, which made the bar holding a clearing
   * event render as "does not clear" and put the histogram -- DBG-05's
   * focal element -- in direct contradiction with the count line beside
   * it. The straddling bucket is split instead, which is the only answer
   * that is true of every event in it (08-UI-SPEC.md's lightness-not-hue
   * split applies to each part). */
  clearingCount: number
  belowCount: number
}

export type WakeTuningScreenState =
  | { kind: "loading" }
  | { kind: "error"; message: string }
  | { kind: "empty" }
  | {
      kind: "ready"
      engine: string
      engineGrades: boolean
      /** Sums with `belowCount` to `totalScored` -- never includes
       * `notScoredCount`, which is reported entirely separately (D-16). */
      clearingCount: number
      belowCount: number
      totalScored: number
      notScoredCount: number
      /** The server had more wake attempts than it would send. The partition
       * above is of the newest ones only, and the screen says so -- the same
       * reason `notScoredCount` is reported separately rather than folded
       * away (D-16). */
      capped: boolean
      histogram: HistogramBucket[]
      /** Newest first, matching the server's own ordering -- unchanged
       * from `response.events`. */
      events: WakeEventDisplay[]
    }

const HISTOGRAM_BUCKET_COUNT = 10

function messageFor(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "Couldn't load wake history. Try again."
}

/**
 * The one comparison this whole screen exists to never disagree with
 * `wake/gate.py::WakeGate.evaluate`'s own boundary (`if score <
 * self._threshold: block(...)`, i.e. a score exactly equal to the
 * threshold is a hit) -- `>=`, matching
 * `scripts/score_wake_engines.py::detections_at_threshold`'s own
 * inclusive comparison. A `null` score (not yet scored, D-16's
 * pre-migration case) neither clears nor fails: it returns `false` here
 * and the caller below excludes it from both counts rather than reading
 * this `false` as "below."
 */
export function clearsThreshold(score: number | null, threshold: number): boolean {
  return score !== null && score >= threshold
}

function historicalOutcomeFor(event: WakeEvent): HistoricalOutcome {
  if (event.allowed) return "started_turn"
  if (event.block_reason === "below_threshold") return "blocked_below_threshold"
  if (event.block_reason === "refractory") return "blocked_refractory"
  if (event.block_reason === "media_playing") return "blocked_media_playing"
  return "blocked_unknown_reason"
}

/**
 * Ten equal-width buckets covering [0, 1] with no overlap and no gap: a
 * score in `[i/10, (i+1)/10)` belongs to bucket `i`; a score of exactly
 * `1.0` is clamped into the last bucket rather than falling off the top
 * edge. A `null` score (not yet scored) is excluded, matching the live
 * count's own denominator.
 */
function buildHistogram(events: readonly WakeEvent[], threshold: number): HistogramBucket[] {
  const buckets: HistogramBucket[] = []
  for (let index = 0; index < HISTOGRAM_BUCKET_COUNT; index += 1) {
    const from = index / HISTOGRAM_BUCKET_COUNT
    const to = (index + 1) / HISTOGRAM_BUCKET_COUNT
    buckets.push({ from, to, count: 0, clearingCount: 0, belowCount: 0 })
  }
  for (const event of events) {
    if (event.score === null) continue
    let index = Math.floor(event.score * HISTOGRAM_BUCKET_COUNT)
    if (index >= HISTOGRAM_BUCKET_COUNT) index = HISTOGRAM_BUCKET_COUNT - 1
    if (index < 0) index = 0
    buckets[index].count += 1
    // Asked of the event's own score, through the same `clearsThreshold`
    // the count line uses -- so a bar and the sentence under it can never
    // disagree about the same event.
    if (clearsThreshold(event.score, threshold)) buckets[index].clearingCount += 1
    else buckets[index].belowCount += 1
  }
  return buckets
}

/**
 * Recomputes the whole partition from the already-held event list and
 * the currently previewed `threshold` -- no argument here is, or could
 * be, a network result. A threshold change re-runs this function against
 * the same `query.data` the initial fetch produced; no second
 * `apiFetch` call exists anywhere in this module (08-PATTERNS.md's
 * no-network-call-per-drag discipline).
 */
export function deriveWakeTuningScreenState(
  query: QueryLike<WakeEventsResponse>,
  threshold: number,
): WakeTuningScreenState {
  if (query.status === "pending") return { kind: "loading" }
  if (query.status === "error") return { kind: "error", message: messageFor(query.error) }
  const response = query.data
  if (!response) return { kind: "loading" }
  if (response.events.length === 0) return { kind: "empty" }

  const events: WakeEventDisplay[] = response.events.map((event) => ({
    id: event.id,
    source: event.source,
    recordedAt: event.recorded_at,
    score: event.score,
    clearsPreviewedThreshold: clearsThreshold(event.score, threshold),
    historicalOutcome: historicalOutcomeFor(event),
    blockReason: event.block_reason,
  }))

  const scoredEvents = response.events.filter((event) => event.score !== null)
  const clearingCount = scoredEvents.filter((event) => clearsThreshold(event.score, threshold)).length
  const belowCount = scoredEvents.length - clearingCount

  return {
    kind: "ready",
    engine: response.engine,
    engineGrades: response.engine_grades,
    clearingCount,
    belowCount,
    totalScored: scoredEvents.length,
    notScoredCount: response.not_scored_session_count,
    capped: response.capped,
    histogram: buildHistogram(response.events, threshold),
    events,
  }
}
