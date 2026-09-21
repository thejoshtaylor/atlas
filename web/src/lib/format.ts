// Formatters shared by more than one screen. Imports nothing -- not
// React, not a route, not a fetch module -- so any screen can reach one
// without pulling a neighbour's module graph in behind it.
//
// `formatMs` lived in `routes/dev-mic/DevMicRoute.tsx` and was imported
// from there by `routes/sessions/deriveSessionsScreenState.ts`, a module
// whose own header says it imports nothing from React. `DevMicRoute.tsx`
// imports React, the transports module and the audio-worklet plumbing, so
// the Sessions screen transitively depended on the developer-microphone
// route and loaded its graph whenever `/sessions` did. Reusing one
// formatter was right; reaching into a route component to get it was not.

/** A duration in milliseconds, to one decimal place -- or "not reached"
 * for a stage mark a turn never got to. The "not reached" branch is the
 * reason this is one function and not a template literal per call site:
 * it composes into every sentence that quotes a duration (the Sessions
 * list's "not reached end to end", the developer page's per-stage rows),
 * so a stage that never happened reads the same way everywhere. */
export function formatMs(ms: number | null | undefined): string {
  return ms === null || ms === undefined ? "not reached" : `${ms.toFixed(1)} ms`
}

/** A duration in seconds, to one decimal place -- or "not recorded" for a
 * session with no reading. A sibling of `formatMs`, not a call to it: the
 * pre-roll is a whole-seconds-scale fact about a recording an operator is
 * about to hear, and quoting it in milliseconds would read as a latency
 * measurement, which it is not (plan 08-12, DBG-03). */
export function formatSeconds(seconds: number | null | undefined): string {
  return seconds === null || seconds === undefined ? "not recorded" : `${seconds.toFixed(1)} s`
}
