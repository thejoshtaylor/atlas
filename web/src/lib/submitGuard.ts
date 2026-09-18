// The re-entrancy guard `components/state/SubmitButton.tsx` wraps around
// every submitting control's handler. Framework-free and independently
// testable (`submitGuard.test.ts`) so the double-submit property --
// "a second submit cannot be issued while the first is in flight" (T-03-14)
// -- is verified without rendering a component or simulating a click.
//
// A plain boolean flag checked synchronously, not React state: two calls
// arriving in the same tick (a fast double-click, or a test firing two
// calls back-to-back with no `await` between them) must not both pass the
// check, and React state updates are batched/asynchronous, so a `useState`
// flag alone cannot make that guarantee.
export function createSubmitGuard<Args extends unknown[]>(
  fn: (...args: Args) => Promise<unknown>,
): {
  run: (...args: Args) => Promise<unknown>
  isPending: () => boolean
} {
  let pending = false

  return {
    isPending: () => pending,
    run: async (...args: Args) => {
      if (pending) return undefined
      pending = true
      try {
        return await fn(...args)
      } finally {
        pending = false
      }
    },
  }
}
