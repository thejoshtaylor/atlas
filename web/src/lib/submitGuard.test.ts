import { describe, expect, test } from "bun:test"
import { createSubmitGuard } from "./submitGuard"

describe("createSubmitGuard -- T-03-14: a second submit cannot be issued while the first is in flight", () => {
  test("firing two submits back-to-back, with no await between them, issues exactly one call", async () => {
    let calls = 0
    let resolveFirst: (() => void) | undefined
    const guard = createSubmitGuard(async () => {
      calls += 1
      await new Promise<void>((resolve) => {
        resolveFirst = resolve
      })
    })

    // Both fired in the same tick -- the second must be dropped by the
    // guard's synchronous check, not merely "usually" win a race.
    const first = guard.run()
    const second = guard.run()

    expect(guard.isPending()).toBe(true)
    expect(calls).toBe(1)

    resolveFirst?.()
    await Promise.all([first, second])

    expect(calls).toBe(1)
    expect(guard.isPending()).toBe(false)
  })

  test("a submit after the first has settled is allowed to run again", async () => {
    let calls = 0
    const guard = createSubmitGuard(async () => {
      calls += 1
    })

    await guard.run()
    await guard.run()

    expect(calls).toBe(2)
  })
})
