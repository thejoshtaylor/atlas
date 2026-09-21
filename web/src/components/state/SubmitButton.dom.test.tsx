// D-17 backfill (08-09-PLAN.md). `SubmitButton` is the single most
// load-bearing shared interaction in this application -- every
// submitting form (the wizard steps, sign-in, sending an invite, saving
// a credential, and every save in this phase) routes through it, and
// its own docstring names the reason a double write must be impossible:
// a stored credential is never readable back, so an ambiguous double
// write cannot be inspected afterward (T-03-14). Until now that
// guarantee rested on a unit test of the guard function
// (`submitGuard.test.ts`) plus a regex confirming the component imports
// it -- neither proves the button itself honours the guard. This mounts
// it and drives real clicks.

import { afterEach, expect, test } from "bun:test"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"

import { SubmitButton } from "./SubmitButton"

afterEach(() => {
  cleanup()
})

/** A promise the test resolves/rejects by hand, so the pending window is
 * under the test's control rather than a timer's. */
function deferred<T>() {
  let resolve!: (value: T) => void
  let reject!: (reason?: unknown) => void
  const promise = new Promise<T>((res, rej) => {
    resolve = res
    reject = rej
  })
  return { promise, resolve, reject }
}

test("a rapid double click calls the handler once, disables the button, and shows the pending label", () => {
  const { promise } = deferred<void>()
  let calls = 0
  render(
    <SubmitButton onSubmit={() => { calls++; return promise }} pendingLabel="Saving…">
      Save
    </SubmitButton>,
  )

  const button = screen.getByRole("button", { name: "Save" })
  fireEvent.click(button)
  fireEvent.click(button)

  expect(calls).toBe(1)
  expect((button as HTMLButtonElement).disabled).toBe(true)
  expect(button.getAttribute("aria-busy")).toBe("true")
  expect(screen.getByText("Saving…")).toBeTruthy()
})

test("once the promise resolves, the button re-enables and a further click calls the handler again", async () => {
  const first = deferred<void>()
  let calls = 0
  const onSubmit = () => {
    calls += 1
    return calls === 1 ? first.promise : Promise.resolve()
  }
  render(<SubmitButton onSubmit={onSubmit}>Save</SubmitButton>)

  const button = screen.getByRole("button", { name: "Save" })
  fireEvent.click(button)
  expect((button as HTMLButtonElement).disabled).toBe(true)

  first.resolve()
  await screen.findByText("Save")
  expect((button as HTMLButtonElement).disabled).toBe(false)

  fireEvent.click(button)
  expect(calls).toBe(2)
})

test("a rejected save re-enables the button rather than leaving it stuck", async () => {
  const { promise, reject } = deferred<void>()

  render(<SubmitButton onSubmit={() => promise}>Save</SubmitButton>)

  const button = screen.getByRole("button", { name: "Save" })
  fireEvent.click(button)
  expect((button as HTMLButtonElement).disabled).toBe(true)

  reject(new Error("boom"))
  await screen.findByText("Save")
  expect((screen.getByRole("button", { name: "Save" }) as HTMLButtonElement).disabled).toBe(false)
})
