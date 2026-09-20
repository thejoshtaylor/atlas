// The first rendered-DOM test in this project's history (D-17,
// 08-01-PLAN.md). Every prior `web/` test asserts over source text or
// plain objects; this one mounts a real component into happy-dom
// (registered via `web/happydom.ts`, preloaded per `web/bunfig.toml`)
// and drives it with a real DOM event, proving the harness wiring works
// end to end against a component this repository already ships.

import { afterEach, expect, test } from "bun:test"
import { cleanup, fireEvent, render, screen } from "@testing-library/react"

import { ErrorState } from "./ErrorState"

afterEach(() => {
  cleanup()
})

test("renders the message and a Retry button that calls onRetry exactly once when clicked", () => {
  let calls = 0
  render(<ErrorState message="Could not load sessions" onRetry={() => calls++} />)

  expect(screen.getByText("Could not load sessions")).toBeTruthy()
  const button = screen.getByText("Retry")
  fireEvent.click(button)

  expect(calls).toBe(1)
})

test("renders message and detail but no button at all when onRetry is absent", () => {
  render(<ErrorState message="Session removed" detail="Removed by the retention sweep" />)

  expect(screen.getByText("Session removed")).toBeTruthy()
  expect(screen.getByText("Removed by the retention sweep")).toBeTruthy()
  expect(screen.queryByRole("button")).toBeNull()
})

test("renders a custom retryLabel on the button instead of the default", () => {
  render(<ErrorState message="Could not load" onRetry={() => {}} retryLabel="Try again" />)

  expect(screen.getByText("Try again")).toBeTruthy()
  expect(screen.queryByText("Retry")).toBeNull()
})

test("with onRetry present, the button still renders (unchanged behavior)", () => {
  render(<ErrorState message="Could not load" onRetry={() => {}} />)

  expect(screen.getByRole("button")).toBeTruthy()
})
