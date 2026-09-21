// D-17 backfill (08-09-PLAN.md). Every loading state in this phase and
// the last three depends on `SkeletonList` rendering something rather
// than nothing -- the whole reason a skeleton is preferred over a
// momentarily empty list on a card list (03-UI-SPEC.md's UI
// Considerations table). Proven here by counting the rendered rows.

import { afterEach, expect, test } from "bun:test"
import { cleanup, render, screen } from "@testing-library/react"

import { SkeletonList } from "./SkeletonList"

afterEach(() => {
  cleanup()
})

test("renders exactly as many placeholder rows as the rows prop asks for", () => {
  const { container } = render(<SkeletonList rows={5} />)

  expect(screen.getByRole("status", { name: "Loading" })).toBeTruthy()
  expect(container.querySelectorAll('[data-slot="skeleton"]').length).toBe(5)
})

test("defaults to three rows when none is given", () => {
  const { container } = render(<SkeletonList />)

  expect(container.querySelectorAll('[data-slot="skeleton"]').length).toBe(3)
})
