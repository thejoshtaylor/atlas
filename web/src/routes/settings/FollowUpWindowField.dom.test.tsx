// Plan 09-07 Task 2. `mock.module` replaces `@/lib/followUp` before
// `FollowUpWindowField` is imported -- a dynamic `import()` inside the
// test body is what makes that ordering hold
// (`PluginsRoute.dom.test.tsx`'s established pattern).
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"

import type * as React from "react"

afterEach(() => {
  cleanup()
})

function stubFollowUp(
  queryClient: QueryClient,
  options: {
    fetchFollowUpWindow: () => Promise<unknown>
    saveFollowUpWindow?: (input: unknown) => Promise<unknown>
  },
) {
  mock.module("@/lib/followUp", () => ({
    FOLLOW_UP_WINDOW_QUERY_KEY: ["settings", "follow-up-window"],
    fetchFollowUpWindow: options.fetchFollowUpWindow,
    followUpWindowQueryOptions: {
      queryKey: ["settings", "follow-up-window"],
      queryFn: options.fetchFollowUpWindow,
    },
    saveFollowUpWindowMutationOptions: {
      mutationFn:
        options.saveFollowUpWindow ??
        (async () => {
          throw new Error("saveFollowUpWindow is not stubbed in this test")
        }),
      onSuccess: (status: unknown) => {
        queryClient.setQueryData(["settings", "follow-up-window"], status)
      },
    },
  }))
}

function renderField(FollowUpWindowField: React.ComponentType, queryClient: QueryClient) {
  return render(
    <QueryClientProvider client={queryClient}>
      <FollowUpWindowField />
    </QueryClientProvider>,
  )
}

test("renders the current value from the server", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  stubFollowUp(queryClient, {
    fetchFollowUpWindow: async () => ({ window_s: 6, default_s: 6, resolved_from: "config" }),
  })
  const { FollowUpWindowField } = await import("./FollowUpWindowField")

  renderField(FollowUpWindowField, queryClient)

  expect(await screen.findByText("Answer window")).toBeTruthy()
  const input = (await screen.findByLabelText("Seconds")) as HTMLInputElement
  expect(input.value).toBe("6")
})

test("saving sends the edited value and the field reflects the saved response", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const calls: unknown[] = []
  stubFollowUp(queryClient, {
    fetchFollowUpWindow: async () => ({ window_s: 6, default_s: 6, resolved_from: "config" }),
    saveFollowUpWindow: async (input) => {
      calls.push(input)
      return { window_s: 8, default_s: 6, resolved_from: "database" }
    },
  })
  const { FollowUpWindowField } = await import("./FollowUpWindowField")

  renderField(FollowUpWindowField, queryClient)
  const input = (await screen.findByLabelText("Seconds")) as HTMLInputElement
  fireEvent.change(input, { target: { value: "8" } })

  fireEvent.click(screen.getByRole("button", { name: "Save answer window" }))

  await waitFor(() => expect(calls).toEqual([{ window_s: 8 }]))
})

test("a save failure shows the server's own refusal text verbatim", async () => {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  const { ApiError } = await import("@/lib/api")
  stubFollowUp(queryClient, {
    fetchFollowUpWindow: async () => ({ window_s: 6, default_s: 6, resolved_from: "config" }),
    saveFollowUpWindow: async () => {
      throw new ApiError(400, "the follow-up window must be a number between 3 and 15 seconds")
    },
  })
  const { FollowUpWindowField } = await import("./FollowUpWindowField")

  renderField(FollowUpWindowField, queryClient)
  const input = (await screen.findByLabelText("Seconds")) as HTMLInputElement
  fireEvent.change(input, { target: { value: "20" } })
  fireEvent.click(screen.getByRole("button", { name: "Save answer window" }))

  expect(
    await screen.findByText("the follow-up window must be a number between 3 and 15 seconds"),
  ).toBeTruthy()
})
