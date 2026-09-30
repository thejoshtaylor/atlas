// 260929-j08: the inbox and the per-member clip list. `mock.module` replaces
// `@/lib/speakerInbox` before the components load, so a dynamic `import()` in
// each test keeps that order (the pattern of `SpeakersRoute.dom.test.tsx`).
import { afterEach, expect, mock, test } from "bun:test"
import { QueryClient, QueryClientProvider } from "@tanstack/react-query"
import { cleanup, fireEvent, render, screen, waitFor } from "@testing-library/react"

afterEach(() => {
  cleanup()
})

const speakers = [
  {
    id: 1,
    display_name: "Ann",
    linked_user_id: null,
    created_at: "2026-09-01T00:00:00Z",
    enrolled_phrases: 5,
    required_phrases: 5,
    model_id: "cam++",
    retroactive_clips: 0,
    can_control_home: true,
  },
]

function item(id: string, transcript: string | null) {
  return { session_id: id, started_at: "2026-09-29T10:00:00Z", transcript, speech_ms: 2000, score: 0.2, blocked: true }
}

function stubInbox(options: {
  items?: unknown[]
  clips?: unknown[]
  assign?: (input: unknown) => Promise<unknown>
  deleteClip?: (input: unknown) => Promise<unknown>
}) {
  mock.module("@/lib/speakerInbox", () => ({
    voiceInboxQueryOptions: {
      queryKey: ["speakers", "voice-inbox"],
      queryFn: async () => ({ items: options.items ?? [] }),
    },
    voiceInboxAudioUrl: (id: string) => `/audio/${id}`,
    assignVoiceMutationOptions: {
      mutationFn: options.assign ?? (async () => ({ speaker_id: 1, phrase_index: 100, speech_ms: 2000, dropped_phrase_indices: [] })),
    },
    retroactiveClipsQueryOptions: (speakerId: number) => ({
      queryKey: ["speakers", speakerId, "retroactive-clips"],
      queryFn: async () => ({ clips: options.clips ?? [] }),
    }),
    retroactiveClipAudioUrl: (speakerId: number, index: number) => `/clip/${speakerId}/${index}`,
    deleteRetroactiveClipMutationOptions: { mutationFn: options.deleteClip ?? (async () => undefined) },
  }))
}

function wrap(node: React.ReactElement) {
  const queryClient = new QueryClient({ defaultOptions: { queries: { retry: false } } })
  return render(<QueryClientProvider client={queryClient}>{node}</QueryClientProvider>)
}

test("two items show their transcript and an audio element with the audio URL", async () => {
  stubInbox({ items: [item("s1", "turn on the lamp"), item("s2", "what time is it")] })
  const { VoiceInbox } = await import("./VoiceInbox")

  const { container } = wrap(<VoiceInbox speakers={speakers} />)

  expect(await screen.findByText("turn on the lamp")).toBeTruthy()
  expect(screen.getByText("what time is it")).toBeTruthy()
  const sources = Array.from(container.querySelectorAll("audio")).map((audio) => audio.getAttribute("src"))
  expect(sources).toEqual(["/audio/s1", "/audio/s2"])
})

test("a transcript with angle brackets renders as literal text", async () => {
  stubInbox({ items: [item("s1", "<b>x</b>")] })
  const { VoiceInbox } = await import("./VoiceInbox")

  const { container } = wrap(<VoiceInbox speakers={speakers} />)

  expect(await screen.findByText("<b>x</b>")).toBeTruthy()
  expect(container.querySelector("b")).toBeNull()
})

test("an empty inbox says so", async () => {
  stubInbox({ items: [] })
  const { VoiceInbox } = await import("./VoiceInbox")

  wrap(<VoiceInbox speakers={speakers} />)

  expect(await screen.findByText("No unrecognized voices.")).toBeTruthy()
})

test("assigning to an existing member sends the session id and that member", async () => {
  const calls: unknown[] = []
  stubInbox({
    items: [item("s1", "turn on the lamp")],
    assign: async (input) => {
      calls.push(input)
      return { speaker_id: 1, phrase_index: 100, speech_ms: 2000, dropped_phrase_indices: [] }
    },
  })
  const { VoiceInbox } = await import("./VoiceInbox")

  wrap(<VoiceInbox speakers={speakers} />)

  await screen.findByText("turn on the lamp")
  fireEvent.click(screen.getByRole("button", { name: "Assign" }))
  fireEvent.click(screen.getByLabelText("Ann"))
  fireEvent.click(screen.getByRole("button", { name: "Assign voice" }))

  await waitFor(() => expect(calls).toEqual([{ sessionId: "s1", target: { kind: "existing", speakerId: 1 } }]))
})

test("assigning to a new member sends the typed name", async () => {
  const calls: unknown[] = []
  stubInbox({
    items: [item("s1", "turn on the lamp")],
    assign: async (input) => {
      calls.push(input)
      return { speaker_id: 2, phrase_index: 100, speech_ms: 2000, dropped_phrase_indices: [] }
    },
  })
  const { VoiceInbox } = await import("./VoiceInbox")

  wrap(<VoiceInbox speakers={speakers} />)

  await screen.findByText("turn on the lamp")
  fireEvent.click(screen.getByRole("button", { name: "Assign" }))
  fireEvent.click(screen.getByLabelText("New member"))
  fireEvent.change(screen.getByLabelText("Name"), { target: { value: "Guest" } })
  fireEvent.click(screen.getByRole("button", { name: "Assign voice" }))

  await waitFor(() => expect(calls).toEqual([{ sessionId: "s1", target: { kind: "new", displayName: "Guest" } }]))
})

test("a rejected assign shows the server's message", async () => {
  stubInbox({
    items: [item("s1", "turn on the lamp")],
    assign: async () => {
      const { ApiError } = await import("@/lib/api")
      throw new ApiError(409, "this turn is already assigned to a member")
    },
  })
  const { VoiceInbox } = await import("./VoiceInbox")

  wrap(<VoiceInbox speakers={speakers} />)

  await screen.findByText("turn on the lamp")
  fireEvent.click(screen.getByRole("button", { name: "Assign" }))
  fireEvent.click(screen.getByLabelText("Ann"))
  fireEvent.click(screen.getByRole("button", { name: "Assign voice" }))

  expect(await screen.findByText("this turn is already assigned to a member")).toBeTruthy()
})

test("a member with two clips lists two audio elements, and Remove deletes the first", async () => {
  const calls: unknown[] = []
  stubInbox({
    clips: [
      { phrase_index: 100, session_id: "s1", speech_ms: 2000, created_at: "2026-09-29T10:00:00Z" },
      { phrase_index: 101, session_id: "s2", speech_ms: 1500, created_at: "2026-09-29T11:00:00Z" },
    ],
    deleteClip: async (input) => {
      calls.push(input)
    },
  })
  const { RetroactiveClips } = await import("./RetroactiveClips")

  const { container } = wrap(<RetroactiveClips speaker={{ ...speakers[0], retroactive_clips: 2 }} />)

  await waitFor(() => expect(container.querySelectorAll("audio").length).toBe(2))
  fireEvent.click(screen.getAllByRole("button", { name: "Remove" })[0])

  await waitFor(() => expect(calls).toEqual([{ speakerId: 1, phraseIndex: 100 }]))
})
