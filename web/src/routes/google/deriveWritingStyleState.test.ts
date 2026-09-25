// 09-11-PLAN.md Task 1. No React import -- see the module's own header.
import { describe, expect, test } from "bun:test"
import { deriveWritingStyleState, shouldPoll } from "./deriveWritingStyleState"

describe("deriveWritingStyleState", () => {
  test("not_learned shows 'Not learned yet.'", () => {
    expect(
      deriveWritingStyleState({ status: "not_learned", status_detail: null, messages_scanned: 0, learned_at: null }),
    ).toBe("Not learned yet.")
  })

  test("learning shows 'Learning from your Sent mail...'", () => {
    expect(
      deriveWritingStyleState({ status: "learning", status_detail: null, messages_scanned: 0, learned_at: null }),
    ).toBe("Learning from your Sent mail...")
  })

  test("ready shows the message count and the learned date", () => {
    const sentence = deriveWritingStyleState({
      status: "ready",
      status_detail: null,
      messages_scanned: 187,
      learned_at: "2026-09-20T00:00:00Z",
    })
    expect(sentence).toContain("187")
    expect(sentence).toContain("Learned from")
    expect(sentence).toContain("sent messages")
  })

  test("failed shows the server's status_detail verbatim", () => {
    const sentence = deriveWritingStyleState({
      status: "failed",
      status_detail: "no Sent messages found",
      messages_scanned: 0,
      learned_at: null,
    })
    expect(sentence).toBe("Could not learn the style: no Sent messages found")
  })
})

describe("shouldPoll", () => {
  test("polls only while learning", () => {
    expect(shouldPoll("learning")).toBe(true)
    expect(shouldPoll("not_learned")).toBe(false)
    expect(shouldPoll("ready")).toBe(false)
    expect(shouldPoll("failed")).toBe(false)
  })
})
