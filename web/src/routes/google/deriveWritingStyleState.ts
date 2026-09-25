// WritingStyleSection's own logic, kept apart from its JSX -- copied in
// shape from `deriveGoogleAccountState.ts`. Imports nothing from React.
import type { GoogleStyle, GoogleStyleStatus } from "@/lib/google"

/** The one status sentence for each of `StyleResponse.status`'s four
 * values (09-11-PLAN.md Task 1 `<behavior>`). Never a second, paraphrased
 * copy at the call site. */
export function deriveWritingStyleState(
  style: Pick<GoogleStyle, "status" | "status_detail" | "messages_scanned" | "learned_at">,
): string {
  switch (style.status) {
    case "not_learned":
      return "Not learned yet."
    case "learning":
      return "Learning from your Sent mail..."
    case "ready": {
      const date = style.learned_at ? new Date(style.learned_at).toLocaleDateString() : ""
      return `Learned from ${style.messages_scanned} sent messages on ${date}.`
    }
    case "failed":
      return `Could not learn the style: ${style.status_detail ?? ""}`
  }
}

/** Only the `learning` status polls -- every other status is a resting
 * state with nothing new to fetch. */
export function shouldPoll(status: GoogleStyleStatus): boolean {
  return status === "learning"
}
