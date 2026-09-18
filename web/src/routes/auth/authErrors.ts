// Pulled out of SignInRoute/AcceptInviteRoute so the "a wrong email and a
// wrong password read the same" property (this plan's own acceptance
// criteria) is testable without a DOM -- this project has no
// rendered-component test infrastructure (see CalibrationRoute's derive
// module, 03-06, for the established pattern this file copies).
import { ApiError } from "@/lib/api"

/**
 * `_invalid_credentials_error` (`routes/auth.py`) already answers a
 * 401 identically for an unknown email, a wrong password, and a disabled
 * account -- this function does not re-introduce a distinction the
 * server deliberately removed. Every 401 renders the Copywriting
 * Contract's exact text, regardless of whatever `detail` string the
 * server happened to send.
 */
export function classifyLoginError(error: unknown): string {
  if (error instanceof ApiError && error.status === 401) {
    return "Wrong email or password. Try again."
  }
  return error instanceof Error ? error.message : "Sign-in failed."
}

/**
 * `_invite_invalid_error` (`routes/accounts.py`) already answers one
 * refusal for "not found," "expired," and "already used" -- this
 * function surfaces that server detail unmodified rather than
 * re-deriving a guess at which of the three happened.
 */
export function classifyAcceptInviteError(error: unknown): string {
  if (error instanceof ApiError) return error.message
  return "This invite is invalid, expired, or already used."
}
