// The wizard's own authenticated surface: the five-step read (`GET
// /api/wizard`) and the per-step completion mutations
// (`routes/wizard.py`, plan 03-09). The browser never marks a step
// complete itself -- every mutation here asks the server, and the
// component that calls it renders whatever came back, including a
// refusal. `web/src/lib/setup.ts` is the *unauthenticated* sibling this
// module never duplicates (the status route's own `steps` are booleans
// only; this module's are the fuller `detail`-carrying kind an
// authenticated admin actually acts on).
import type { UseMutationOptions } from "@tanstack/react-query"
import { ApiError, apiFetch } from "./api"
import { queryClient } from "./queryClient"
import { SETUP_STATUS_QUERY_KEY } from "./setup"
import { SESSION_QUERY_KEY, type Session } from "./session"

/** `STEP_ORDER` (`routes/wizard.py`) -- the five step names, reproduced
 * here as a typed union rather than a bare `string` so a typo in a step
 * name is a compile error, not a silent no-match at runtime. This is
 * presentation order only, mirroring what the server already returns; it
 * is never itself the authority on completion. */
export type WizardStepName = "admin_account" | "hub" | "provider_set" | "audio_source" | "room"

export const WIZARD_STEP_ORDER: readonly WizardStepName[] = [
  "admin_account",
  "hub",
  "provider_set",
  "audio_source",
  "room",
]

/** Display-size titles naming what each step is for (03-UI-SPEC.md's
 * Focal Point row for the wizard: "a title at Display size naming what
 * this step is for"). The room step reuses `CalibrationRoute`'s own
 * heading instead of this one -- see `RoomStep.tsx`. */
export const WIZARD_STEP_TITLES: Record<WizardStepName, string> = {
  admin_account: "Create the admin account",
  hub: "Connect Home Assistant",
  provider_set: "Add your provider keys",
  audio_source: "Choose the audio source",
  room: "Test the microphone and speaker",
}

/** `WizardStepStatus`'s exact shape (`src/spire_voice/routes/wizard.py`). */
export interface WizardStepStatus {
  name: WizardStepName
  complete: boolean
  detail: Record<string, unknown> | null
}

/** `WizardStatusResponse`'s exact shape. */
export interface WizardStatus {
  steps: WizardStepStatus[]
  first_unfinished_step: WizardStepName | null
}

export const WIZARD_QUERY_KEY = ["wizard"] as const

export function fetchWizardStatus(): Promise<WizardStatus> {
  return apiFetch<WizardStatus>("/api/wizard")
}

export const wizardStatusQueryOptions = {
  queryKey: WIZARD_QUERY_KEY,
  queryFn: fetchWizardStatus,
}

// --- Create the admin account -------------------------------------------

export interface CreateAdminInput {
  email: string
  display_name: string
  password: string
}

function createAdmin(input: CreateAdminInput): Promise<Session> {
  return apiFetch<Session>("/api/auth/create-admin", { method: "POST", body: input })
}

/**
 * `POST /api/auth/create-admin` (`routes/auth.py`) signs the new admin in
 * on success -- this mutation writes the session cache the same way
 * `session.ts`'s own `loginMutationOptions` does, and invalidates the
 * unauthenticated setup-status query so `SetupGuard`/`WizardRoute` see
 * the admin_account step complete on their very next read, with no
 * second signal needed.
 */
export const createAdminMutationOptions: UseMutationOptions<Session, unknown, CreateAdminInput> = {
  mutationFn: createAdmin,
  onSuccess: (session) => {
    queryClient.setQueryData(SESSION_QUERY_KEY, session)
    void queryClient.invalidateQueries({ queryKey: SETUP_STATUS_QUERY_KEY })
  },
}

// --- The hub step ---------------------------------------------------------

function checkHubStep(): Promise<WizardStepStatus> {
  return apiFetch<WizardStepStatus>("/api/wizard/steps/hub/check", { method: "POST" })
}

export const checkHubStepMutationOptions: UseMutationOptions<WizardStepStatus, unknown, void> = {
  mutationFn: checkHubStep,
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: WIZARD_QUERY_KEY })
    void queryClient.invalidateQueries({ queryKey: SETUP_STATUS_QUERY_KEY })
  },
}

/** The three distinguishable hub-check failures (Task 2's own
 * instruction) -- `_HubCheckFailed`'s three named categories
 * (`routes/wizard.py`), each sent as `"{category}: {message}"` in the
 * response `detail`. A fixed heading per category sends the operator to
 * a different place; the server's own detail renders underneath it
 * verbatim, matching `deriveCalibrationScreenState.ts`'s established
 * `classify()` shape (03-06) for the identical reason: the branching is
 * unit-testable with no rendered DOM. */
export type HubCheckFailureReason = "unreachable" | "unauthorized" | "unparseable" | "unknown"

const HUB_FAILURE_HEADING: Record<HubCheckFailureReason, string> = {
  unreachable: "Couldn't reach Home Assistant.",
  unauthorized: "Home Assistant rejected the token.",
  unparseable: "Home Assistant's response couldn't be understood.",
  unknown: "Couldn't verify the connection. Check the address and token and try again.",
}

export function classifyHubCheckError(error: unknown): { heading: string; detail: string | undefined } {
  if (error instanceof ApiError) {
    const category = error.message.split(":", 1)[0] as HubCheckFailureReason
    const reason: HubCheckFailureReason =
      category === "unreachable" || category === "unauthorized" || category === "unparseable"
        ? category
        : "unknown"
    return { heading: HUB_FAILURE_HEADING[reason], detail: reason === "unknown" ? undefined : error.message }
  }
  return { heading: HUB_FAILURE_HEADING.unknown, detail: undefined }
}

// --- The audio-source step -------------------------------------------------

export interface SetAudioSourceInput {
  source: string
}

function setAudioSource(input: SetAudioSourceInput): Promise<WizardStepStatus> {
  return apiFetch<WizardStepStatus>("/api/wizard/audio-source", { method: "PUT", body: input })
}

export const setAudioSourceMutationOptions: UseMutationOptions<WizardStepStatus, unknown, SetAudioSourceInput> = {
  mutationFn: setAudioSource,
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: WIZARD_QUERY_KEY })
    void queryClient.invalidateQueries({ queryKey: SETUP_STATUS_QUERY_KEY })
  },
}

// --- Finish -----------------------------------------------------------------

export interface WizardFinishResult {
  complete: boolean
  completed_at: string
}

function finishWizard(): Promise<WizardFinishResult> {
  return apiFetch<WizardFinishResult>("/api/wizard/finish", { method: "POST" })
}

/**
 * `POST /api/wizard/finish` (`routes/wizard.py`) refuses with a 409
 * naming every outstanding step in one comma-joined `detail` string
 * ("setup is not finished -- outstanding steps: hub, provider_set") --
 * this mutation carries that string unmodified rather than picking out
 * only the first name, so a caller rendering `error.message` already
 * shows every one (Task 3's own acceptance criterion).
 */
export const finishWizardMutationOptions: UseMutationOptions<WizardFinishResult, unknown, void> = {
  mutationFn: finishWizard,
  onSuccess: () => {
    void queryClient.invalidateQueries({ queryKey: WIZARD_QUERY_KEY })
    void queryClient.invalidateQueries({ queryKey: SETUP_STATUS_QUERY_KEY })
  },
}
