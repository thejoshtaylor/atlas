import { useMutation } from "@tanstack/react-query"
import { useNavigate } from "react-router-dom"
import { toast } from "sonner"
import { ErrorState } from "@/components/state/ErrorState"
import { CalibrationRoute } from "@/routes/calibration/CalibrationRoute"
import { WIZARD_STEP_ORDER, finishWizardMutationOptions } from "@/lib/wizard"

/**
 * The wizard's last step. Mounts plan 03-06's calibration screen --
 * mounted, not copied: there is one implementation of the microphone and
 * speaker test, and this is a second place it appears
 * (03-UI-SPEC.md's Focal Point row: "the one screen where the outcome
 * outranks the button" holds exactly as true here as standalone). Its own
 * heading ("Test the microphone and speaker") stands in for
 * `WizardStepShell`'s title, so this does not wrap it in a second one.
 *
 * `onFinish` calls `POST /api/wizard/finish`, which refuses while any
 * step is outstanding and names every one in one comma-joined string
 * (`_finish_incomplete_error`, `routes/wizard.py`) -- rendered verbatim
 * below rather than truncated to the first name, so an operator two steps
 * short is told both (Task 3's own acceptance criterion). The room step
 * itself cannot reach this refusal for its own reason: `CalibrationRoute`
 * only shows the "Finish setup" control once a result exists (03-06's own
 * `showFinish`), so `onFinish` is never reachable before a calibration is
 * on file.
 */
export function RoomStep() {
  const navigate = useNavigate()
  const finish = useMutation(finishWizardMutationOptions)

  const handleFinish = async () => {
    try {
      await finish.mutateAsync()
      // "Take them to the application proper, with a short statement of
      // what now works and what the next thing to do is" (Task 3's own
      // instruction) -- a toast rather than a change to `HomeRoute.tsx`
      // (outside this task's declared file scope), since `Toaster` is
      // mounted at the App root and survives the navigation below.
      toast.success("atlas is set up.", {
        description: "Say the wake phrase to try it. Sign in again any time to change settings.",
      })
      navigate("/", { replace: true })
    } catch {
      // Surfaced by finish.isError below; nothing further to do here.
    }
  }

  return (
    <div className="flex min-h-svh flex-col gap-6 p-6">
      <p className="text-label text-muted-foreground">
        Step {WIZARD_STEP_ORDER.indexOf("room") + 1} of {WIZARD_STEP_ORDER.length}
      </p>
      <div className="mx-auto w-full max-w-sm">
        <CalibrationRoute onFinish={() => void handleFinish()} />
        {finish.isError ? (
          <div className="mt-4">
            <ErrorState
              message="Setup isn't finished yet."
              detail={finish.error instanceof Error ? finish.error.message : undefined}
              onRetry={() => void handleFinish()}
            />
          </div>
        ) : null}
      </div>
    </div>
  )
}
