import type { ReactNode } from "react"
import { WIZARD_STEP_ORDER, type WizardStepName } from "@/lib/wizard"

/**
 * The shell every configuration step (Task 1's own text) shares: a step
 * indicator, a title at Display size naming what this step is for, and
 * the step's own content (03-UI-SPEC.md's Focal Point row for "Wizard
 * (any step)"). The room step (`RoomStep.tsx`) does not use this --
 * `CalibrationRoute` already carries its own Display-size heading and is
 * the one screen where the outcome outranks the button, so wrapping it in
 * a second title here would duplicate one and compete with the other.
 */
export function WizardStepShell({
  step,
  title,
  children,
}: {
  step: WizardStepName
  title: string
  children: ReactNode
}) {
  const stepIndex = WIZARD_STEP_ORDER.indexOf(step)
  return (
    <main className="flex min-h-svh flex-col items-center justify-center p-6">
      <div className="flex w-full max-w-sm flex-col gap-6">
        <p className="text-label text-muted-foreground">
          Step {stepIndex + 1} of {WIZARD_STEP_ORDER.length}
        </p>
        <h1 className="text-display font-semibold">{title}</h1>
        {children}
      </div>
    </main>
  )
}
