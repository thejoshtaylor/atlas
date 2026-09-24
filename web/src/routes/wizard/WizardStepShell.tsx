import type { ReactNode } from "react"
import { AtlasGlobe } from "@/components/brand/AtlasGlobe"
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
        <div className="flex items-center gap-4">
          <AtlasGlobe spinning label="ATLAS" className="size-10 text-foreground" />
          <div className="flex flex-1 flex-col gap-1.5">
            <p className="readout text-label text-muted-foreground">
              Step {stepIndex + 1} of {WIZARD_STEP_ORDER.length}
            </p>
            <div className="flex gap-1" aria-hidden>
              {WIZARD_STEP_ORDER.map((name, index) => (
                <span
                  key={name}
                  className={index <= stepIndex ? "h-0.5 flex-1 rounded-full bg-primary" : "h-0.5 flex-1 rounded-full bg-border"}
                />
              ))}
            </div>
          </div>
        </div>
        <h1 className="text-display font-semibold">{title}</h1>
        {children}
      </div>
    </main>
  )
}
