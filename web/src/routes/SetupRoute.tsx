import { useWizardStore } from "@/stores/wizardStore"

/**
 * The wizard's route -- reachable while no admin exists yet (the
 * create-admin route stays open) and, per CONTEXT.md's "partial, wizard
 * abandoned" row, also reachable by login afterward so a half-finished
 * install is never a lockout. The step sequence itself (source ->
 * provider set -> hub -> mic/speaker test -> finish) is plan 03-09's and
 * 03-10's scope; this route is the mount point they fill, matching the
 * relationship `AppShell` already has with the screens later plans add.
 */
export function SetupRoute() {
  const currentStep = useWizardStore((state) => state.currentStep)

  return (
    <main className="flex min-h-svh flex-col items-center justify-center gap-2 p-6 text-center">
      <h1 className="text-display font-semibold">Set up spire-voice</h1>
      <p className="text-body text-muted-foreground">Step {currentStep + 1}</p>
    </main>
  )
}
