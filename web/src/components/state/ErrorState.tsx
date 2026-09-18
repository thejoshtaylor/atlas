import { Button } from "@/components/ui/button"

export interface ErrorStateProps {
  message: string
  onRetry: () => void
  retryLabel?: string
  /**
   * An optional second line, shown verbatim below `message`. Added for
   * 03-06's calibration screen, whose failed-run treatments (a disabled
   * route, a run already in progress, an unmeasurable run) each show a
   * fixed heading *and* the server's own named reason underneath it --
   * two distinct lines, never merged into one paraphrase. Optional so
   * every existing caller (a single-line failure) is unaffected.
   */
  detail?: string
}

/**
 * A message plus a retry callback -- every failed load in this
 * application (the policy list, the accounts list, the setup-status
 * check) renders through this rather than a bespoke per-screen error
 * block. The message itself comes from the caller (usually an
 * `ApiError`'s `.message`, which is the server's own named `detail`) so
 * this component never paraphrases a reason 03-UI-SPEC.md's Copywriting
 * Contract already spells out.
 */
export function ErrorState({ message, onRetry, retryLabel = "Retry", detail }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-border p-8 text-center">
      <p className="text-body text-foreground">{message}</p>
      {detail ? <p className="text-label text-muted-foreground">{detail}</p> : null}
      <Button variant="outline" onClick={onRetry}>
        {retryLabel}
      </Button>
    </div>
  )
}
