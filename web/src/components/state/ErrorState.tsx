import { Button } from "@/components/ui/button"

export interface ErrorStateProps {
  message: string
  onRetry: () => void
  retryLabel?: string
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
export function ErrorState({ message, onRetry, retryLabel = "Retry" }: ErrorStateProps) {
  return (
    <div className="flex flex-col items-center gap-3 rounded-lg border border-border p-8 text-center">
      <p className="text-body text-foreground">{message}</p>
      <Button variant="outline" onClick={onRetry}>
        {retryLabel}
      </Button>
    </div>
  )
}
