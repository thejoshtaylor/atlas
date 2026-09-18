import * as React from "react"

export interface EmptyStateProps {
  heading: React.ReactNode
  body: React.ReactNode
  action?: React.ReactNode
}

/**
 * The heading/body/action shape every empty list in this application
 * uses -- 03-UI-SPEC.md's Copywriting Contract owns the exact copy per
 * screen (the denylist, the allowlist, accounts/invites); this component
 * only owns the shape so four screens don't each invent their own.
 */
export function EmptyState({ heading, body, action }: EmptyStateProps) {
  return (
    <div className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border p-8 text-center">
      <p className="text-heading font-semibold text-foreground">{heading}</p>
      <p className="text-body text-muted-foreground">{body}</p>
      {action ? <div className="mt-2">{action}</div> : null}
    </div>
  )
}
