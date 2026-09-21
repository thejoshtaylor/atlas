import * as React from "react"
import { Button } from "@/components/ui/button"
import { createSubmitGuard } from "@/lib/submitGuard"
import type { VariantProps } from "class-variance-authority"
import type { buttonVariants } from "@/components/ui/button"

export interface SubmitButtonProps
  extends Omit<React.ComponentProps<"button">, "onClick">,
    VariantProps<typeof buttonVariants> {
  /**
   * Every submitting form in this application -- the wizard steps,
   * sign-in, sending an invite, saving a credential -- uses this
   * component instead of a per-form pending flag. `onSubmit` should
   * resolve (or reject) when the request settles; the button stays
   * disabled for exactly that long, and a second click issued while it
   * is still pending is dropped rather than starting a second request
   * (T-03-14: a stored credential is never readable back, so an
   * ambiguous double write cannot be inspected afterward).
   */
  onSubmit: () => Promise<unknown>
  pendingLabel?: React.ReactNode
}

export function SubmitButton({
  onSubmit,
  pendingLabel,
  children,
  disabled,
  variant,
  size = "default",
  ...props
}: SubmitButtonProps) {
  const [pending, setPending] = React.useState(false)

  // A ref, not a value closed over at creation: `onSubmit` is a new
  // function identity on most renders (an inline arrow function is the
  // normal caller shape), and the guard itself is created exactly once
  // per mounted button so its `pending` flag is shared across every
  // click on *this* button without leaking into any other instance.
  const onSubmitRef = React.useRef(onSubmit)
  onSubmitRef.current = onSubmit

  const guardRef = React.useRef<ReturnType<typeof createSubmitGuard> | null>(null)
  if (guardRef.current === null) {
    guardRef.current = createSubmitGuard(() => onSubmitRef.current())
  }

  const handleClick = () => {
    // `.catch(() => {})` after `.finally(...)`, not instead of it: nothing
    // downstream ever reads this promise (a click handler is `void` by
    // nature), so a rejected `onSubmit` -- e.g. `handleAddRule` in
    // `PolicyRoute.tsx`, which awaits its mutation with no catch of its
    // own -- would otherwise surface as an unhandled promise rejection on
    // every real submit failure this component's callers do not already
    // catch themselves (found while mounting this component for the
    // first time, 08-09-PLAN.md Task 1).
    void guardRef.current!.run()
      .finally(() => {
        setPending(guardRef.current!.isPending())
      })
      .catch(() => {})
    // Reflects the guard's synchronous decision immediately -- the
    // `finally` above only matters for the eventual `false` transition,
    // since `run()` itself already rejected a re-entrant call before any
    // microtask boundary.
    setPending(guardRef.current!.isPending())
  }

  return (
    <Button
      type="button"
      variant={variant}
      size={size}
      disabled={disabled || pending}
      aria-busy={pending}
      onClick={handleClick}
      {...props}
    >
      {pending && pendingLabel !== undefined ? pendingLabel : children}
    </Button>
  )
}
