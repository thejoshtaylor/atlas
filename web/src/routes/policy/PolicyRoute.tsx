import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import {
  POLICY_QUERY_KEY,
  addRuleMutationOptions,
  fetchPolicy,
  removeRuleMutationOptions,
  setModeMutationOptions,
  type Mode,
  type PolicyRule,
} from "@/lib/policy"
import { usePolicyDraftStore } from "@/stores/policyDraftStore"
import { derivePolicyScreenState, formatRuleCount, visibleRules } from "./derivePolicyScreenState"

// SAFE-06, SAFE-07, 03-UI-SPEC.md's Focal Point row: "the card list
// itself, not a CTA. The operator came to read what is currently
// denied; the count line sits directly above it and the mode control
// sits apart from the add control so the two are never mis-clicked for
// each other." Every specific instruction below traces back to that row
// or to the Copywriting Contract's exact strings -- this is the
// highest-consequence screen in the product (this plan's own
// <critical_constraints>).

function PolicyRuleRow({
  rule,
  mode,
  disabled,
}: {
  rule: PolicyRule
  mode: Mode
  disabled: boolean
}) {
  const removeRule = useMutation(removeRuleMutationOptions)
  const setPendingRemovalId = usePolicyDraftStore((state) => state.setPendingRemovalId)
  const pendingRemovalId = usePolicyDraftStore((state) => state.pendingRemovalId)
  const open = pendingRemovalId === rule.id

  return (
    <li className="flex items-center justify-between gap-3 px-4 py-3">
      <div className="flex min-w-0 flex-col gap-1">
        <span className="readout truncate text-label text-foreground">{rule.value}</span>
        <div className="flex flex-wrap gap-1.5">
          {mode === "allow_all_except_denylist" ? (
            <Badge variant="denied">Denied for control · readable</Badge>
          ) : null}
          {!rule.resolved ? <Badge variant="denied">Not found in Home Assistant</Badge> : null}
        </div>
      </div>
      <AlertDialog open={open} onOpenChange={(next) => setPendingRemovalId(next ? rule.id : null)}>
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="shrink-0"
          disabled={disabled || removeRule.isPending}
          onClick={() => setPendingRemovalId(rule.id)}
        >
          Remove
        </Button>
        <AlertDialogContent>
          <AlertDialogHeader>
            <AlertDialogTitle>Remove from denylist</AlertDialogTitle>
            <AlertDialogDescription>
              {`Remove ${rule.value} from the denylist? Voice commands will be able to control it again.`}
            </AlertDialogDescription>
          </AlertDialogHeader>
          <AlertDialogFooter>
            <AlertDialogCancel onClick={() => setPendingRemovalId(null)}>Cancel</AlertDialogCancel>
            <AlertDialogAction
              variant="destructive"
              onClick={() => {
                setPendingRemovalId(null)
                void removeRule.mutateAsync({ ruleId: rule.id })
              }}
            >
              Remove from denylist
            </AlertDialogAction>
          </AlertDialogFooter>
        </AlertDialogContent>
      </AlertDialog>
    </li>
  )
}

export function PolicyRoute() {
  const query = useQuery({ queryKey: POLICY_QUERY_KEY, queryFn: fetchPolicy })
  const screen = derivePolicyScreenState(query)

  const setMode = useMutation(setModeMutationOptions)
  const addRule = useMutation(addRuleMutationOptions)
  const pendingValue = usePolicyDraftStore((state) => state.pendingValue)
  const setPendingValue = usePolicyDraftStore((state) => state.setPendingValue)
  const clearDraft = usePolicyDraftStore((state) => state.clear)

  const [pendingModeTarget, setPendingModeTarget] = React.useState<Mode | null>(null)
  const [modeError, setModeError] = React.useState<string | null>(null)

  const controlsDisabled = screen.kind !== "ready"
  const currentMode: Mode = screen.kind === "ready" ? screen.mode : "allow_all_except_denylist"
  const rules = screen.kind === "ready" ? visibleRules(screen.mode, screen.rules) : []

  const handleModeSelect = (next: Mode) => {
    if (next === currentMode) return
    setModeError(null)
    if (next === "allowlist_only") {
      // The destructive direction -- confirm before anything changes.
      setPendingModeTarget(next)
      return
    }
    void setMode.mutateAsync({ mode: next }).catch(() => setModeError("Couldn't switch modes. Try again."))
  }

  const confirmModeSwitch = () => {
    const target = pendingModeTarget
    setPendingModeTarget(null)
    if (!target) return
    void setMode.mutateAsync({ mode: target }).catch(() => setModeError("Couldn't switch modes. Try again."))
  }

  const handleAddRule = async () => {
    const kind = currentMode === "allowlist_only" ? "allow_entity" : "deny_entity"
    await addRule.mutateAsync({ kind, value: pendingValue.trim() })
    clearDraft()
  }

  const addLabel = currentMode === "allowlist_only" ? "Add to allowlist" : "Add to denylist"
  const countLabel = rules.length === 0 ? null : formatRuleCount(currentMode, rules.length)

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Safety policy</h1>
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        <>
          {countLabel ? <p className="text-label text-muted-foreground">{countLabel}</p> : null}

          {rules.length === 0 ? (
            currentMode === "allowlist_only" ? (
              <EmptyState
                heading="Nothing allowed yet."
                body="In allow-only mode, an empty allowlist means voice reaches nothing. Add an entity to allow it."
              />
            ) : (
              <EmptyState
                heading="Nothing denied yet."
                body="Every entity is reachable by voice. Add an entity to block it from voice control — it stays readable."
              />
            )
          ) : (
            <ul className="panel-list">
              {rules.map((rule) => (
                <PolicyRuleRow key={rule.id} rule={rule} mode={currentMode} disabled={controlsDisabled} />
              ))}
            </ul>
          )}
        </>
      ) : null}

      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
        <p className="text-heading font-semibold text-foreground">Mode</p>
        <AlertDialog
          open={pendingModeTarget !== null}
          onOpenChange={(next) => {
            if (!next) setPendingModeTarget(null)
          }}
        >
          <RadioGroup
            value={currentMode}
            onValueChange={(value) => handleModeSelect(value as Mode)}
            aria-label="Safety policy mode"
          >
            <div className="touch-target flex items-center gap-2">
              <RadioGroupItem
                value="allow_all_except_denylist"
                id="mode-deny-list"
                disabled={controlsDisabled || setMode.isPending}
              />
              <Label htmlFor="mode-deny-list">Deny-list (everything reachable except the denylist)</Label>
            </div>
            <div className="touch-target flex items-center gap-2">
              <RadioGroupItem
                value="allowlist_only"
                id="mode-allow-only"
                disabled={controlsDisabled || setMode.isPending}
              />
              <Label htmlFor="mode-allow-only">Allow-only (nothing reachable except the allowlist)</Label>
            </div>
          </RadioGroup>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Switch mode</AlertDialogTitle>
              <AlertDialogDescription>
                Switch to allow-only mode? Every entity not on the allowlist becomes unreachable by voice,
                immediately.
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={() => setPendingModeTarget(null)}>Cancel</AlertDialogCancel>
              <AlertDialogAction variant="destructive" onClick={confirmModeSwitch}>
                Switch mode
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
        {modeError ? <p className="text-body text-destructive">{modeError}</p> : null}
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="policy-add-value">Entity id</Label>
          <Input
            id="policy-add-value"
            placeholder="light.example_lamp"
            value={pendingValue}
            disabled={controlsDisabled}
            onChange={(event) => setPendingValue(event.target.value)}
          />
        </div>
        <SubmitButton onSubmit={handleAddRule} disabled={controlsDisabled || !pendingValue.trim()}>
          {addLabel}
        </SubmitButton>
      </div>
    </div>
  )
}
