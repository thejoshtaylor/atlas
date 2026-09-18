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
  AlertDialogTrigger,
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
  ACCOUNTS_QUERY_KEY,
  INVITES_QUERY_KEY,
  createInviteMutationOptions,
  fetchAccounts,
  fetchInvites,
  formatPendingInviteCount,
  removeAccountMutationOptions,
  revokeInviteMutationOptions,
  type Account,
  type Invite,
  type InviteCreated,
} from "@/lib/accounts"
import type { Role } from "@/lib/session"
import { deriveAccountsScreenState } from "./deriveAccountsScreenState"

// WEB-04, WEB-05, 03-UI-SPEC.md's Focal Point row: "the 'Send invite' CTA
// [is the focal point]; the card list, then the pending-invite count
// line, is secondary." Every role limit shown here is presentation only
// -- `RequireRole` (components/layout/RequireRole.tsx) already keeps a
// non-admin off this route entirely, and `require_role(Role.ADMIN)`
// (`routes/accounts.py`) is what actually refuses a write; nothing in
// this file is the thing that holds T-03-47's line.

const ROLE_OPTIONS: { value: Role; label: string }[] = [
  { value: "viewer", label: "Viewer" },
  { value: "operator", label: "Operator" },
  { value: "admin", label: "Admin" },
]

function AccountRow({
  account,
  disabled,
}: {
  account: Account
  disabled: boolean
}) {
  const remove = useMutation(removeAccountMutationOptions)

  return (
    <li className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex min-w-0 flex-col">
        <span className="truncate text-body font-medium text-foreground">{account.display_name}</span>
        <span className="truncate text-label text-muted-foreground">{account.email}</span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <Badge variant="secondary">{account.role}</Badge>
        <AlertDialog>
          <AlertDialogTrigger asChild>
            <Button type="button" variant="outline" size="sm" disabled={disabled || remove.isPending}>
              Remove
            </Button>
          </AlertDialogTrigger>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>Remove access</AlertDialogTitle>
              <AlertDialogDescription>
                {`Remove ${account.display_name}'s access? They can no longer sign in.`}
              </AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel>Cancel</AlertDialogCancel>
              <AlertDialogAction
                variant="destructive"
                onClick={() => void remove.mutateAsync({ accountId: account.id })}
              >
                Remove access
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
      </div>
    </li>
  )
}

function InviteRow({ invite, disabled }: { invite: Invite; disabled: boolean }) {
  const revoke = useMutation(revokeInviteMutationOptions)

  return (
    <li className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card p-4">
      <div className="flex min-w-0 flex-col">
        <span className="truncate text-body font-medium text-foreground">
          {invite.email ?? "Untargeted invite link"}
        </span>
        <span className="truncate text-label text-muted-foreground">
          Expires {new Date(invite.expires_at).toLocaleDateString()}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <Badge variant="secondary">{invite.role}</Badge>
        <Button
          type="button"
          variant="outline"
          size="sm"
          disabled={disabled || revoke.isPending}
          onClick={() => void revoke.mutateAsync({ inviteId: invite.id })}
        >
          Revoke
        </Button>
      </div>
    </li>
  )
}

function InviteTokenPanel({ invite, onDismiss }: { invite: InviteCreated; onDismiss: () => void }) {
  const [copied, setCopied] = React.useState(false)

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-primary bg-card p-4">
      <p className="text-body font-medium text-foreground">Invite sent.</p>
      <p className="text-label text-muted-foreground">
        This link will not be shown again. Copy it now and send it to {invite.email ?? "the person you're inviting"}.
      </p>
      <code className="scroll-field rounded-md border border-border bg-muted px-3 py-2 text-label">
        {invite.token}
      </code>
      <div className="flex gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={() => {
            void navigator.clipboard?.writeText(invite.token)
            setCopied(true)
          }}
        >
          {copied ? "Copied" : "Copy"}
        </Button>
        <Button type="button" variant="ghost" onClick={onDismiss}>
          Dismiss
        </Button>
      </div>
    </div>
  )
}

export function AccountsRoute() {
  const accounts = useQuery({ queryKey: ACCOUNTS_QUERY_KEY, queryFn: fetchAccounts })
  const invites = useQuery({ queryKey: INVITES_QUERY_KEY, queryFn: fetchInvites })
  const screen = deriveAccountsScreenState({ accounts, invites })

  const [inviteRole, setInviteRole] = React.useState<Role>("operator")
  const [inviteEmail, setInviteEmail] = React.useState("")
  const [justCreated, setJustCreated] = React.useState<InviteCreated | null>(null)

  const createInvite = useMutation(createInviteMutationOptions)

  const handleSendInvite = async () => {
    const created = await createInvite.mutateAsync({
      role: inviteRole,
      email: inviteEmail || undefined,
    })
    // A local state write, not the query cache -- a refetch of
    // INVITES_QUERY_KEY (createInviteMutationOptions' own onSuccess)
    // must never replace this token before the admin dismisses it.
    setJustCreated(created)
    setInviteEmail("")
  }

  const controlsDisabled = screen.kind !== "ready"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Accounts and invites</h1>
      </div>

      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
        <p className="text-heading font-semibold text-foreground">Send invite</p>
        <RadioGroup
          value={inviteRole}
          onValueChange={(value) => setInviteRole(value as Role)}
          aria-label="Role for this invite"
        >
          {ROLE_OPTIONS.map((option) => (
            <div key={option.value} className="flex items-center gap-2">
              <RadioGroupItem value={option.value} id={`invite-role-${option.value}`} disabled={controlsDisabled} />
              <Label htmlFor={`invite-role-${option.value}`}>{option.label}</Label>
            </div>
          ))}
        </RadioGroup>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="invite-email">Email (optional)</Label>
          <Input
            id="invite-email"
            type="email"
            value={inviteEmail}
            disabled={controlsDisabled}
            onChange={(event) => setInviteEmail(event.target.value)}
          />
        </div>
        <SubmitButton onSubmit={handleSendInvite} disabled={controlsDisabled} pendingLabel="Sending…">
          Send invite
        </SubmitButton>
        {justCreated ? (
          <InviteTokenPanel invite={justCreated} onDismiss={() => setJustCreated(null)} />
        ) : null}
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void accounts.refetch().then(() => invites.refetch())} />
      ) : null}

      {screen.kind === "ready" ? (
        <>
          <div className="flex flex-col gap-2">
            {screen.invites.length === 0 ? (
              <EmptyState
                heading="No one invited yet."
                body="Invite an operator or viewer to share access."
              />
            ) : (
              <>
                <p className="text-label text-muted-foreground">
                  {formatPendingInviteCount(screen.invites.length)}
                </p>
                <ul className="flex flex-col gap-2">
                  {screen.invites.map((invite) => (
                    <InviteRow key={invite.id} invite={invite} disabled={controlsDisabled} />
                  ))}
                </ul>
              </>
            )}
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-heading font-semibold text-foreground">People with access</p>
            <ul className="flex flex-col gap-2">
              {screen.accounts.map((account) => (
                <AccountRow key={account.id} account={account} disabled={controlsDisabled} />
              ))}
            </ul>
          </div>
        </>
      ) : null}
    </div>
  )
}
