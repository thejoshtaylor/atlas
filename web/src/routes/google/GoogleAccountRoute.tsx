import * as React from "react"
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query"
import { useNavigate, useParams, useSearchParams } from "react-router-dom"
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
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import {
  GOOGLE_ACCOUNTS_QUERY_KEY,
  fetchGoogleAccount,
  googleAccountQueryKey,
  refreshCalendarsMutationOptions,
  setCalendarAccessMutationOptions,
  startGoogleLink,
  unlinkGoogleAccountMutationOptions,
  updateGoogleAccountMutationOptions,
  type GoogleAccount,
  type GoogleCalendar,
  type GoogleCalendarAccess,
} from "@/lib/google"
import {
  READ_ONLY_CALENDAR_NOTE,
  accountStatusDisplay,
  calendarAccessOptions,
  deriveGoogleAccountState,
} from "./deriveGoogleAccountState"
import { linkAvailability } from "./deriveGoogleAccountsScreenState"
import { WritingStyleSection } from "./WritingStyleSection"

// GOOG-01, GOOG-02, 09-CONTEXT.md D-03/D-04/D-05/GOOG-12: one linked
// account's own page -- its label, whether it is the default, each
// calendar's access, re-linking, and unlinking. Built in
// `PluginEditorRoute.tsx`'s shape: one scrolling screen, no nav entry of
// its own.

const LINKED_BANNER = "Linked. Every calendar starts off -- choose what ATLAS may see below."
const UNLINK_BODY = "ATLAS loses access to this account's calendars and mail. Drafts already in Gmail stay there."

// R2-WR-09: `DELETE /api/google/accounts/:id` deletes the row BEFORE it
// reconciles the running Google tools, so a 503 from THAT route -- one
// that actually reached it -- means the unlink happened and only the
// reconcile failed; the server stopped the tools (a narrowing change,
// D-05). The server's own text says "retry this action", but a retry of
// this DELETE can only 404, so this page says what is true instead and
// hands the sentence to the list page.
function unlinkedToolsStoppedNotice(label: string): string {
  return `Unlinked ${label}. The Google tools stopped and did not start again. They start again after the next account change or token refresh.`
}

function CalendarRow({ accountId, calendar }: { accountId: number; calendar: GoogleCalendar }) {
  const setAccess = useMutation(setCalendarAccessMutationOptions)
  const [error, setError] = React.useState<string | null>(null)
  const options = calendarAccessOptions(calendar)
  const readWriteDisabled = options.find((option) => option.value === "read_write")?.disabled ?? false

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center gap-2">
        <span className="truncate text-body font-medium text-foreground">{calendar.name}</span>
        {calendar.is_primary ? <Badge variant="outline">Primary</Badge> : null}
      </div>
      <RadioGroup
        value={calendar.access}
        disabled={setAccess.isPending}
        onValueChange={(value) => {
          setError(null)
          setAccess.mutate(
            { accountId, calendarId: calendar.id, access: value as GoogleCalendarAccess },
            {
              onError: (err) =>
                setError(err instanceof ApiError ? err.message : "Couldn't update this calendar. Try again."),
            },
          )
        }}
        aria-label={`${calendar.name} access`}
      >
        {options.map((option) => (
          <div key={option.value} className="touch-target flex items-center gap-2">
            <RadioGroupItem
              value={option.value}
              id={`calendar-${calendar.id}-${option.value}`}
              disabled={option.disabled}
            />
            <Label htmlFor={`calendar-${calendar.id}-${option.value}`}>{option.label}</Label>
          </div>
        ))}
      </RadioGroup>
      {readWriteDisabled ? <p className="text-label text-muted-foreground">{READ_ONLY_CALENDAR_NOTE}</p> : null}
      {error ? <p className="text-body text-destructive">{error}</p> : null}
    </div>
  )
}

function AccountDetail({ account }: { account: GoogleAccount }) {
  const navigate = useNavigate()
  const queryClient = useQueryClient()
  const [searchParams] = useSearchParams()
  const linked = searchParams.get("linked") === "1"
  const status = accountStatusDisplay(account.status)

  const [labelDraft, setLabelDraft] = React.useState(account.label)
  const [labelError, setLabelError] = React.useState<string | null>(null)
  const updateLabel = useMutation(updateGoogleAccountMutationOptions)

  const [defaultError, setDefaultError] = React.useState<string | null>(null)
  const updateDefault = useMutation(updateGoogleAccountMutationOptions)

  const refreshCalendars = useMutation(refreshCalendarsMutationOptions)
  const [refreshError, setRefreshError] = React.useState<string | null>(null)

  const startLink = useMutation({ mutationFn: startGoogleLink })
  const [relinkError, setRelinkError] = React.useState<string | null>(null)

  const unlink = useMutation(unlinkGoogleAccountMutationOptions)
  const [unlinkOpen, setUnlinkOpen] = React.useState(false)
  const [unlinkError, setUnlinkError] = React.useState<string | null>(null)

  const relinkAvailability = linkAvailability(window.location.protocol)

  const handleSaveLabel = async () => {
    setLabelError(null)
    try {
      await updateLabel.mutateAsync({ accountId: account.id, label: labelDraft })
    } catch (err) {
      setLabelError(err instanceof ApiError ? err.message : "Couldn't save this label. Try again.")
      throw err
    }
  }

  const handleToggleDefault = (checked: boolean) => {
    setDefaultError(null)
    updateDefault.mutate(
      { accountId: account.id, is_default: checked },
      {
        onError: (err) =>
          setDefaultError(err instanceof ApiError ? err.message : "Couldn't update this account. Try again."),
      },
    )
  }

  const handleRefresh = () => {
    setRefreshError(null)
    refreshCalendars.mutate(
      { accountId: account.id },
      {
        onError: (err) =>
          setRefreshError(err instanceof ApiError ? err.message : "Couldn't check for new calendars. Try again."),
      },
    )
  }

  const handleRelink = async () => {
    setRelinkError(null)
    try {
      const result = await startLink.mutateAsync({ label: account.label, relink_account_id: account.id })
      window.location.assign(result.authorization_url)
    } catch (err) {
      setRelinkError(err instanceof ApiError ? err.message : "Couldn't start linking. Try again.")
    }
  }

  const handleUnlink = async () => {
    setUnlinkError(null)
    try {
      await unlink.mutateAsync({ accountId: account.id })
      setUnlinkOpen(false)
      navigate("/google")
    } catch (err) {
      if (err instanceof ApiError && err.status === 503) {
        // R3-WR-02: `apiFetch` turns EVERY 503 into this same `ApiError`
        // shape, whether or not the DELETE ever reached the route above --
        // k3s's default Traefik ingress (or any reverse proxy in front of
        // the Compose deployment) returns its own plain-text 503 when the
        // pod has no ready endpoint, during a rollout or a restart. Ask the
        // server what actually happened rather than guessing from the
        // status code alone: refetch the account, and treat it as gone
        // only when that refetch itself 404s -- the one shape
        // `_unknown_account_error` always raises once the row is truly
        // deleted, whatever wording the DELETE route's own 503 detail
        // carried this time.
        let confirmedGone = false
        try {
          await fetchGoogleAccount(account.id)
        } catch (refetchErr) {
          confirmedGone = refetchErr instanceof ApiError && refetchErr.status === 404
        }
        if (confirmedGone) {
          // The mutation's own `onSuccess` never ran (the DELETE 503'd
          // too), so do its cache work here, then leave.
          queryClient.removeQueries({ queryKey: googleAccountQueryKey(account.id) })
          void queryClient.invalidateQueries({ queryKey: GOOGLE_ACCOUNTS_QUERY_KEY, exact: true })
          setUnlinkOpen(false)
          navigate("/google", { state: { notice: unlinkedToolsStoppedNotice(account.label) } })
          return
        }
        // The account still exists (or the refetch itself failed some
        // other way, e.g. the same proxy again) -- the DELETE never
        // completed. Show what the server said and stay on the page; a
        // retry is exactly what the operator needs, not a false "gone".
        setUnlinkError(err.message)
        return
      }
      setUnlinkError(err instanceof ApiError ? err.message : "Couldn't unlink this account. Try again.")
    }
  }

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-3">
        <h1 className="truncate text-display font-semibold">{account.label}</h1>
        <Badge variant={status.badgeVariant}>{status.badgeText}</Badge>
      </div>
      <p className="text-body text-muted-foreground">{account.email}</p>

      {linked ? <p className="text-body text-foreground">{LINKED_BANNER}</p> : null}
      {account.status_detail ? <p className="text-body text-muted-foreground">{account.status_detail}</p> : null}

      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="google-account-label">Label</Label>
          <Input id="google-account-label" value={labelDraft} onChange={(event) => setLabelDraft(event.target.value)} />
        </div>
        {labelError ? <p className="text-body text-destructive">{labelError}</p> : null}
        <SubmitButton onSubmit={handleSaveLabel} disabled={labelDraft.trim() === ""}>
          Save label
        </SubmitButton>

        <div className="touch-target flex items-center gap-2">
          <Checkbox
            id="google-account-default"
            checked={account.is_default}
            disabled={updateDefault.isPending}
            onCheckedChange={(checked) => handleToggleDefault(checked === true)}
          />
          <Label htmlFor="google-account-default">Use for new events when I don&apos;t name an account</Label>
        </div>
        {defaultError ? <p className="text-body text-destructive">{defaultError}</p> : null}
      </div>

      <div className="flex flex-col gap-3">
        <div className="flex items-center justify-between gap-3">
          <p className="text-heading font-semibold text-foreground">Calendars</p>
          <Button
            type="button"
            variant="outline"
            size="sm"
            disabled={refreshCalendars.isPending}
            onClick={handleRefresh}
          >
            Find new calendars
          </Button>
        </div>
        {refreshError ? <p className="text-body text-destructive">{refreshError}</p> : null}
        {account.calendars.map((calendar) => (
          <CalendarRow key={calendar.id} accountId={account.id} calendar={calendar} />
        ))}
      </div>

      {account.status === "needs_relink" ? (
        <div className="flex flex-col gap-2">
          {!relinkAvailability.available ? (
            <p className="text-body text-destructive">{relinkAvailability.reason}</p>
          ) : null}
          <Button
            type="button"
            variant="outline"
            disabled={startLink.isPending || !relinkAvailability.available}
            onClick={() => void handleRelink()}
          >
            Link again
          </Button>
          {relinkError ? <p className="text-body text-destructive">{relinkError}</p> : null}
        </div>
      ) : null}

      <WritingStyleSection accountId={account.id} />

      <div className="flex flex-col gap-2 border-t border-border pt-4">
        <AlertDialog open={unlinkOpen} onOpenChange={setUnlinkOpen}>
          <Button type="button" variant="destructive" onClick={() => setUnlinkOpen(true)}>
            Unlink
          </Button>
          <AlertDialogContent>
            <AlertDialogHeader>
              <AlertDialogTitle>{`Unlink ${account.label}?`}</AlertDialogTitle>
              <AlertDialogDescription>{UNLINK_BODY}</AlertDialogDescription>
            </AlertDialogHeader>
            <AlertDialogFooter>
              <AlertDialogCancel onClick={() => setUnlinkOpen(false)}>Cancel</AlertDialogCancel>
              <AlertDialogAction variant="destructive" disabled={unlink.isPending} onClick={() => void handleUnlink()}>
                {`Unlink ${account.label}`}
              </AlertDialogAction>
            </AlertDialogFooter>
          </AlertDialogContent>
        </AlertDialog>
        {unlinkError ? <p className="text-body text-destructive">{unlinkError}</p> : null}
      </div>
    </div>
  )
}

export function GoogleAccountRoute() {
  const params = useParams<{ id: string }>()
  const accountId = Number(params.id)
  const query = useQuery({
    queryKey: googleAccountQueryKey(accountId),
    queryFn: () => fetchGoogleAccount(accountId),
  })
  const screen = deriveGoogleAccountState(query)

  if (screen.kind === "ready") {
    return <AccountDetail key={screen.account.id} account={screen.account} />
  }

  return (
    <div className="flex flex-col gap-6">
      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}
      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
      ) : null}
      {screen.kind === "not_found" ? <ErrorState message="This Google account could not be found." /> : null}
    </div>
  )
}
