import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
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
import { WritingStyleSection } from "./WritingStyleSection"

// GOOG-01, GOOG-02, 09-CONTEXT.md D-03/D-04/D-05/GOOG-12: one linked
// account's own page -- its label, whether it is the default, each
// calendar's access, re-linking, and unlinking. Built in
// `PluginEditorRoute.tsx`'s shape: one scrolling screen, no nav entry of
// its own.

const LINKED_BANNER = "Linked. Every calendar starts off -- choose what ATLAS may see below."
const UNLINK_BODY = "ATLAS loses access to this account's calendars and mail. Drafts already in Gmail stay there."

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
    } catch {
      setRelinkError("Couldn't start linking. Try again.")
    }
  }

  const handleUnlink = async () => {
    setUnlinkError(null)
    try {
      await unlink.mutateAsync({ accountId: account.id })
      setUnlinkOpen(false)
      navigate("/google")
    } catch {
      setUnlinkError("Couldn't unlink this account. Try again.")
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
          <Button type="button" variant="outline" disabled={startLink.isPending} onClick={() => void handleRelink()}>
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
