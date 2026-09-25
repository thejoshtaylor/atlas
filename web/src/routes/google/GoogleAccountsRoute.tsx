import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Link, useSearchParams } from "react-router-dom"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import {
  GOOGLE_ACCOUNTS_QUERY_KEY,
  GOOGLE_CLIENT_QUERY_KEY,
  fetchGoogleAccounts,
  fetchGoogleClient,
  linkErrorMessage,
  saveGoogleClientMutationOptions,
  startGoogleLink,
  type GoogleAccount,
  type GoogleClientStatus,
} from "@/lib/google"
import { deriveGoogleAccountsScreenState, linkAvailability } from "./deriveGoogleAccountsScreenState"

// D-01: the fourth setup step, verbatim -- the reason a Testing-mode
// consent screen is the wrong default for a household assistant that
// must stay linked. Named here so this screen and its own test share one
// literal, never two independently-typed copies.
const IN_PRODUCTION_NOTE =
  "Set the app's publishing status to In production. While it is in Testing, Google ends every link after 7 days, and every account would unlink once a week."

// GOOG-01, GOOG-02, 09-CONTEXT.md D-01/D-02/D-03/D-04/GOOG-12: the screen
// where a stranger who cloned this repository sets up their own Google
// OAuth client and links their first account, following D-01's
// instructions, without editing a file. Built in `PluginsRoute.tsx`'s own
// card-list shape; the write-only secret field follows
// `SettingsRoute.tsx`'s own "saved" convention, never showing a value
// back once stored.

function ClientSetupCard({ client }: { client: GoogleClientStatus }) {
  const save = useMutation(saveGoogleClientMutationOptions)
  // The mutation's resolved value is the source of truth for this card's
  // own render, not a refetch of the client query -- the same
  // `loadPlugin(result)` pattern `PluginEditorRoute.tsx` uses after its
  // own config save, so the "Secret saved" state appears the instant the
  // save resolves regardless of when (or whether) a caller's mocked
  // `onSuccess` happens to invalidate the query cache.
  const [savedClient, setSavedClient] = React.useState<GoogleClientStatus | null>(null)
  const effectiveClient = savedClient ?? client
  const [editing, setEditing] = React.useState(!effectiveClient.configured)
  const [clientId, setClientId] = React.useState(effectiveClient.client_id ?? "")
  const [clientSecret, setClientSecret] = React.useState("")
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const redirectUri = `${window.location.origin}${effectiveClient.redirect_path}`

  const handleSave = async () => {
    setSaveError(null)
    try {
      const result = await save.mutateAsync({ client_id: clientId, client_secret: clientSecret })
      setSavedClient(result)
      setEditing(false)
      setClientSecret("")
    } catch {
      setSaveError("Couldn't save this client. Try again.")
      throw new Error("save failed")
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <p className="text-heading font-semibold text-foreground">Google OAuth client</p>

      {!effectiveClient.configured ? (
        <ol className="flex flex-col gap-2 text-body text-muted-foreground">
          <li>1. Create a Google Cloud project and turn on the Calendar and Gmail APIs.</li>
          <li>2. Configure the OAuth consent screen for that project.</li>
          <li>
            3. Create a Web application OAuth client with this redirect URI:{" "}
            <span className="readout block break-all text-foreground">{redirectUri}</span>
          </li>
          <li>
            4. <span>{IN_PRODUCTION_NOTE}</span>
          </li>
        </ol>
      ) : null}

      {!editing ? (
        <div className="flex items-center justify-between gap-2">
          <div className="flex min-w-0 flex-col gap-0.5">
            <span className="readout truncate text-body text-foreground">{effectiveClient.client_id}</span>
            <span className="text-label text-muted-foreground">Secret saved</span>
          </div>
          <Button
            type="button"
            variant="outline"
            size="sm"
            onClick={() => {
              setEditing(true)
              setClientId("")
              setClientSecret("")
            }}
          >
            Update
          </Button>
        </div>
      ) : (
        <div className="flex flex-col gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="google-client-id">Client ID</Label>
            <Input
              id="google-client-id"
              className="scroll-field font-mono"
              value={clientId}
              onChange={(event) => setClientId(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="google-client-secret">Client secret</Label>
            <Input
              id="google-client-secret"
              type="password"
              className="scroll-field font-mono"
              value={clientSecret}
              onChange={(event) => setClientSecret(event.target.value)}
            />
          </div>
          <SubmitButton onSubmit={handleSave} disabled={!clientId || !clientSecret}>
            Save client
          </SubmitButton>
        </div>
      )}

      {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
    </div>
  )
}

function AccountCard({ account }: { account: GoogleAccount }) {
  return (
    <li>
      <Link
        to={`/google/accounts/${account.id}`}
        className="touch-target flex flex-col gap-1.5 px-4 py-3 hover:bg-accent"
      >
        <div className="flex items-center justify-between gap-3">
          <span className="truncate text-body font-medium text-foreground">{account.label}</span>
          <div className="flex shrink-0 items-center gap-1.5">
            {account.is_default ? <Badge variant="secondary">Default</Badge> : null}
            {account.status === "needs_relink" ? <Badge variant="denied">Needs re-link</Badge> : null}
            {account.status === "unreachable" ? <Badge variant="denied">Unreachable</Badge> : null}
          </div>
        </div>
        <span className="truncate text-label text-muted-foreground">{account.email}</span>
        {account.refresh_token_expires_at ? (
          <span className="text-label text-muted-foreground">
            {`Link expires ${new Date(account.refresh_token_expires_at).toLocaleDateString()}`}
          </span>
        ) : null}
      </Link>
    </li>
  )
}

export function GoogleAccountsRoute() {
  const [searchParams] = useSearchParams()
  const linkError = searchParams.get("link_error")

  const clientQuery = useQuery({ queryKey: GOOGLE_CLIENT_QUERY_KEY, queryFn: fetchGoogleClient })
  const accountsQuery = useQuery({ queryKey: GOOGLE_ACCOUNTS_QUERY_KEY, queryFn: fetchGoogleAccounts })
  const screen = deriveGoogleAccountsScreenState(clientQuery, accountsQuery)

  const [label, setLabel] = React.useState("")
  const [linkStartError, setLinkStartError] = React.useState<string | null>(null)
  const startLink = useMutation({ mutationFn: startGoogleLink })
  const availability = linkAvailability(window.location.protocol)

  const handleLink = async () => {
    setLinkStartError(null)
    try {
      const result = await startLink.mutateAsync({ label, relink_account_id: null })
      window.location.assign(result.authorization_url)
    } catch {
      setLinkStartError("Couldn't start linking. Try again.")
      throw new Error("link failed")
    }
  }

  const linkErrorText = linkErrorMessage(linkError)

  return (
    <div className="flex flex-col gap-6">
      <h1 className="text-display font-semibold">Google accounts</h1>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState
          message={screen.message}
          onRetry={() => {
            void clientQuery.refetch()
            void accountsQuery.refetch()
          }}
        />
      ) : null}

      {screen.kind === "ready" ? (
        <>
          <ClientSetupCard client={screen.client} />

          {linkErrorText ? <p className="text-body text-destructive">{linkErrorText}</p> : null}

          {!availability.available ? <p className="text-body text-destructive">{availability.reason}</p> : null}

          <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="google-link-label">Label</Label>
              <Input id="google-link-label" value={label} onChange={(event) => setLabel(event.target.value)} />
            </div>
            {linkStartError ? <p className="text-body text-destructive">{linkStartError}</p> : null}
            <SubmitButton onSubmit={handleLink} disabled={!availability.available || !label}>
              Link
            </SubmitButton>
          </div>

          <ul className="panel-list">
            {screen.accounts.map((account) => (
              <AccountCard key={account.id} account={account} />
            ))}
          </ul>
        </>
      ) : null}
    </div>
  )
}
