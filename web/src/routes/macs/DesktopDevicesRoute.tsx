import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import {
  buildPairLink,
  createDesktopDeviceMutationOptions,
  desktopDevicesQueryOptions,
  pageLocation,
  type DesktopDevice,
  type DesktopDeviceCreated,
} from "@/lib/desktopDevices"
import { deriveDesktopDevicesScreenState } from "./deriveDesktopDevicesScreenState"
import { lastSeenLine } from "./formatLastSeen"

// An admin pairs a Mac that runs the ATLAS menu bar app, sees its token
// once, and watches which Mac is online (PAIR-01, PAIR-04). The route sits
// behind the admin `RequireRole` block in `App.tsx`. That is presentation
// only. `require_role(Role.ADMIN)` in `routes/desktop_devices.py` holds
// the line.

function statusBadge(device: DesktopDevice): { label: string; variant: "default" | "secondary" | "outline" } {
  if (device.revoked) return { label: "Revoked", variant: "outline" }
  if (device.connected) return { label: "Online", variant: "default" }
  return { label: "Offline", variant: "secondary" }
}

function DesktopDeviceRow({ device, now }: { device: DesktopDevice; now: Date }) {
  const badge = statusBadge(device)
  const seen = lastSeenLine(device, now)

  return (
    <li className="flex flex-col px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-col">
          <div className="flex min-w-0 items-center gap-2">
            <span
              className={`truncate text-body font-semibold ${device.revoked ? "text-muted-foreground" : "text-foreground"}`}
            >
              {device.name}
            </span>
            {device.is_default ? <Badge variant="outline">Default</Badge> : null}
          </div>
          <span className="truncate text-label text-muted-foreground" title={seen.title ?? undefined}>
            {seen.text}
          </span>
        </div>
        <div className="flex shrink-0 items-center">
          <Badge variant={badge.variant}>{badge.label}</Badge>
        </div>
      </div>
    </li>
  )
}

type CopyState = "idle" | "copied" | "failed"

// A copy button says "Copied" only after the clipboard write resolves.
// `navigator.clipboard` can be missing (insecure context) and `writeText`
// can reject (denied permission). A wrong "Copied" would send the admin to
// paste an empty value, and the one-time token cannot be shown again.
function useCopy(value: string): [CopyState, () => Promise<void>] {
  const [state, setState] = React.useState<CopyState>("idle")
  const copy = async () => {
    if (!navigator.clipboard) {
      setState("failed")
      return
    }
    try {
      await navigator.clipboard.writeText(value)
      setState("copied")
    } catch {
      setState("failed")
    }
  }
  return [state, copy]
}

function DesktopDeviceTokenPanel({ device, onDismiss }: { device: DesktopDeviceCreated; onDismiss: () => void }) {
  const { host, protocol } = pageLocation()
  const pairLink = buildPairLink(host, device.token)
  const [linkState, copyLink] = useCopy(pairLink)
  const [tokenState, copyToken] = useCopy(device.token)

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-primary bg-card p-4">
      <p className="text-body font-semibold text-foreground">Mac added.</p>
      <p className="text-label text-muted-foreground">
        This token will not be shown again. Open the pair link on that Mac, or paste the token into the ATLAS app.
      </p>
      {protocol !== "https:" ? (
        <p className="text-label text-destructive">
          This page is not on https. The Mac app pairs over wss:// only, so the server must be reachable over https.
        </p>
      ) : null}

      <Label>Pair link</Label>
      <code className="scroll-field rounded-md border border-border bg-muted px-3 py-2 text-label font-mono">
        {pairLink}
      </code>
      {linkState === "failed" ? <CopyFailed /> : null}
      <div>
        <Button type="button" variant="outline" onClick={() => void copyLink()}>
          {linkState === "copied" ? "Copied" : "Copy link"}
        </Button>
      </div>

      <Label>Token</Label>
      <code className="scroll-field rounded-md border border-border bg-muted px-3 py-2 text-label font-mono">
        {device.token}
      </code>
      {tokenState === "failed" ? <CopyFailed /> : null}
      <div>
        <Button type="button" variant="outline" onClick={() => void copyToken()}>
          {tokenState === "copied" ? "Copied" : "Copy token"}
        </Button>
      </div>

      <div className="flex flex-wrap gap-2">
        <Button asChild variant="outline">
          <a href={pairLink}>Open in ATLAS</a>
        </Button>
        <Button type="button" variant="ghost" onClick={onDismiss}>
          Dismiss
        </Button>
      </div>
    </div>
  )
}

function CopyFailed() {
  return (
    <p className="text-label text-destructive">
      Could not copy automatically. Select the text above and copy it manually.
    </p>
  )
}

export function DesktopDevicesRoute() {
  const devices = useQuery(desktopDevicesQueryOptions)
  const screen = deriveDesktopDevicesScreenState({ devices })
  const now = new Date()

  const [name, setName] = React.useState("")
  // The token lives here and nowhere else. A refetch of the list must not
  // replace it, and only Dismiss clears it.
  const [justCreated, setJustCreated] = React.useState<DesktopDeviceCreated | null>(null)
  const [createError, setCreateError] = React.useState<string | null>(null)
  const createDevice = useMutation(createDesktopDeviceMutationOptions)

  const handleAddMac = async () => {
    setCreateError(null)
    try {
      const created = await createDevice.mutateAsync({ name })
      setJustCreated(created)
      setName("")
    } catch (error) {
      setCreateError(error instanceof ApiError ? error.message : "Could not add the Mac. Try again.")
    }
  }

  const controlsDisabled = screen.kind !== "ready"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-2">
        <h1 className="text-display font-semibold">Macs</h1>
        <p className="text-body text-muted-foreground">
          A Mac here runs the ATLAS menu bar app. Add a Mac, then open its pair link on that Mac.
        </p>
      </div>

      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
        <p className="text-heading font-semibold text-foreground">Add Mac</p>
        <div className="flex flex-col gap-2">
          <Label htmlFor="desktop-device-name">Name</Label>
          <Input
            id="desktop-device-name"
            value={name}
            maxLength={64}
            disabled={controlsDisabled}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        {createError ? <p className="text-label text-destructive">{createError}</p> : null}
        <SubmitButton onSubmit={handleAddMac} disabled={controlsDisabled || name.length === 0} pendingLabel="Adding…">
          Add Mac
        </SubmitButton>
        {justCreated ? (
          <DesktopDeviceTokenPanel device={justCreated} onDismiss={() => setJustCreated(null)} />
        ) : null}
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void devices.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        screen.devices.length === 0 ? (
          <EmptyState heading="No Macs paired yet." body="Add a Mac above, then open its pair link on that Mac." />
        ) : (
          <ul className="panel-list">
            {screen.devices.map((device) => (
              <DesktopDeviceRow key={device.id} device={device} now={now} />
            ))}
          </ul>
        )
      ) : null}
    </div>
  )
}
