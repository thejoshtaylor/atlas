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
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import {
  createEdgeDeviceMutationOptions,
  edgeDevicesQueryOptions,
  revokeEdgeDeviceMutationOptions,
  type EdgeDevice,
  type EdgeDeviceCreated,
} from "@/lib/edgeDevices"
import { deriveEdgeDevicesScreenState } from "./deriveEdgeDevicesScreenState"

// D-03: an admin pairs a Raspberry Pi microphone, sees its token exactly
// once, watches which one is connected, and revokes one at once. Every
// write here sits behind the admin `RequireRole` block in `App.tsx`
// (presentation only, T-03-47's own convention, matching
// `AccountsRoute.tsx`) -- `require_role(Role.ADMIN)`
// (`routes/edge_devices.py`) is what actually holds the line.

function statusLabel(device: EdgeDevice): "Connected" | "Not connected" | "Revoked" {
  if (device.revoked) return "Revoked"
  return device.connected ? "Connected" : "Not connected"
}

function formatLastConnected(lastConnectedAt: string | null): string {
  if (!lastConnectedAt) return "Never"
  return new Date(lastConnectedAt).toLocaleString()
}

function EdgeDeviceTokenPanel({ device, onDismiss }: { device: EdgeDeviceCreated; onDismiss: () => void }) {
  const [copied, setCopied] = React.useState(false)

  return (
    <div className="flex flex-col gap-2 rounded-lg border border-primary bg-card p-4">
      <p className="text-body font-medium text-foreground">Device added.</p>
      <p className="text-label text-muted-foreground">
        This token will not be shown again. Copy it now into the Pi's config file.
      </p>
      <code className="scroll-field rounded-md border border-border bg-muted px-3 py-2 text-label">
        {device.token}
      </code>
      <div className="flex gap-2">
        <Button
          type="button"
          variant="outline"
          onClick={() => {
            void navigator.clipboard?.writeText(device.token)
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

function EdgeDeviceRow({ device, disabled }: { device: EdgeDevice; disabled: boolean }) {
  const revoke = useMutation(revokeEdgeDeviceMutationOptions)
  const status = statusLabel(device)

  return (
    <li className="flex items-center justify-between gap-3 px-4 py-3">
      <div className="flex min-w-0 flex-col">
        <span className="truncate text-body font-medium text-foreground">{device.name}</span>
        <span className="truncate text-label text-muted-foreground">
          Last connected: {formatLastConnected(device.last_connected_at)}
        </span>
      </div>
      <div className="flex shrink-0 items-center gap-2">
        <Badge variant={status === "Connected" ? "default" : "secondary"}>{status}</Badge>
        {device.revoked ? null : (
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button type="button" variant="outline" size="sm" disabled={disabled || revoke.isPending}>
                Revoke
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>{`Revoke ${device.name}?`}</AlertDialogTitle>
                <AlertDialogDescription>
                  The Pi disconnects now and cannot reconnect with this token.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  variant="destructive"
                  onClick={() => void revoke.mutateAsync({ deviceId: device.id })}
                >
                  Revoke device
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        )}
      </div>
    </li>
  )
}

export function EdgeDevicesRoute() {
  const devices = useQuery(edgeDevicesQueryOptions)
  const screen = deriveEdgeDevicesScreenState({ devices })

  const [name, setName] = React.useState("")
  const [justCreated, setJustCreated] = React.useState<EdgeDeviceCreated | null>(null)
  const [createError, setCreateError] = React.useState<string | null>(null)

  const createDevice = useMutation(createEdgeDeviceMutationOptions)

  const handleAddDevice = async () => {
    setCreateError(null)
    try {
      const created = await createDevice.mutateAsync({ name })
      // A local state write, not the query cache -- a refetch of
      // EDGE_DEVICES_QUERY_KEY (createEdgeDeviceMutationOptions' own
      // onSuccess) must never replace this token before the admin
      // dismisses it.
      setJustCreated(created)
      setName("")
    } catch (error) {
      setCreateError(error instanceof ApiError ? error.message : "Couldn't add the device. Try again.")
    }
  }

  const controlsDisabled = screen.kind !== "ready"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Edge devices</h1>
        <p className="text-body text-muted-foreground">
          A device here is a Raspberry Pi microphone. Copy its token into the Pi's config file.
        </p>
      </div>

      <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
        <p className="text-heading font-semibold text-foreground">Add device</p>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="edge-device-name">Name</Label>
          <Input
            id="edge-device-name"
            value={name}
            disabled={controlsDisabled}
            onChange={(event) => setName(event.target.value)}
          />
        </div>
        {createError ? <p className="text-label text-destructive">{createError}</p> : null}
        <SubmitButton
          onSubmit={handleAddDevice}
          disabled={controlsDisabled || name.length === 0}
          pendingLabel="Adding…"
        >
          Add device
        </SubmitButton>
        {justCreated ? <EdgeDeviceTokenPanel device={justCreated} onDismiss={() => setJustCreated(null)} /> : null}
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void devices.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        screen.devices.length === 0 ? (
          <EmptyState heading="No devices paired yet." body="Add a Raspberry Pi microphone above." />
        ) : (
          <ul className="panel-list">
            {screen.devices.map((device) => (
              <EdgeDeviceRow key={device.id} device={device} disabled={controlsDisabled} />
            ))}
          </ul>
        )
      ) : null}
    </div>
  )
}
