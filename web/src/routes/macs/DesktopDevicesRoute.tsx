import { useQuery } from "@tanstack/react-query"
import { Badge } from "@/components/ui/badge"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { desktopDevicesQueryOptions, type DesktopDevice } from "@/lib/desktopDevices"
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

export function DesktopDevicesRoute() {
  const devices = useQuery(desktopDevicesQueryOptions)
  const screen = deriveDesktopDevicesScreenState({ devices })
  const now = new Date()

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
