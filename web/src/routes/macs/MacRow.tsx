import * as React from "react"
import { useMutation } from "@tanstack/react-query"
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
import { Checkbox } from "@/components/ui/checkbox"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import {
  revokeDesktopDeviceMutationOptions,
  testDesktopDeviceMutationOptions,
  updateDesktopDeviceMutationOptions,
  type DesktopDevice,
} from "@/lib/desktopDevices"
import { lastSeenLine } from "./formatLastSeen"

// One Mac row (PAIR-03, PAIR-04). Line 1 names the Mac and its state. Line 2
// holds the controls and only shows for a Mac that is not revoked (D-17).
// Every write sits behind the admin `RequireRole` block in `App.tsx`
// (presentation only). `require_role(Role.ADMIN)` in
// `routes/desktop_devices.py` holds the line.

function statusBadge(device: DesktopDevice): { label: string; variant: "default" | "secondary" | "outline" } {
  if (device.revoked) return { label: "Revoked", variant: "outline" }
  if (device.connected) return { label: "Online", variant: "default" }
  return { label: "Offline", variant: "secondary" }
}

/** The server's message for an ApiError, else the fallback copy. */
function failureMessage(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback
}

const TEST_TIMEOUT_LINE = "No answer after 5 seconds. Make sure the Mac is awake and online, then try again."
const TEST_FAILED_LINE = "Could not run the test. Try again."
const TEST_OFFLINE_LINE = "An offline Mac cannot be tested."

function TestResultLine({
  device,
  result,
  failed,
}: {
  device: DesktopDevice
  result: { answered: boolean; rtt_ms: number | null } | undefined
  failed: boolean
}) {
  let content: React.ReactNode = null
  let destructive = false
  if (!device.connected) {
    content = TEST_OFFLINE_LINE
  } else if (failed) {
    content = TEST_FAILED_LINE
    destructive = true
  } else if (result) {
    if (result.answered) {
      content = (
        <>
          Answered in <span className="readout">{result.rtt_ms ?? 0}</span> ms
        </>
      )
    } else {
      content = TEST_TIMEOUT_LINE
      destructive = true
    }
  }
  return (
    <p
      aria-live="polite"
      className={`text-label empty:hidden ${destructive ? "text-destructive" : "text-muted-foreground"}`}
    >
      {content}
    </p>
  )
}

export function MacRow({
  device,
  now,
  controlsDisabled,
}: {
  device: DesktopDevice
  now: Date
  controlsDisabled: boolean
}) {
  const badge = statusBadge(device)
  const seen = lastSeenLine(device, now)

  const rename = useMutation(updateDesktopDeviceMutationOptions)
  const setDefault = useMutation(updateDesktopDeviceMutationOptions)
  const revoke = useMutation(revokeDesktopDeviceMutationOptions)
  const test = useMutation(testDesktopDeviceMutationOptions)

  const [renaming, setRenaming] = React.useState(false)
  const [draft, setDraft] = React.useState(device.name)
  const renameButton = React.useRef<HTMLButtonElement>(null)
  const renameInput = React.useRef<HTMLInputElement>(null)

  // Focus follows the editor: into the input on open, back to Rename on close.
  const wasRenaming = React.useRef(false)
  React.useEffect(() => {
    if (renaming) renameInput.current?.focus()
    else if (wasRenaming.current) renameButton.current?.focus()
    wasRenaming.current = renaming
  }, [renaming])

  const openRename = () => {
    rename.reset()
    setDraft(device.name)
    setRenaming(true)
  }

  const closeRename = () => {
    rename.reset()
    setRenaming(false)
  }

  const saveRename = () => {
    if (draft.length === 0 || rename.isPending) return
    if (draft === device.name) {
      closeRename()
      return
    }
    rename.mutate({ deviceId: device.id, changes: { name: draft } }, { onSuccess: () => setRenaming(false) })
  }

  const renameError = rename.isError ? failureMessage(rename.error, "Could not rename the Mac. Try again.") : null
  const defaultError = setDefault.isError
    ? failureMessage(setDefault.error, "Could not change the default Mac. Try again.")
    : null
  const revokeError = revoke.isError ? failureMessage(revoke.error, "Could not revoke the Mac. Try again.") : null

  const defaultId = `mac-default-${device.id}`

  return (
    <li className="flex flex-col px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-col">
          {renaming ? (
            <div className="flex flex-wrap items-center gap-2">
              <Input
                ref={renameInput}
                aria-label="Mac name"
                value={draft}
                maxLength={64}
                className="min-w-0 flex-1"
                onChange={(event) => setDraft(event.target.value)}
                onKeyDown={(event) => {
                  if (event.key === "Enter") {
                    event.preventDefault()
                    saveRename()
                  } else if (event.key === "Escape") {
                    event.preventDefault()
                    closeRename()
                  }
                }}
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                disabled={rename.isPending || draft.length === 0}
                onClick={saveRename}
              >
                Save
              </Button>
              <Button type="button" variant="ghost" size="sm" onClick={closeRename}>
                Cancel
              </Button>
            </div>
          ) : (
            <div className="flex min-w-0 items-center gap-2">
              <span
                className={`truncate text-body font-semibold ${device.revoked ? "text-muted-foreground" : "text-foreground"}`}
              >
                {device.name}
              </span>
              {device.is_default ? <Badge variant="outline">Default</Badge> : null}
            </div>
          )}
          <span className="truncate text-label text-muted-foreground" title={seen.title ?? undefined}>
            {seen.text}
          </span>
        </div>
        <div className="flex shrink-0 items-center">
          <Badge variant={badge.variant}>{badge.label}</Badge>
        </div>
      </div>

      {device.revoked ? null : (
        <div className="flex flex-wrap items-center gap-2 pt-2">
          <div className="touch-target flex items-center gap-2">
            <Checkbox
              id={defaultId}
              checked={device.is_default}
              disabled={controlsDisabled || setDefault.isPending}
              onCheckedChange={(checked) =>
                setDefault.mutate({ deviceId: device.id, changes: { is_default: checked === true } })
              }
            />
            <Label htmlFor={defaultId}>Default Mac</Label>
          </div>
          <Button
            ref={renameButton}
            type="button"
            variant="outline"
            size="sm"
            disabled={controlsDisabled || rename.isPending}
            onClick={openRename}
          >
            Rename
          </Button>
          <SubmitButton
            variant="outline"
            size="sm"
            pendingLabel="Testing…"
            disabled={controlsDisabled || !device.connected}
            onSubmit={() => test.mutateAsync({ deviceId: device.id })}
          >
            Test
          </SubmitButton>
          <AlertDialog>
            <AlertDialogTrigger asChild>
              <Button type="button" variant="outline" size="sm" disabled={controlsDisabled || revoke.isPending}>
                Revoke
              </Button>
            </AlertDialogTrigger>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>{`Revoke ${device.name}?`}</AlertDialogTitle>
                <AlertDialogDescription>
                  The Mac disconnects now and cannot reconnect with this token. It will ask to pair again.
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel>Cancel</AlertDialogCancel>
                <AlertDialogAction variant="destructive" onClick={() => revoke.mutate({ deviceId: device.id })}>
                  Revoke Mac
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        </div>
      )}

      {device.revoked ? null : <TestResultLine device={device} result={test.data} failed={test.isError} />}
      {renameError ? <p className="text-label text-destructive">{renameError}</p> : null}
      {defaultError ? <p className="text-label text-destructive">{defaultError}</p> : null}
      {revokeError ? <p className="text-label text-destructive">{revokeError}</p> : null}
    </li>
  )
}
