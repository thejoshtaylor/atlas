import * as React from "react"
import { useMutation } from "@tanstack/react-query"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ApiError } from "@/lib/api"
import {
  deleteTimerMutationOptions,
  updateTimerMutationOptions,
  type Timer,
  type TimerChanges,
  type Weekday,
} from "@/lib/timers"
import { DayPicker } from "./DayPicker"
import { describeDays, formatClock, formatCountdown, secondsLeft } from "./deriveTimersScreenState"

// One row: a timer with its countdown, or an alarm with its clock time. Each
// row shows its own inline error (the WR-01 pattern of `EdgeDeviceRow`).

function errorText(error: unknown, fallback: string): string {
  return error instanceof ApiError ? error.message : fallback
}

function TimerEditForm({
  timer,
  nowMs,
  onDone,
  onSave,
}: {
  timer: Timer
  nowMs: number
  onDone: () => void
  onSave: (changes: TimerChanges) => Promise<boolean>
}) {
  const left = timer.kind === "timer" ? secondsLeft(timer, nowMs) : 0
  const [label, setLabel] = React.useState(timer.label)
  const [minutes, setMinutes] = React.useState(String(Math.floor(left / 60)))
  const [seconds, setSeconds] = React.useState(String(left % 60))
  const [time, setTime] = React.useState(timer.time ?? "07:00")
  const [days, setDays] = React.useState<Weekday[]>(timer.days)
  const id = `timer-edit-${timer.id}`

  const handleSave = async () => {
    const changes: TimerChanges = {}
    if (label !== timer.label) changes.label = label
    if (timer.kind === "timer") {
      const total = Number(minutes || 0) * 60 + Number(seconds || 0)
      if (total !== left) changes.remaining_seconds = total
    } else {
      if (time !== timer.time) changes.time = time
      if (days.join() !== timer.days.join()) changes.days = days
    }
    if (Object.keys(changes).length === 0) {
      onDone()
      return
    }
    if (await onSave(changes)) onDone()
  }

  return (
    <div className="flex flex-col gap-3 pt-2">
      <div className="flex flex-col gap-1.5">
        <Label htmlFor={`${id}-label`}>Label</Label>
        <Input id={`${id}-label`} value={label} onChange={(event) => setLabel(event.target.value)} />
      </div>
      {timer.kind === "timer" ? (
        <div className="flex gap-3">
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={`${id}-minutes`}>Minutes left</Label>
            <Input
              id={`${id}-minutes`}
              type="number"
              min={0}
              value={minutes}
              onChange={(event) => setMinutes(event.target.value)}
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={`${id}-seconds`}>Seconds left</Label>
            <Input
              id={`${id}-seconds`}
              type="number"
              min={0}
              max={59}
              value={seconds}
              onChange={(event) => setSeconds(event.target.value)}
            />
          </div>
        </div>
      ) : (
        <>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor={`${id}-time`}>Time</Label>
            <Input id={`${id}-time`} type="time" value={time} onChange={(event) => setTime(event.target.value)} />
          </div>
          <DayPicker idPrefix={`${id}-day`} value={days} onChange={setDays} />
        </>
      )}
      <div className="flex gap-2">
        <Button type="button" size="sm" onClick={() => void handleSave()}>
          Save
        </Button>
        <Button type="button" size="sm" variant="ghost" onClick={onDone}>
          Cancel
        </Button>
      </div>
    </div>
  )
}

export function TimerItem({ timer, nowMs }: { timer: Timer; nowMs: number }) {
  const update = useMutation(updateTimerMutationOptions)
  const remove = useMutation(deleteTimerMutationOptions)
  const [error, setError] = React.useState<string | null>(null)
  const [editing, setEditing] = React.useState(false)
  const busy = update.isPending || remove.isPending

  // Returns whether the change was saved. A failure shows on this row.
  const save = async (changes: TimerChanges): Promise<boolean> => {
    setError(null)
    try {
      await update.mutateAsync({ id: timer.id, changes })
      return true
    } catch (caught) {
      setError(errorText(caught, "Couldn't save the change. Try again."))
      return false
    }
  }

  const handleDelete = async () => {
    setError(null)
    try {
      await remove.mutateAsync({ id: timer.id })
    } catch (caught) {
      setError(errorText(caught, "Couldn't delete it. Try again."))
    }
  }

  const isTimer = timer.kind === "timer"
  const title = timer.label || (isTimer ? "Timer" : "Alarm")

  return (
    <li className="flex flex-col gap-1 px-4 py-3">
      <div className="flex items-center justify-between gap-3">
        <div className="flex min-w-0 flex-col">
          <span className="truncate text-body font-medium text-foreground">{title}</span>
          {isTimer ? (
            <span className="text-label text-muted-foreground">{formatCountdown(secondsLeft(timer, nowMs))}</span>
          ) : (
            <span className="text-label text-muted-foreground">
              {formatClock(timer.time ?? "00:00")} · {describeDays(timer.days)}
            </span>
          )}
        </div>
        <div className="flex shrink-0 flex-wrap items-center justify-end gap-2">
          {timer.paused ? <Badge variant="secondary">Paused</Badge> : null}
          {!isTimer && !timer.enabled ? <Badge variant="secondary">Off</Badge> : null}
          {isTimer ? (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => void save({ paused: !timer.paused })}
            >
              {timer.paused ? "Resume" : "Pause"}
            </Button>
          ) : (
            <Button
              type="button"
              variant="outline"
              size="sm"
              disabled={busy}
              onClick={() => void save({ enabled: !timer.enabled })}
            >
              {timer.enabled ? "Turn off" : "Turn on"}
            </Button>
          )}
          <Button type="button" variant="outline" size="sm" disabled={busy} onClick={() => setEditing(!editing)}>
            Edit
          </Button>
          <Button type="button" variant="outline" size="sm" disabled={busy} onClick={() => void handleDelete()}>
            Delete
          </Button>
        </div>
      </div>
      {editing ? (
        <TimerEditForm
          timer={timer}
          nowMs={nowMs}
          onDone={() => setEditing(false)}
          onSave={save}
        />
      ) : null}
      {error ? <p className="text-label text-destructive">{error}</p> : null}
    </li>
  )
}
