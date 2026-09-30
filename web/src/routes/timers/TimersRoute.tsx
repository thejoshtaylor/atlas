import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { ApiError } from "@/lib/api"
import { createTimerMutationOptions, timersQueryOptions, type Weekday } from "@/lib/timers"
import { DayPicker } from "./DayPicker"
import { deriveTimersScreenState } from "./deriveTimersScreenState"
import { TimerItem } from "./TimerItem"

// Operators list, create, edit, pause and delete timers and alarms here. They
// ring on the house speaker. `require_role(Role.OPERATOR)`
// (`routes/timers.py`) holds the line; the `RequireRole` block in `App.tsx`
// is presentation only.

function NewTimerPanel({ disabled }: { disabled: boolean }) {
  const create = useMutation(createTimerMutationOptions)
  const [label, setLabel] = React.useState("")
  const [minutes, setMinutes] = React.useState("")
  const [seconds, setSeconds] = React.useState("")
  const [error, setError] = React.useState<string | null>(null)
  const total = Number(minutes || 0) * 60 + Number(seconds || 0)

  const handleStart = async () => {
    setError(null)
    try {
      await create.mutateAsync({ kind: "timer", label, duration_seconds: total })
      setLabel("")
      setMinutes("")
      setSeconds("")
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Couldn't start the timer. Try again.")
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <p className="text-heading font-semibold text-foreground">New timer</p>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="new-timer-label">Timer label</Label>
        <Input
          id="new-timer-label"
          value={label}
          disabled={disabled}
          onChange={(event) => setLabel(event.target.value)}
        />
      </div>
      <div className="flex gap-3">
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="new-timer-minutes">Minutes</Label>
          <Input
            id="new-timer-minutes"
            type="number"
            min={0}
            value={minutes}
            disabled={disabled}
            onChange={(event) => setMinutes(event.target.value)}
          />
        </div>
        <div className="flex flex-col gap-1.5">
          <Label htmlFor="new-timer-seconds">Seconds</Label>
          <Input
            id="new-timer-seconds"
            type="number"
            min={0}
            max={59}
            value={seconds}
            disabled={disabled}
            onChange={(event) => setSeconds(event.target.value)}
          />
        </div>
      </div>
      {error ? <p className="text-label text-destructive">{error}</p> : null}
      <SubmitButton onSubmit={handleStart} disabled={disabled || total < 1} pendingLabel="Starting…">
        Start timer
      </SubmitButton>
    </div>
  )
}

function NewAlarmPanel({ disabled }: { disabled: boolean }) {
  const create = useMutation(createTimerMutationOptions)
  const [label, setLabel] = React.useState("")
  const [time, setTime] = React.useState("")
  const [days, setDays] = React.useState<Weekday[]>([])
  const [error, setError] = React.useState<string | null>(null)

  const handleSet = async () => {
    setError(null)
    try {
      await create.mutateAsync({ kind: "alarm", label, time, days })
      setLabel("")
      setTime("")
      setDays([])
    } catch (caught) {
      setError(caught instanceof ApiError ? caught.message : "Couldn't set the alarm. Try again.")
    }
  }

  return (
    <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
      <p className="text-heading font-semibold text-foreground">New alarm</p>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="new-alarm-label">Alarm label</Label>
        <Input
          id="new-alarm-label"
          value={label}
          disabled={disabled}
          onChange={(event) => setLabel(event.target.value)}
        />
      </div>
      <div className="flex flex-col gap-1.5">
        <Label htmlFor="new-alarm-time">Time</Label>
        <Input
          id="new-alarm-time"
          type="time"
          value={time}
          disabled={disabled}
          onChange={(event) => setTime(event.target.value)}
        />
      </div>
      <DayPicker idPrefix="new-alarm-day" value={days} onChange={setDays} disabled={disabled} />
      <p className="text-label text-muted-foreground">Pick no days to ring once.</p>
      {error ? <p className="text-label text-destructive">{error}</p> : null}
      <SubmitButton onSubmit={handleSet} disabled={disabled || time.length === 0} pendingLabel="Setting…">
        Set alarm
      </SubmitButton>
    </div>
  )
}

export function TimersRoute() {
  const timers = useQuery(timersQueryOptions)
  const screen = deriveTimersScreenState({ timers })
  const [nowMs, setNowMs] = React.useState(() => Date.now())

  React.useEffect(() => {
    const handle = setInterval(() => setNowMs(Date.now()), 1000)
    return () => clearInterval(handle)
  }, [])

  const controlsDisabled = screen.kind !== "ready"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-col gap-1">
        <h1 className="text-display font-semibold">Timers and alarms</h1>
        <p className="text-body text-muted-foreground">
          Timers and alarms ring on the house speaker. You can also set them by voice.
        </p>
      </div>

      <NewTimerPanel disabled={controlsDisabled} />
      <NewAlarmPanel disabled={controlsDisabled} />

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void timers.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        <>
          <div className="flex flex-col gap-2">
            <h2 className="text-heading font-semibold">Timers</h2>
            {screen.timers.length === 0 ? (
              <EmptyState heading="No timers running." body="Start one above, or say it out loud." />
            ) : (
              <ul className="panel-list">
                {screen.timers.map((timer) => (
                  <TimerItem key={timer.id} timer={timer} nowMs={nowMs} />
                ))}
              </ul>
            )}
          </div>
          <div className="flex flex-col gap-2">
            <h2 className="text-heading font-semibold">Alarms</h2>
            {screen.alarms.length === 0 ? (
              <EmptyState heading="No alarms set." body="Set one above, or say it out loud." />
            ) : (
              <ul className="panel-list">
                {screen.alarms.map((alarm) => (
                  <TimerItem key={alarm.id} timer={alarm} nowMs={nowMs} />
                ))}
              </ul>
            )}
          </div>
        </>
      ) : null}
    </div>
  )
}
