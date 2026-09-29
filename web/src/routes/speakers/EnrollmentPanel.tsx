import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Button } from "@/components/ui/button"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { ApiError } from "@/lib/api"
import { edgeDevicesQueryOptions } from "@/lib/edgeDevices"
import { enrollPhraseMutationOptions, enrollmentPhrasesQueryOptions, type Speaker } from "@/lib/speakers"
import { deriveEnrollmentState, enrollmentReducer, initialEnrollmentState } from "./deriveEnrollmentState"

// D-02, D-03: one prompted phrase at a time, read into the real Pi
// microphone through the enroll-phrase route -- never the browser's own
// microphone capture API, never a file input. Opens from a member row in
// `SpeakersRoute.tsx` (Task 3), one panel per open, always starting at
// phrase 0 (see `deriveEnrollmentState.ts`'s own note on why "Enroll" and
// "Re-record phrases" share that starting point).

export function EnrollmentPanel({ speaker }: { speaker: Speaker }) {
  const devices = useQuery(edgeDevicesQueryOptions)
  const phrases = useQuery(enrollmentPhrasesQueryOptions)
  const enroll = useMutation(enrollPhraseMutationOptions)

  const [reducerState, dispatch] = React.useReducer(
    enrollmentReducer,
    initialEnrollmentState(speaker.required_phrases),
  )

  const connectedDevices = (devices.data ?? []).filter((device) => device.connected)
  const phraseList = phrases.data?.phrases ?? []
  const screen = deriveEnrollmentState(reducerState, phraseList)

  const handleRecord = async () => {
    if (reducerState.deviceId === null) return
    dispatch({ type: "start_recording" })
    try {
      const result = await enroll.mutateAsync({
        speakerId: speaker.id,
        phraseIndex: reducerState.phraseIndex,
        deviceId: reducerState.deviceId,
      })
      dispatch({ type: "record_success", speechMs: result.speech_ms, enrolledPhrases: result.enrolled_phrases })
    } catch (error) {
      const message = error instanceof ApiError ? error.message : "Couldn't reach the server. Try again."
      dispatch({ type: "record_failure", message })
    }
  }

  if (connectedDevices.length === 0) {
    return (
      <p className="text-label text-muted-foreground">
        Connect a Pi microphone first. The Edge devices screen shows its status.
      </p>
    )
  }

  if (screen.kind === "complete") {
    return <p className="text-body text-foreground">Enrolled. This member is recognized on the next turn.</p>
  }

  if (screen.kind === "choose_device") {
    return (
      <RadioGroup
        value={reducerState.deviceId !== null ? String(reducerState.deviceId) : undefined}
        onValueChange={(value) => dispatch({ type: "select_device", deviceId: Number(value) })}
        className="flex flex-col gap-2"
      >
        {connectedDevices.map((device) => (
          <div key={device.id} className="flex items-center gap-3">
            <RadioGroupItem value={String(device.id)} id={`enrollment-device-${device.id}`} />
            <Label htmlFor={`enrollment-device-${device.id}`}>{device.name}</Label>
          </div>
        ))}
      </RadioGroup>
    )
  }

  return (
    <div className="flex flex-col gap-3">
      {screen.kind === "ready" || screen.kind === "recording" ? (
        <>
          <p className="text-body text-foreground">Read this phrase now, in your normal voice:</p>
          <p className="text-heading font-semibold">{screen.phrase}</p>
          {screen.kind === "recording" ? (
            <p role="status" aria-live="polite" className="text-label text-muted-foreground">
              Listening...
            </p>
          ) : null}
          <Button
            type="button"
            onClick={() => void handleRecord()}
            disabled={screen.kind === "recording"}
          >
            Record phrase
          </Button>
        </>
      ) : null}
      {screen.kind === "recorded" ? (
        <>
          <p className="text-body text-foreground">Recorded {(screen.speechMs / 1000).toFixed(1)}s</p>
          <Button type="button" onClick={() => dispatch({ type: "next_phrase" })}>
            Continue
          </Button>
        </>
      ) : null}
      {screen.kind === "failed" ? (
        <>
          <p className="text-label text-destructive">{screen.message}</p>
          <Button type="button" onClick={() => void handleRecord()}>
            Try again
          </Button>
        </>
      ) : null}
    </div>
  )
}
