# Testing the microphone and speaker (echo-path calibration)

This runbook closes the first of Phase 2's two open gaps. Camera barge-in (VOICE-07)
ships disabled because no real echo-path calibration exists. `app.py` refuses to start
with correlation enabled and no valid calibration on file, so this cannot be half-done
silently. Follow this document to take a real measurement, then move to the next step.

The measurement itself has one implementation: `run_echo_calibration`
(`src/atlas/calibration/runner.py`). Two things call it. This runbook uses the
browser, at `/calibration`. `scripts/dev-calibrate-echo.sh` runs the same measurement
from the command line, with no browser at all. Use whichever is easier to reach. Both
write the same record and neither is a weaker copy of the other.

## Before you start

1. Confirm `.env` holds real `TAPO_USER` and `TAPO_PASSWORD` values, and the camera
   answers on the network.
2. Confirm `calibration.route_enabled: true` is set in your configuration. The route
   answers 403 by name until you set it. This flag is off by default because the
   route makes a real speaker play a sound and a real microphone record the room.
3. Sign in to the webapp as an admin or operator, and open `/calibration` on the
   device you will use, at the width you will actually use it.

## Where to stand

Stand where you normally speak to the assistant. Do not stand next to the camera to
get a cleaner reading. A calibration measures the room you will actually use, not the
best case the room can produce.

Write down where you stood in the placement note field before you press the control.
Two words are enough: "kitchen counter" or "across the room." The note travels with
the measurement, so a later operator reading the result knows what room it describes.

## Running the test

Press the control that starts the test. The screen says it is testing the microphone
and the speaker, and the control stays disabled until the test finishes. A short
probe sound plays through the speaker. The microphone records the room for a few
seconds after it. Nothing repeats and nothing else in the house reacts during this
window.

If the screen reports a failure, read its message. A disabled-route message names
the configuration key to set. An in-progress message means another run is already
under way. Any of them, correct the cause and press the control again.

## Reading the three numbers

The screen shows three measured results, in words rather than raw field names.

- **Round-trip delay.** How long the assistant's own voice takes to leave the
  speaker, travel the room, and return to the microphone. This is the delay Tier 2
  correlation shifts its comparison by, so barge-in detection lines up the assistant's
  own echo against what it actually said.
- **Arrival level.** How loud the assistant's own voice is by the time it returns to
  the microphone, relative to what it played. A low value means the echo is faint
  next to the room's own noise floor. A near-zero result across repeated runs is a
  sign the microphone may not be picking up the speaker well enough for barge-in to
  work reliably here.
- **Automatic gain control.** Whether the camera quietly changes the microphone's
  level while it plays back. Three outcomes exist: detected, not detected, or could
  not be determined. This number decides how Tier 2 correlation runs, described next.

The confidence beside the three numbers is how strongly the recording actually
matched the probe sound. A low confidence with an otherwise plausible delay is worth
a second run before you trust the result.

## What the result decides

The automatic gain control verdict decides which mode `barge_in`'s correlation runs
in for the camera:

- **Not detected.** Correlation runs in fixed mode. The camera holds a steady level,
  so a single measured gain stays valid.
- **Detected, or could not be determined.** Correlation runs in tracking mode. The
  gain estimate adjusts over time (`barge_in.tracking_adaptation_rate` controls how
  fast), because a single fixed number would drift out of date as the camera's own
  gain moves.

You do not choose the mode yourself. The stored calibration's verdict chooses it the
next time the process starts.

## Closing VOICE-07

Only after you have a result you trust, make these two configuration edits together:

1. Set `barge_in.correlation_enabled: true`.
2. Set `barge_in.sources.camera.enabled: true`.

Restart the process. Startup refuses to boot if step 1 is true with no valid,
non-stale calibration on file, so a missing or expired measurement stops the process
loudly rather than running uncalibrated.

With both settings on, talk over a reply. Say something else while the assistant is
mid-sentence, at a normal speaking volume, from where you stood for the calibration.
The reply should stop.

Write down what happened, met or missed. If the reply keeps talking over you, the
floor or guard window may need a second look. If the reply cuts off with nobody
speaking, that is the one piece of evidence this project's context treats as a
reason to revisit full acoustic echo cancellation. Record it as that evidence, not as
a tuning note, and say so in the phase summary that closes VOICE-07.

A calibration expires after `calibration.max_age_days` (30 by default). Repeat this
runbook once the stored measurement goes stale, or after any change to where the
camera or its speaker sit in the room.
