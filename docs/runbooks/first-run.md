# First run: from a clean install to a working assistant

This is the walk a stranger takes the first time they open atlas. Deploying it
with Helm or Docker Compose is a separate step, covered by Phase 7's own documentation
-- this runbook starts once the process is up and you open it in a browser.

The wizard is at `/setup`. It has no other entry point, and it is the only screen a
clean install offers: every other address in the application answers "atlas
isn't set up yet" until an admin exists. There is no default password and no printed
setup token anywhere -- the admin account you create in the first step is the only way
in.

## Before you start

Have these ready. The wizard will ask for all of them, in this order.

1. **A password you are choosing yourself**, for the one admin account this install
   will ever create through the browser. Nothing is generated for you and nothing is
   printed to a log.
2. **A Home Assistant address and a long-lived access token.** The address is set at
   deploy time (the `HA_URL` environment variable in your `.env` or Helm values) --
   the wizard's hub step shows a field for it so you can confirm it matches what you
   configured, but that field is not itself where the address is set. The token is
   generated from your Home Assistant profile page and is what the wizard actually
   saves.
3. **API keys for your speech-to-text, language model, and text-to-speech
   providers.** Whichever providers STACK.md and your `config.yaml` name. If you
   already set these as environment variables on an existing deployment, the wizard
   recognizes them as already set and will not ask you to re-enter them.
4. **A working camera and speaker in the room you actually use.** The last step plays
   a sound through the speaker and records it back through the microphone -- you need
   to be standing where you will actually talk to the assistant, not next to the
   camera.

## What each step asks for, and why

The wizard is five steps. You always land on the first one you have not finished --
closing the tab and coming back later, or signing out and back in, resumes there, never
back at step one and never locked out.

1. **Create the admin account.** Email, a display name, and the password you chose
   above. This is the only account this route will ever create; a second attempt
   after the first succeeds is refused by name, not by a generic failure.
2. **Connect Home Assistant.** Enter the token, save it, and press Continue. The
   wizard makes a real call to Home Assistant using the exact read the assistant
   itself performs at startup -- a hub that passes this check is a hub the assistant
   can actually use. If it fails, the message names one of three things: the address
   could not be reached, the token was rejected, or the response could not be
   understood. Each sends you to a different fix -- read it, it is written to be
   acted on.
3. **Add your provider keys.** One field per key. A key already set from your
   environment shows as set, with nothing further to do. Pressing Continue with a key
   still blank names exactly which one is missing rather than silently refusing to
   move on.
4. **Choose the audio source.** Today there is one: the camera. The badge on this step
   says whether the choice takes effect immediately or needs a restart -- audio
   source changes need a restart, so plan to restart the process once setup finishes
   if you want this to apply before the first conversation.
5. **Test the microphone and speaker.** Covered on its own below.

## Testing the microphone and speaker

This step runs the same measurement `docs/runbooks/echo-path-calibration.md`
describes in full -- read that document for what the three numbers mean and what to
do with them. The short version: stand where you normally speak to the assistant,
write down where you stood in the placement note, and press the control that starts
the test. A short sound plays through the speaker; the microphone records the room for
a few seconds afterward.

The wizard cannot be finished before this step produces a real, stored result. This is
not a UI restriction alone -- the finish action asks the server, and the server
refuses while any step (including this one) is outstanding, so skipping the test is
not a way around it.

If a run fails, the screen names why: the route is disabled, another run is already
in progress, or the run completed but measured nothing usable. Fix the named cause and
press the control again.

## Finishing

Once every step holds, press Finish setup. If anything is still outstanding, the
refusal names every step that is, not just the first -- fix all of them and press it
again. On success you land in the application itself, with a short note on what to do
next: say the wake phrase and confirm the assistant answers. A finished wizard is a
form filled in; hearing a real reply is what confirms you have a working assistant.

## When a step refuses

Every refusal in this wizard names a real, specific reason -- never a bare "something
went wrong." If a step refuses:

1. Read the message. It is written to send you to what to fix, not just that
   something failed.
2. Fix the named cause (a wrong token, an unreachable address, a missing key) and
   press the same control again. Nothing you have already completed is lost --
   step state lives on the server, not in your browser tab.
3. If the message itself does not tell you what to do, that is a gap in this
   runbook or in the wizard's own copy, not something you did wrong. Write down the
   exact text and where it happened.

## After the wizard

Sign back in at any time to revisit settings, the safety policy, or accounts --
finishing the wizard does not lock you out of changing anything later. The developer
microphone page (`/dev-mic`, reachable from the navigation once signed in) is a
separate, authenticated diagnostic tool for testing a single voice turn without a
camera or a wake word -- it is not part of first-run setup and most operators will
never need it.
