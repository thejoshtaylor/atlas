# Recording the wake-word corpus and scoring both engines

This runbook closes requirement DBG-04: the shipped wake engine must come from a
score against real audio, not from an argument about narrowband speech recognition.
Follow this document from top to bottom. It takes about an hour in the room with the
camera.

No fixture and no synthetic corpus can do this job. A generated wake phrase scores
the engines on audio no operator will ever produce in real use. A threshold tuned
against a fake recording carries false confidence, which is worse than no tuning at
all. The only trustworthy corpus is a real recording of this camera, in this room,
of a real person saying the wake phrase.

## Before you start

Confirm three things:

1. A Vosk model exists at the path `config.example.yaml` names under `wake.vosk.model_path`
   (for example `/models/vosk-model-small-en-us-0.15`). Without it, both engines are
   unavailable and a scoring run produces a report with no comparison in it.
2. `.env` holds real `TAPO_USER` and `TAPO_PASSWORD` values, and the camera answers on
   the network.
3. You have a quiet hour, and ideally a second speaker. The corpus floor below needs
   more than one voice.

## What to record, and why the floor is what it is

Record two things: a set of positive recordings (you saying the wake phrase) and one
continuous negative recording (everything else).

The floor is **20**: at least 20 positive recordings, spanning more than one distance
from the camera and more than one speaker, plus a negative recording that runs long
enough to include the television playing and ordinary room speech that is not the
wake phrase. `scripts/score_wake_engines.py` checks this floor and reports a corpus
below it as provisional, however clean its scores look. Do not try to beat the floor
check with a thin corpus. It exists because a small sample proves nothing about false
accepts, and a false accept is a house that acts on nothing you said.

`scripts/capture_wake_corpus.py` prints the position you are recording under, and the
running count for it, before every single capture. Watch that line. It is the only
defense against the corpus's worst failure mode: moving to a new spot in the room
without telling the script, so every later recording is labeled under the wrong
position and nothing in the audio itself reveals the mistake.

## Recording the corpus

Run these commands from the repository root. `scripts/dev-capture-corpus.sh` sources
`.env` for you and hands off to `capture_wake_corpus.py`; it never prints a credential
value.

1. Start a positive recording session:

   ```bash
   scripts/dev-capture-corpus.sh positive
   ```

2. When the script asks for a distance, answer with a short label such as `close`,
   `across the room`, or `far corner`. When it asks for a speaker, give a name or
   initial.
3. Press Enter to record each utterance. Say "hey spire" clearly, then wait for the
   next prompt. Watch the printed position label and count after each recording.
4. Type `n` to declare a new position (a new distance, a new speaker, or both), or
   `q` to finish this session once you have covered at least two distances and two
   speakers with 20 or more recordings in total.
5. Start the negative recording:

   ```bash
   scripts/dev-capture-corpus.sh negative --seconds 180 --note "television plus ambient talk"
   ```

6. During this recording, turn on the television, let it play, and speak normally in
   the room without ever saying the wake phrase. Press Ctrl-C to end the recording
   early if you need to; the script keeps what it already captured.

Recordings and their metadata land in `data/wake_corpus/`, a directory the repository
already ignores. Nothing you record here reaches git.

## Running the scoring harness

Score both engines against the corpus you just recorded:

```bash
PYTHONPATH=src:mcp .venv/bin/python scripts/score_wake_engines.py \
  --report-out data/wake_corpus/report.json
```

This prints the floor check and each engine's result to the terminal, and writes the
same information as a durable JSON file at the path you gave `--report-out`. The
report never contains audio, transcript text, or anything derived from the camera's
RTSP URL. It exists so a run can be reread and compared later, not just watched once
and forgotten.

## Reading the report

Open the file `--report-out` named. Look at three fields:

- `engines`: one entry per engine. An engine that could not run appears here too,
  with `available: false` and `unavailable_reason` naming the missing file — most
  likely openWakeWord, which has no "hey spire" model in this project yet. Training
  one is an offline machine-learning pipeline outside this codebase's scope, so a
  report naming openWakeWord as unavailable is an expected, honest outcome, not a
  bug. A run where only one engine could run compares nothing; it names one engine
  that works.
- `floor_met`: `true` only when your corpus reached the stated floor.
- `classification`: either `measured` or `provisional`. This is the field the next
  section acts on.

A `measured` classification requires all of: the floor was met, both engines
actually ran, and their results are different enough to prefer one over the other. A
`provisional` classification means one of those three did not hold — read
`classification_reasons` in the same file for exactly which one. A provisional
result is a real, honest outcome. It is not a failure of this runbook, and it is not
something to explain away.

## Closing the requirement

Only if `classification` reads `measured`, make these two edits together:

1. In `config/config.example.yaml`, change the `wake-default-evidence` comment line
   inside the `wake:` block from `provisional` to `measured`, adding the date, the
   positive count, and the negative duration from the report, in this exact shape:

   ```
   # wake-default-evidence: measured date=2026-09-20 positives=24 negative_duration_s=185.3
   ```

   Also set `wake.engine` to whichever engine the report recommends.
2. In `.planning/REQUIREMENTS.md`, change DBG-04's checkbox from `[ ]` to `[x]`.

Do the second edit without the first, or the first without the second, and
`tests/test_score_wake_engines.py` fails the whole suite. That test is the mechanism
this runbook exists to serve: it reads both files and refuses to let the
requirement's mark and the configuration's evidence line disagree. Run the full test
suite after making both edits, and confirm it passes before you consider DBG-04
closed.

If the report reads `provisional`, make neither edit. Leave the evidence line and the
requirement exactly as they are, and record what the report said in this phase's
plan summary: the positive count, the negative duration, and each engine's result. A
provisional result recorded honestly is worth more than a measured claim recorded
from a corpus that never existed.
