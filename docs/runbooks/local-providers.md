# Running the fully local provider set

This runbook covers PROV-06: a provider set that needs no cloud account. Follow it
before you select a local provider on the Providers screen.

## What the local set is

The local set has three parts, one per provider slot.

- **Speech to text**: `faster-whisper`, a speech recognizer that runs on the CPU.
- **Text to speech**: Piper, a speech synthesizer that runs on the CPU. Piper is
  optional. See the licence note below before you install it.
- **Language model**: any server that speaks the OpenAI-compatible chat API, for
  example `llama.cpp`'s `llama-server` or Ollama. You run this server yourself, on
  your own network. Enter its URL on the Providers screen.

None of the three needs an account, an API key, or a network call outside your own
network.

## Which models it needs

The speech-to-text and text-to-speech parts need model files on disk before you
select them. The language-model part needs no model file here, because the server
you run holds its own model.

`config/config.example.yaml` names the paths under a `/models` root:

- A `faster-whisper` model directory (`stt.local_model_dir`, size set by
  `stt.local_model_size`).
- A Piper voice file and its matching configuration file (`tts.piper_voice_path`,
  `tts.piper_config_path`).

## How to provision them

Run one command from the repository root:

```bash
scripts/dev-fetch-models.sh
```

This downloads the files named above into the paths your configuration file
names, and prints which files it wrote and which it found already there. Run it
again at any time. A file already present is left alone, so a second run after an
interrupted first run costs nothing.

**Every file it writes is checked against a sha256 pinned in
`scripts/fetch_models.py`.** These are published release artifacts whose bytes do
not change, and each one is handed straight to a native extension
(`ctranslate2`, `onnxruntime`) once it lands -- so a file that does not match is
deleted rather than kept. A consequence worth knowing before you hit it: the
script fetches only the `faster-whisper` sizes this repository has pinned
(`small` and `base`). Point `stt.local_model_size` at another size and the
script refuses by name rather than downloading something it cannot verify; add
that size's digests to `_PINNED_SHA256`, checked against the source yourself, if
you want one.

The wake-word model (`scripts/fetch_wake_model.py`, a separate step) is pinned
the same way, and its archive is verified before it is opened at all.

Nothing else in this project downloads a model. The application never fetches a
model at boot. If a model file is missing, the matching provider slot on the
Providers screen shows as degraded, and names the missing file.

If you prefer to call the script directly, set `PYTHONPATH=src:mcp` and pass
`--config` with your configuration file:

```bash
PYTHONPATH=src:mcp .venv/bin/python scripts/fetch_models.py --config config/config.example.yaml
```

## What to select on the Providers screen

Open the Providers screen as an admin, and for each slot pick the local option:

1. **Speech to text**: select `faster-whisper (local)`.
2. **Text to speech**: select `Piper (local)`, if you installed the optional
   Piper package (see below).
3. **Language model**: select `Self-hosted (local)`, and enter the URL of your
   own OpenAI-compatible server in the Server URL field. You can save this
   choice with the field left blank. The slot then reports itself degraded, by
   name, until you fill in the URL and restart.

A provider choice takes effect after a restart. The screen states this.

## Piper's licence

Piper ships as an optional package, not installed by default. Its package,
`piper-tts`, is licensed under GPL-3.0, not the licence the rest of this project
uses. If you install it, and you redistribute a build that includes it, GPL-3.0
applies to that build. Install it only if you accept this:

```bash
pip install spire-voice[piper]
```

See the repository's `README.md` for this project's own licence.

## Measured latency

This is a real measurement, not a target. This project's reply-time budget is
already about seven times over on the cloud path, and the local set is slower
still. Nothing about the local set is gated on that budget. What is owed here
is the honest number.

Run this once both models are provisioned:

```bash
PYTHONPATH=src:mcp .venv/bin/python scripts/measure_local_providers.py
```

It runs the provisioned Piper voice over a fixed sentence and the provisioned
faster-whisper model over that same sentence's own synthesized speech, five
times each, and prints the median, minimum, and maximum wall-clock time per
stage.

**Measured 2026-09-20, on a 12-core Apple M4 Pro, CPU only, no GPU or other
accelerator used** (`faster-whisper` "small", int8; Piper "en_US-lessac-medium",
medium quality; 5 repetitions each):

| Stage | Median | Min | Max |
|---|---|---|---|
| Speech to text (`faster-whisper`) | 1238 ms | 1203 ms | 1276 ms |
| Text to speech (Piper) | 112 ms | 105 ms | 472 ms |

**The speech-to-text figure is measured against a synthetic microphone.** The
harness hands the recognizer clean 16 kHz speech that Piper has just
synthesized, so it does not include the A-law decode and resample every real
camera turn pays before the recognizer sees anything. Treat it as a floor for
that stage, not as the camera path's own number.

This host is a development machine, not this project's target deployment
host. A different host will measure differently. Re-run the command above on
your own deployment host and use that number, not this one, to judge whether
the local set is fast enough for your use.
