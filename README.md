# spire-voice

A self-hosted voice assistant for the home.

A local wake word starts it. Speech recognition, reasoning, and speech synthesis
then run through providers you choose, cloud or local. Every extra ability it
reaches is an MCP server, so you add a new ability without changing this
project's own code. A React admin webapp configures all of it.

**Core value:** a spoken sentence makes your house do the right thing, fast enough
to feel like a reply, and the assistant never touches what you marked off limits.

## What it needs

- A camera that streams RTSP audio, or a browser microphone for development.
- Python 3.11 or newer, and Postgres.
- Node/bun, to build the admin webapp.
- A speech-to-text provider, a language-model provider, and a text-to-speech
  provider, one of each. Pick a fully cloud set, a fully local set, or a mix.

The fully local set needs no cloud account and no API key. See
[Local providers](#local-providers) below.

## Running it for development

1. Copy `.env.example` to `.env` and fill in the values it names. `.env` is
   gitignored. Nothing in it belongs in git — this repository is public.
2. Start a throwaway local Postgres:

   ```bash
   eval "$(scripts/dev-postgres.sh)"
   ```

3. Create a Python virtual environment and install the backend:

   ```bash
   python3 -m venv .venv
   .venv/bin/pip install -e '.[dev]'
   ```

4. Build the admin webapp:

   ```bash
   cd web && bun install && bun run build && cd ..
   ```

5. Run the backend:

   ```bash
   scripts/dev-run.sh
   ```

Run the test suite with `.venv/bin/python -m pytest`. See `docs/runbooks/` for
the wake-word corpus, echo calibration, and other one-time setup steps.

## Local providers

A fully local provider set runs the whole assistant with no cloud account:

- **Speech to text:** `faster-whisper`, installed by default.
- **Text to speech:** Piper, an optional install. See [Licence](#licence).
- **Language model:** any server that speaks the OpenAI-compatible chat API,
  running on your own network.

Speech-to-text and text-to-speech both need a model file on disk before you
select them. One command provisions both:

```bash
scripts/dev-fetch-models.sh
```

Nothing in this project downloads a model at boot. See
[`docs/runbooks/local-providers.md`](docs/runbooks/local-providers.md) for the
full setup and the licence note.

**The local set is slower than the cloud set, and this is a plain, published
fact, not a target this project claims to meet.** Measured on a 12-core Apple
M4 Pro, CPU only, no GPU, on 2026-09-20:

| Stage | Median |
|---|---|
| Speech to text (`faster-whisper`, small, int8) | 1238 ms |
| Text to speech (Piper, medium quality) | 112 ms |

A different host will measure differently. See the runbook for the full
report and the command to re-run it on your own host.

## Deployment

Deployment (Helm chart, Docker Compose file) is documented separately once
that work lands — this section is a placeholder.

## Licence

This project is licensed under the MIT License.

Piper, the optional local text-to-speech package (`piper-tts`), is licensed
under GPL-3.0, not MIT. It is not installed by default:

```bash
pip install 'spire-voice[piper]'
```

If you install it, and you redistribute a build that includes it, GPL-3.0
applies to that build. See
[`docs/runbooks/local-providers.md`](docs/runbooks/local-providers.md) for the
full note.
