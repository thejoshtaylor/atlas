# ATLAS

**A**ssistant for **T**asks, **L**ogistics, **A**utomation, and **S**cheduling.
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

The speech-to-text figure is measured against clean synthesized 16 kHz
speech, not camera audio: a real camera turn also pays an A-law decode and
a resample this figure does not include.

A different host will measure differently. See the runbook for the full
report and the command to re-run it on your own host.

## Deployment

Two supported ways to run this for real, both with no cloud account required
for the database and no file to hand-edit for the database or the secret key:

- **Docker Compose** — one command, a bundled Postgres, reachable at
  `http://127.0.0.1:8080` by default. See
  [`docs/runbooks/deploy-compose.md`](docs/runbooks/deploy-compose.md).
- **Helm** — a chart with no third-party dependency, for a Kubernetes cluster
  you already run. See
  [`docs/runbooks/deploy-helm.md`](docs/runbooks/deploy-helm.md).

Both runbooks state plainly what port is published and what that protects, how
the bundled database's password works, how the session/credential-encryption
key is generated once and must be backed up, and what a Kubernetes deployment
assumes about TLS. `scripts/verify-clean-clone.sh` proves the Compose runbook's
own commands work from a real, fresh clone — run it yourself if you want to see
that proof rather than take the document's word for it.

### Sharing a node with other workloads

The Helm chart gives the ATLAS container a CPU request of one core and a
memory request of 1 GiB. Change these through `resources` in your own values
file. Lower them on a small node.

The chart sets no CPU limit on purpose. A CPU limit throttles the process, and
that delays audio and replies.

A request keeps a share of CPU for ATLAS when the node is busy. It does not
stop other pods from using the rest. If other heavy workloads run on the same
node, for example CI runners or video recording, give each of them a CPU
limit.

`priorityClassName` is optional. Set it to the name of a PriorityClass that
you create yourself. The scheduler then places ATLAS ahead of lower-priority
pods, and can evict them to make room. The chart does not create the
PriorityClass.

Docker Compose reserves the same amount of memory. Compose has no CPU
reservation outside Swarm mode. Docker already gives every container an equal
CPU weight by default.

## Licence

This project is licensed under the MIT License.

Piper, the optional local text-to-speech package (`piper-tts`), is licensed
under GPL-3.0, not MIT. It is not installed by default:

```bash
pip install 'atlas[piper]'
```

If you install it, and you redistribute a build that includes it, GPL-3.0
applies to that build. See
[`docs/runbooks/local-providers.md`](docs/runbooks/local-providers.md) for the
full note.
