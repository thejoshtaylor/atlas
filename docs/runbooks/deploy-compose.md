# Deploying with Docker Compose

This is the full walk from a clean clone to a running assistant, reachable in a
browser. Everything here is a command you run -- nothing asks you to hand-edit a
file, and every value below is invented for illustration, never a real address or
credential.

## What you need first

- Docker and the `docker compose` plugin (Compose v2 -- `docker compose version`,
  not the older standalone `docker-compose`).
- Nothing else. The database is bundled; you do not install or configure Postgres
  yourself.

## Bring it up

```bash
git clone https://github.com/<your-fork-or-this-repository>/spire-voice.git
cd spire-voice
docker compose up -d --build --wait
```

This builds the image, starts the bundled database, runs migrations, and waits
for the application to report healthy. `.env` is optional -- every variable
`docker-compose.yml` reads has a fallback, so this works on a clean clone with no
`.env` file at all. If you do want to set a provider API key or a Home Assistant
address before the first boot, copy `.env.example` to `.env` and fill in what you
know; anything left unset there stays a "not yet configured" slot you fill in from
the webapp instead (see [Providers, credentials, and restarts](#providers-credentials-and-restarts)
below).

## The wake-word model -- one command, before the first start

The application's default configuration listens for the phrase "hey spire" using
Vosk, a wake-word engine that needs a model file on disk. Nothing in this project
downloads that model automatically -- an offline deployment must not need the
internet at a moment you did not choose. Provision it once, before the first
`docker compose up`, with the image you just built:

```bash
docker compose run --rm app python scripts/fetch_wake_model.py --config config/config.example.yaml
```

This downloads the model into the same `spire-models` volume the running
application reads from, and prints `already present` on every run after the
first -- safe to run again, and safe to run before `docker compose up` has ever
started the `app` service. Skipping this step is the one way a fresh clone fails
to start: the application refuses to boot with a message naming the missing model
directory, by design, rather than pretending to listen for a wake word it cannot
detect.

## Open it

```bash
curl -fsS http://127.0.0.1:8080/health
```

Then open `http://127.0.0.1:8080` in a browser -- the first-run wizard greets
you, since no admin account exists yet. See
[`docs/runbooks/first-run.md`](first-run.md) for what it asks and why.

## What this deployment actually does, stated plainly

**The published port is bound to this host's own loopback address, not every
network interface.** `docker-compose.yml` publishes `127.0.0.1:8080:8080` by
default -- nothing off this machine can reach it. This is what lets the session
cookie ship without the `Secure` flag out of the box: the cookie never crosses a
real network. If you widen this (for example to `0.0.0.0:8080:8080` so another
device on your network can reach it), you must also set `COOKIE_SECURE=true` and
put a TLS-terminating proxy in front of this port -- an admin session cookie
travelling in the clear across a real network is a credential leak waiting to
happen. `docker-compose.yml`'s own comments name this trade-off at the line that
makes it.

**The database is bundled. You never provide one.** `docker-compose.yml` starts a
`postgres:18` container for you, reachable only on the internal Compose network
(no port is published for it). Its password defaults to the obviously-invented
placeholder `changeme`, set via the `POSTGRES_PASSWORD` environment variable --
change it before you expose this deployment beyond your own machine, by setting
`POSTGRES_PASSWORD` in your `.env` before the first `docker compose up` (Postgres
only reads this variable when its data volume is empty; changing it later needs a
matching change to the database's own stored password, not just the environment
variable).

**The secret key is generated once, on first boot, and persisted -- back it up.**
This one value derives both the session-signing key and the key every stored
provider credential is encrypted with. `deploy/docker-entrypoint.sh` generates it
the first time no key is found, writes it to the `spire-data` volume, and reuses
that same file on every later start -- confirmed by restarting the container and
comparing the file byte-for-byte. If that volume is lost with no backup, every
stored credential becomes permanently unreadable and every signed-in session is
invalidated at once. Back up `spire-data` the way you would any other secret. You
can also set `SPIRE_SECRET_KEY` yourself in `.env` before the first start if you
would rather manage it outside this project's own generation step; see
`.env.example`'s own comment for the exact command that generates a valid one.

## Providers, credentials, and restarts

Provider credentials (a speech-to-text key, a language-model key, a
text-to-speech key, a Home Assistant token) are entered from the webapp's
Providers and Settings screens once you have signed in -- never from a file you
edit. **Changing which provider a slot uses needs a restart of the application
container**, and the screen says so: your choice is saved immediately, but the
slot keeps running whichever provider it started with until you restart
(`docker compose restart app`).

## The local provider set

A fully local provider set needs no cloud account at all -- see
[`docs/runbooks/local-providers.md`](local-providers.md) for what it is, how to
provision its own model files, and the measured latency you should expect on a
machine with no GPU: it is slower than the cloud set, published as a real number
in that runbook, not a target this project claims to meet. If you select Piper
for text-to-speech, read that runbook's licence note first -- it is an optional
package under a different licence than the rest of this project.

## Running more than one instance on the same host

`scripts/verify-clean-clone.sh` (used to prove this document is true; see
[Verifying this document is true](#verifying-this-document-is-true) below) needs
to run alongside a stack you may already have up, so it overrides the Compose
project name and the published port:

```bash
COMPOSE_PROJECT_NAME=spire-voice-second SPIRE_PORT=8081 docker compose up -d --build --wait
```

Use the same override if you want a second, independent instance of your own.

## Stopping it

```bash
docker compose down
```

This stops the containers and keeps every named volume (`spire-data`,
`spire-models`, `spire-db-data`) -- your data, your models, and your secret key
all survive. Add `-v` only if you intend to discard all of it, including the
database.

## Verifying this document is true

This document is proved, not merely written: `scripts/verify-clean-clone.sh`
clones this repository into a fresh temporary directory and runs exactly the
commands above -- nothing else -- then checks the health route and the served
page, and tears the whole thing down whether it succeeds or fails. Run it
yourself:

```bash
./scripts/verify-clean-clone.sh
```

If any command here ever stops working from a real clean clone, that script
fails loudly rather than this document quietly going stale.
