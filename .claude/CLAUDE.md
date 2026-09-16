<!-- GSD:project-start source:PROJECT.md -->

## Project

**spire-voice**

spire-voice is a self-hosted voice assistant for the home. A local wake word starts
it. Speech, reasoning, and speech synthesis then run through providers the operator
chooses, which can be cloud services or local models. Every capability it can reach
is an MCP server, so an operator adds new abilities without a change to the core.
A React admin webapp configures all of it.

The first deployment is the author's house. The repository is public, and a stranger
must be able to deploy it with Helm or Docker Compose and set it up in the browser.

**Core Value:** A spoken sentence makes the house do the right thing, fast enough to feel like a
reply and not a request, and the assistant never touches what the operator marked
off limits.

### Constraints

- **Audio**: The camera gives 8 kHz narrowband mono with no echo cancellation. This
  sets the accuracy ceiling for every recognizer. Do not design around a better
  microphone that does not exist.
- **Latency**: The RTSP buffer and the end-of-speech detection are hard floors of
  about 350 ms together. A reply under one second is reachable. A reply under 500 ms
  is not, on this camera.
- **Hardware**: No GPU. Local speech models and local language models must run on the
  CPU, and the design must say when that is too slow.
- **Security**: The microphone hears the television, and a television can speak any
  sentence. Treat all transcribed text as untrusted input.
- **Public repository**: No entity name, address, credential, or network detail from
  the author's house belongs in git. House data lives in cluster configuration.
- **Tech stack**: Python and FastAPI on the server, React and bun in the browser,
  Postgres for state, Helm and Docker Compose for deployment. This matches the
  author's other projects.
<!-- GSD:project-end -->

<!-- GSD:stack-start source:research/STACK.md -->

## Technology Stack

## 1. Local wake word engines

| Technology | Version | Purpose | Why Recommended |
|---|---|---|---|
| **openWakeWord** | 0.6.0 (Feb 2024 — no release since; see caveat below) | Neural wake word detector, ONNX/TFLite models | Zero-shot custom phrase training pipeline, tiny models (~1-2MB), runs on `onnxruntime` CPU with no GPU. `pip install openwakeword`. **Confidence: HIGH** (version/format verified on [GitHub releases](https://github.com/dscripka/openWakeWord/releases) and [README](https://github.com/dscripka/openWakeWord)). |
| **Vosk** (`vosk` PyPI package) | 0.3.45 (Dec 2022 on PyPI; upstream `alphacep/vosk-api` repo shows commits into Dec 2025 and open PRs/issues through Aug 2026) | Kaldi-descended offline ASR, used here as a grammar-constrained wake detector | Kaldi's telephony heritage (Switchboard/Fisher-style 8kHz training data lineage) makes it far more tolerant of narrowband audio than a model trained on synthetic 16kHz speech. Constraining the decoder grammar to `["hey spire", "[unk]"]` turns a full ASR engine into a cheap, low-false-positive wake detector. **Confidence: MEDIUM** (PyPI version is stale-looking, but upstream repo activity is current — the PyPI package lag is a packaging-hygiene issue, not a project-health one). |

### The 8kHz narrowband question, explicitly

- **openWakeWord** — trained on 16kHz audio, and its melspectrogram front-end expects
- **Vosk** — Kaldi acoustic models trained on telephony corpora already model exactly
- **Practical takeaway for the roadmap**: budget a phase task for recording real

### What else exists, and why it wasn't chosen

| Engine | Why not chosen |
|---|---|
| **Porcupine** (Picovoice) | Proprietary, requires a Picovoice account and `AccessKey` even for fully local/offline inference — this breaks "a fully local provider set works with no cloud account" (PROJECT.md Active Requirements) and complicates redistribution in a public repo. Commercial terms are opaque past a free tier and require contacting sales. **Confidence: MEDIUM** (pricing page findings were vague; the AccessKey/account requirement itself is well documented). |
| **Mycroft Precise / precise-lite** | Upstream company (Mycroft AI) is defunct; OpenVoiceOS carries the code forward but has been actively demoting it in its own default wake-word chain in favor of openWakeWord, keeping precise-onnx only as a fallback for existing custom models. Picking a wake engine that its own community is migrating *away from* is the wrong direction. **Confidence: MEDIUM**. |
| **microWakeWord** | Purpose-built for ESP32 microcontrollers running ESPHome — it's a `.tflite` INT8 model plus a C++ inference loop meant to run *on the microphone device itself*, not as a Python package consuming an RTSP stream on a server. Architecturally the wrong shape for this pod (there's no ESP32 in this deployment; the audio source is a camera over the network). **Confidence: HIGH** — this is a category mismatch, not a quality judgment. |

## 2. Audio I/O in Python

| Technology | Version | Purpose | Why Recommended |
|---|---|---|---|
| **PyAV** (`av` on PyPI) | 18.1.0 (Aug 2026) | Read the RTSP audio-only stream continuously; demux G.711 A-law packets without transcoding | Binds directly to FFmpeg's C libraries (bundles FFmpeg 8.x in the wheel — no system FFmpeg needed), so there's no subprocess-per-restart overhead and no shell-quoting/argv surface. Requires Python ≥3.11. **Confidence: HIGH** (version/requirements verified directly on PyPI). |
| **ffmpeg subprocess** (long-lived, via `asyncio.create_subprocess_exec`) | system FFmpeg 6.x/7.x/8.x (whatever the base image ships) | Write the reply audio into the FIFO that feeds go2rtc | This is what `config.example.yaml`'s `speaker.fifo_path` design already assumes: **one long-lived ffmpeg process** reads the FIFO and stays connected to go2rtc, because a fresh ffmpeg per utterance costs 300-800ms of process-start latency — the single largest avoidable delay in the round trip. This is a subprocess, not PyAV, because the output side is trivial (write raw A-law bytes to a pipe; no demuxing/decoding needed) and a long-lived subprocess with its own supervision (`respawn_backoff_s` already in config) is simpler to reason about than keeping a PyAV output container open indefinitely across the FIFO's read-side reconnects. **Confidence: MEDIUM-HIGH** (this is standard practice for continuous audio egress pipelines; not something a single citation confirms, but it matches the config file's own stated design rationale). |

- **Read (mic):** PyAV, because RTSP demuxing/decoding of A-law packets, PTS handling,
- **Write (speaker):** raw subprocess + FIFO, because the job is "keep a pipe open and

### Ring-buffering pre-roll audio

## 3. Provider SDK choices (STT / LLM / TTS, cloud and local)

| Provider | Python client | Streaming? | Notes |
|---|---|---|---|
| **xAI (Grok)** — brain (LLM) | `openai` SDK (v1.x+) pointed at `base_url="https://api.x.ai/v1"` | Streaming chat completions, yes | xAI's chat API is wire-compatible with OpenAI's Chat Completions format, including `tools=[...]` function-calling schemas — this is exactly why `brain.base_url`/`brain.model` in `config.example.yaml` look like OpenAI config. An official `xai-sdk-python` package also exists (`xai_sdk.Client`/`AsyncClient`), but the OpenAI-compatible path is the lower-friction choice since MCP tool schemas need to become `tools=[...]` entries regardless of which client library builds the HTTP request. **Confidence: HIGH** (xAI's own docs/quickstart confirm OpenAI-client compatibility). Capture the "set `XAI_API_KEY` in the environment *before* constructing the client" gotcha — the OpenAI client reads the key at construction time. |
| **xAI STT** | Raw `websockets` library (no dedicated SDK method documented) | Yes — bidirectional websocket, binary audio frames in, JSON transcript events (`transcript.partial`, `transcript.done`) out | Accepts `encoding=alaw`, `sample_rate=8000` directly — this is the reason the whole camera→STT path in this project needs **zero transcoding**. `endpointing`, `vad_threshold`, and `smart_turn` are documented websocket parameters and map directly onto `config.example.yaml`'s `stt.*` block. **Confidence: HIGH** (verified via `docs.x.ai`). |
| **xAI TTS** | `requests`/`httpx` (REST) for batch, websocket for streaming deltas | Yes — text streamed in as deltas, base64 audio chunks streamed back | Supports `mulaw`/`alaw` at 8kHz output directly (again, zero transcoding to the camera speaker) and an `optimize_streaming_latency` (0/1/2) parameter matching `config.example.yaml`'s `tts.optimize_streaming_latency`. **Confidence: HIGH**. |
| **OpenAI** (alternative LLM/STT/TTS provider) | `openai` SDK, current: **3.3.1** (Aug 2026) | Realtime API supports streaming STT/TTS over websocket; Whisper/`gpt-4o-transcribe` for STT, `tts-1`/`gpt-4o-mini-tts` for TTS | Straightforward alternate provider behind the same interface; note OpenAI's raw audio formats are PCM16/G.711 µ-law/A-law depending on endpoint — check per-endpoint support before assuming zero-transcode parity with xAI. **Confidence: MEDIUM** (version verified; exact audio-format matrix not independently re-verified this session). |
| **Anthropic (Claude)** (LLM-only provider option) | `anthropic` SDK, current: **1.0.0** (Aug 20 2026 — major version bump from the 0.x line; check the SDK's own migration notes before pinning) | No native STT/TTS — LLM only | Fits the "brain" slot only; Anthropic has no first-party speech endpoints, so this provider only makes sense paired with a different STT/TTS provider (or the local pair below). **Confidence: MEDIUM** (version verified via search; v1.0 is recent enough that some ecosystem tooling may still lag). |
| **Deepgram** (STT provider option) | `deepgram-sdk` on PyPI, actively released (3.x line as of early 2025, still the correct package name in 2026 — no rename to a bare `deepgram` package occurred) | Yes — async websocket streaming STT, plus websocket streaming TTS (`Aura`) | Callback/event-driven API (register handlers, call `start_listening()`). Supports raw linear16 and mulaw at configurable sample rates including 8000 Hz — usable directly against the camera's rate without resampling if the operator picks this provider. **Confidence: MEDIUM**. |
| **ElevenLabs** (TTS provider option) | `elevenlabs` on PyPI, 2.x line current in 2026 | Yes — `/v1/text-to-speech/{voice_id}/stream-input` websocket, and a bundled `Conversation` class for full-duplex STT→LLM→TTS pipelines | Best-in-class voice quality but no A-law/8kHz-native output mode documented as prominently as xAI's telephony codecs — expect to resample for the camera speaker unless explicitly requested in µ-law. **Confidence: LOW-MEDIUM** (exact current point version was ambiguous across sources — 2.7.1 vs 2.39.0 conflicting search hits; pin whatever `pip index versions elevenlabs` shows at implementation time rather than trusting this figure). |
| **whisper.cpp** (local STT, no cloud account) | via `pywhispercpp` bindings, or shell out to the compiled binary | Streaming via `--stream` flag; ~0.5-2s behind live speech depending on model size | CPU-only, no GPU required — matches the host's constraint directly. GGML-quantized models (`tiny`/`base`/`small` for real-time use) are the practical range on a CPU-only Xeon; `large-v3` is realistically batch-only (~3x realtime even with `faster-whisper`'s int8 path) and too slow for a <1s reply budget. **Confidence: HIGH** (CPU-only performance figures cross-checked across multiple 2026 comparison sources). |
| **faster-whisper** (local STT, alternative) | `faster-whisper` on PyPI (CTranslate2 backend) | Built-in VAD-driven streaming pipeline | ~2x faster than stock Whisper on CPU via int8/fp16 quantization and SIMD kernels; same model weights/WER as whisper.cpp, difference is purely runtime. Pick this over whisper.cpp if the team wants a Python-native API without shelling out to a compiled binary; pick whisper.cpp if minimizing the dependency footprint (single static binary, no CTranslate2/PyTorch chain) matters more. **Confidence: HIGH**. |
| **Piper** (local TTS, no cloud account) | **`piper1-gpl`** package (OHF-Voice org) — the original `rhasspy/piper` repo was archived Oct 2025; `piper1-gpl` (current: v1.6.0, July 2026) is the maintained successor | No true incremental streaming (renders full utterance quickly, not token-by-token), but synthesis is fast enough on CPU that latency is dominated by model load, not synthesis | Real-time synthesis on a Raspberry Pi 5 CPU alone — trivially fast on this project's 88-core Xeon. **Important licensing flag:** the successor's inference code is **GPL-3.0**, not the old MIT — this is a public repo, so pin this dependency deliberately and disclose it (voice checkpoints themselves carry separate, mostly-permissive per-voice licenses). **Confidence: HIGH** (license change independently confirmed). |

### A single unified provider interface

## 4. MCP in Python

| Technology | Version | Purpose | Why Recommended |
|---|---|---|---|
| **`mcp`** (official Model Context Protocol Python SDK) | v2.x line (protocol rev `2026-07-28`) | Spawn stdio MCP servers as child processes, list tools, call tools, manage lifecycle | This is the only SDK maintained by the protocol's own org (`modelcontextprotocol/python-sdk`), and v2 is a real breaking change from v1 — pin `mcp>=2.0` deliberately and do **not** let a loose `mcp` pin silently jump from v1 to v2 mid-project (v1's `FastMCP` class was renamed `MCPServer`; handler signatures changed from decorator-style to `async (ctx, params) → result`; all protocol field names went snake_case). **Confidence: MEDIUM-HIGH** (v2 existence, rename, and protocol-rev date verified directly against the SDK's own "what's new" page and PyPI; the *exact* stdio-launch code sample could not be pulled from the docs pages fetched this session — implement against the SDK's own `Client`/`StdioServerParameters` API reference at implementation time rather than trusting a remembered snippet). |

## 5. Backend

| Technology | Version | Purpose | Why Recommended |
|---|---|---|---|
| **FastAPI** | 0.141.x (latest, July 2026) | HTTP + WebSocket API, admin webapp backend | Already the project's committed framework (PROJECT.md constraints). Native `WebSocket` support (via Starlette) needs no extra package for live transcript/session streaming. **Confidence: HIGH**. |
| **SQLAlchemy** | 2.0.44 | ORM + async engine for Postgres | 2.0's async engine (`create_async_engine`), created once at app startup inside FastAPI's `lifespan` context and reused across requests — never per-request — is the load-bearing pattern; per-request engine/session creation is the most commonly cited FastAPI+Postgres mistake. **Confidence: MEDIUM** (version synthesized from a cluster of 2026 how-to sources rather than a single authoritative changelog; treat as "current as of mid-2026," pin exactly at implementation time). |
| **asyncpg** | 0.31.0 | Async Postgres driver (runtime path) | Fastest available async driver for the actual request-serving path — 2-3x faster than psycopg3's async mode in most benchmarks, and this project's latency budget (sub-second replies) makes every millisecond in the DB path worth protecting. **Confidence: MEDIUM**. |
| **psycopg2-binary** (or **psycopg** v3 in sync mode) | latest | Sync driver, migrations only | Alembic's migration runner is synchronous; wire it to a separate sync-mode connection string rather than trying to force async through Alembic. `psycopg` (v3) is an equally valid choice here if the team wants one driver family for both sync (migrations) and async (a LISTEN/NOTIFY use case, if one ever appears) — but keep `asyncpg` for the hot request path regardless, since that's where the performance actually matters. |
| **Alembic** | 1.17.1 | Schema migrations | Standard SQLAlchemy migration tool; no real alternative worth considering at this scale. **Confidence: MEDIUM**. |
| **Background scheduler for durable delayed workflow runs** | **Procrastinate** 3.9.0 | Scheduled/delayed macro & workflow execution ("a spoken sentence creates a scheduled workflow with delays and ordered steps") | This is the one place where the project's own Key Decision ("Postgres, not SQLite… pending workflow runs are durable shared state") points directly at a specific tool: Procrastinate is a Postgres-native task queue — no Redis/RabbitMQ broker, `defer()`/`defer_async()` for delayed jobs, retries, locks, and periodic tasks, all living in the same database already chosen for everything else. Actively maintained (releases into Aug 2026). **Do not use `arq`** — it is now in maintenance-only status and effectively unmaintained (its own ecosystem is spawning a spiritual successor, `StreAQ`, because of this); picking it today means picking a dead dependency on day one. **APScheduler** is a legitimate alternative (also supports a Postgres-backed job store) but is fundamentally a *scheduler*, not a *queue* — it's the better fit if the only need were "run this at time T," but this project also needs retries/locks/ordered-step execution for workflows, which is Procrastinate's actual design center. A **custom Postgres-backed poller** (the third option the project's own Key Decisions table floats) is reasonable only if the team specifically wants to avoid the new dependency — at home-assistant scale (a handful of pending workflows, not thousands/sec), a poller is maybe 150 lines of code, but it reinvents retries, locking, and crash-recovery that Procrastinate already solved and tested. **Recommendation: Procrastinate**, unless the team has a strong simplicity preference, in which case a hand-rolled poller is an acceptable, well-scoped alternative — not APScheduler, and never arq. **Confidence: MEDIUM-HIGH** (arq's dead status and Procrastinate's Postgres-native design are independently corroborated across sources; the final recommendation is this researcher's judgment call, flagged as such). |

## 6. Frontend

| Technology | Version | Purpose | Why Recommended |
|---|---|---|---|
| **React** | 19.2 | Admin webapp UI | Project constraint (PROJECT.md). Current stable line as of 2026. |
| **bun** | 1.4.x (1.4.2, Sept 2026) | Package manager + script runner | Project constraint. Note the 2026 ecosystem consensus is to use bun **as package manager/runner**, not necessarily as the dev server/bundler — pair it with Vite rather than bun's own bundler for the actual build, since Vite has the more mature plugin ecosystem for a React SPA. **Confidence: MEDIUM**. |
| **Vite** | 7.x | Dev server + production bundler | Dominant choice for SPAs with no SSR/RSC need (this webapp has none — it's a config/admin console) — faster to configure and run than a meta-framework, and this project has no server-rendering requirement to justify one. **Confidence: MEDIUM**. |
| **shadcn/ui** (Radix primitives + Tailwind, copy-in components) | current | Component library | "Own your components" model fits a project that needs a denylist editor, macro/workflow builder, and live transcript views — all bespoke enough that a copy-in, fully-customizable component beats a black-box installed library. Smallest bundle footprint of the mainstream options (~35-50KB vs Mantine's 80-120KB). **Confidence: MEDIUM**. Use **Mantine** instead only if the team wants a much larger pre-built component catalog (100+ components, built-in charts) and is willing to trade control for velocity — reasonable for a fast admin-panel build, less good long-term for a public repo other people will fork and reskin. |
| **TanStack Query** | current v5 | Server-state fetching/caching (session list, plugin config, macros) | Standard pairing with any REST/FastAPI backend; its `streamedQuery` helper is a documented, purpose-built path for consuming an SSE/async-iterable stream (e.g., a live transcript feed) directly into query state. **Confidence: MEDIUM**. |
| **Zustand** | current | Client-only UI state (wizard step, form drafts, denylist editor scratch state) | Deliberately kept separate from TanStack Query's server-state role — mixing the two responsibilities into one store is the most common state-management anti-pattern in 2026 React guidance. **Confidence: MEDIUM**. |
| **WebSocket** (native browser API, or `react-use-websocket`) | — | Live transcript streaming | WebSocket, not SSE, for the live transcript view: the page needs to be genuinely bidirectional-adjacent (operator can interrupt/cancel a pending workflow from the same view that's streaming transcript deltas), and the backend is already running a persistent per-session connection to the voice pod internally — reusing WebSocket end-to-end avoids maintaining two different push mechanisms for two different features. SSE remains a reasonable alternative for **past-session replay** (a one-directional log tail), where a WebSocket's extra complexity buys nothing. |

## 7. Auth

| Technology | Version | Purpose | Why Recommended |
|---|---|---|---|
| **PyJWT** | 2.13.0+ (pin ≥2.13.0 specifically) | Issue/verify JWTs | FastAPI's own tutorial migrated from `python-jose` to `PyJWT` in its docs without a formal migration note — follow that lead. **Pin ≥2.13.0**: five security advisories were fixed in that release, including CVE-2026-48526 (a public JWK could be accepted as an HMAC secret, forging HS256 tokens) — an old pin here is a real vulnerability, not a hygiene nit. **Do not use `python-jose`**: still widely referenced in older tutorials, but its maintenance has visibly lagged and FastAPI's own docs no longer lead with it. **Confidence: HIGH** (CVE and version verified directly). |
| **pwdlib[argon2]** | current | Password hashing | The direct successor to `passlib` (which is unmaintained) — FastAPI's own docs now recommend `pwdlib` with Argon2id specifically for its resistance to both GPU-accelerated brute force and side-channel timing attacks. `pip install 'pwdlib[argon2]'`. **Do not use bare `passlib`** for new code — it's the predecessor this tool was built to replace. **Confidence: HIGH**. |
| **Custom role dependency**, not a full auth framework | — | Sessions/cookies + JWT with roles (admin/operator/viewer), invite flow | At this project's scale (one household, three roles, an invite-by-admin flow — not multi-tenant OAuth/social login), a hand-rolled `require_role(Role.ADMIN)` FastAPI dependency plus a `users` table with a `role` column and an `invites` table (token + expiry + assigned role) is less code and less abstraction than adopting **`fastapi-users`**. `fastapi-users` is a legitimate, well-maintained option (also defaults to Argon2 via `pwdlib` under the hood as of its recent versions) — reach for it only if the roadmap later grows real multi-tenant needs (OAuth providers, email verification flows, password-reset email infra) that would otherwise mean re-deriving its feature set by hand. **Confidence: MEDIUM** (this is an architectural judgment about scope-fit, not a version claim). |
| **Session model** | HttpOnly, `SameSite=Lax` cookie holding a short-lived JWT (access) + a longer-lived opaque refresh token stored server-side (in Postgres, alongside the `users`/`invites` tables already needed) | Auth transport | Cookie-based (not `localStorage`-held JWT) because this is a same-origin admin webapp talking to its own backend — cookies get CSRF-safe defaults (`SameSite=Lax`) for free and aren't readable by any injected script, which matters more here than in an API-only, cross-origin SPA. Short access-token TTL (minutes) + refresh token rotation limits the blast radius of a leaked access token without forcing re-login every few minutes. |

## 8. Packaging (Helm + Docker Compose)

| Technology | Version | Purpose | Why Recommended |
|---|---|---|---|
| **Multi-stage Dockerfile** | — | One image containing the built React bundle served by (or alongside) the FastAPI backend | Stage 1: `oven/bun:1` — `bun install --frozen-lockfile && bun run build` (Vite output → static `dist/`). Stage 2: `python:3.12-slim` (not `-alpine` — avoids musl/glibc wheel-compatibility pain for `av`/`onnxruntime`/`ctranslate2`'s prebuilt wheels) — install Python deps with `pip install --no-cache-dir`, `COPY --from=stage1 /app/dist ./static`, run as a non-root user, `HEALTHCHECK` hitting a `/healthz` route. Keeps the final image to "backend + its runtime deps + one static folder," with no Node/bun toolchain in the runtime layer. **Confidence: MEDIUM** (pattern is well-established 2026 practice, not a single-source claim). |
| **Helm chart** | Chart API v2 | Kubernetes deployment (k3s target host) | One chart, values-driven: a `Deployment` for the pod (backend + static frontend from the same image), a `ConfigMap` mounting `config.yaml` (matching `config.example.yaml`'s env-expansion contract — `${NAME}` placeholders resolved from a `Secret` via `envFrom`), a `Service`, and a `PersistentVolumeClaim` for `/data` (TTS cache, session recordings) and `/models` (wake-word/local-STT model files). Given the host already runs Harbor + Argo CD + Argo Workflows for another project, structure the chart the same way that project does, for operational consistency — this wasn't independently re-verified this session but is a direct implication of PROJECT.md's own "Host" section. |
| **Docker Compose** | Compose v2 (`docker compose`, not legacy `docker-compose`) | Same app, no Kubernetes | Must ship as a genuinely equivalent path, since "most self-hosters do not run Kubernetes" is an explicit Key Decision. Mirror the Helm chart's mounts as bind-mounts/named volumes (`./config/config.yaml`, `./data`, `./models`) and use a `.env` file for the same `${NAME}` secrets the Helm chart pulls from a K8s `Secret` — keeping one config-loading code path (env-var expansion over the YAML text, per the config file's own header comment) serving both deployment targets is what makes "both ship" tractable without duplicated logic. |

## Alternatives Considered

| Recommended | Alternative | When to Use Alternative |
|---|---|---|
| PyAV for RTSP audio read | Raw ffmpeg subprocess (both directions) | If the team wants to minimize the number of distinct I/O patterns in the codebase at the cost of parsing ffmpeg's text output for stream state — reasonable for a smaller team, but loses PyAV's structured packet/frame API. |
| Procrastinate for delayed workflows | Custom Postgres-backed poller | If the team has a strong preference against adding a new dependency and workflow volume stays in the dozens-of-pending-runs range (true for a single household) — but budget real implementation time for retries/locking/crash-recovery, since that's the part Procrastinate gives for free. |
| shadcn/ui for components | Mantine | If velocity on the admin panel matters more than long-term bundle size/ownership, and the team is comfortable with a heavier, more opinionated dependency. |
| asyncpg for the runtime DB driver | psycopg3 (async mode) | If the team wants one driver family for both sync migrations and async runtime, or wants native `LISTEN/NOTIFY` support with a friendlier API — accept ~2-3x lower raw throughput for that convenience. |
| openWakeWord + Vosk (both ship) | Porcupine | Never, for this project specifically — its account/AccessKey requirement conflicts with the "no cloud account" requirement and the public-repo redistribution goal. Would only reconsider if that requirement changes. |

## What NOT to Use

| Avoid | Why | Use Instead |
|---|---|---|
| `arq` | Maintenance-only/effectively dead upstream; picking it today means adopting a dead dependency on day one | Procrastinate (Postgres-native) or Taskiq (if a Redis/broker-based queue is ever actually needed) |
| `python-jose` for new JWT code | FastAPI's own docs have moved off it; maintenance has visibly lagged relative to PyJWT | PyJWT ≥2.13.0 |
| bare `passlib` | Predecessor library that `pwdlib` was built to replace; not deprecated outright but no longer the recommended path | `pwdlib[argon2]` |
| original `rhasspy/piper` (MIT) | Archived Oct 2025 — a dead repo, and the README itself points at the successor | `OHF-Voice/piper1-gpl` (note: now GPL-3.0, disclose this in the public repo) |
| GStreamer Python bindings for this pipeline | Heavy native/GObject-introspection dependency chain, solves a harder problem (arbitrary pipeline graphs) than "read one RTSP audio stream, write one FIFO" requires; poor fit for slim multi-stage Docker builds | PyAV (read side) + supervised ffmpeg subprocess (write side) |
| `ffmpeg-python` (pip package) as a "better than subprocess" choice | It's still a subprocess call underneath, just with a filter-graph-building DSL on top; adds a dependency with real maintenance gaps for no execution-model benefit here | `asyncio.create_subprocess_exec` directly, or PyAV where structured decoding is actually needed |
| Picovoice Porcupine | Requires an account/AccessKey even for local inference; conflicts with "fully local, no cloud account" and complicates public-repo redistribution | openWakeWord or Vosk (already the chosen pair) |
| `whisper` (original OpenAI CPU implementation, unmodified) for real-time local STT | Meaningfully slower than either derivative on CPU-only hardware; not designed for the <1s reply budget this project targets | `faster-whisper` (CTranslate2, Python-native) or `whisper.cpp` (smaller dependency footprint, shell out to compiled binary) |

## Stack Patterns by Variant

- STT: `faster-whisper` (small/base model, int8) — NOT `large-v3`, which is batch-speed-only on this CPU-only host
- LLM: a local CPU-served model is *not* covered by this research pass (out of scope of the question set — flag as a gap below) — the config's `brain` block assumes an HTTP/OpenAI-compatible endpoint, so any local option must speak that same wire protocol (e.g., a local OpenAI-compatible server) to slot into the same `LLMProvider` adapter without new plumbing.
- TTS: Piper (`piper1-gpl`) — accept the GPL-3.0 licensing note above
- Wake word: openWakeWord or Vosk, per the per-deployment recording test in §1
- Use xAI STT/TTS/brain directly at 8kHz A-law throughout — this is the only provider

## Version Compatibility

| Package A | Compatible With | Notes |
|---|---|---|
| `mcp` ≥2.0 | protocol rev `2026-07-28` | Do not mix a v1-authored server (`FastMCP`, camelCase fields) with a v2 client without checking the SDK's migration guide first — the rename/casing changes are breaking, not additive. |
| `av` (PyAV) 18.1.0 | Python ≥3.11, bundles FFmpeg 8.x | If the base Docker image pins an older Python (e.g., 3.10 for some other dependency's sake), PyAV will not install — resolve the Python floor project-wide before pinning PyAV. |
| SQLAlchemy 2.0.44 + asyncpg 0.31.0 + Alembic 1.17.1 | PostgreSQL 18.3 (or any actively-supported PG major) | Alembic's own migration runner needs a *sync* driver (`psycopg2-binary` or `psycopg` v3 sync mode) even though the app runtime uses `asyncpg` — two driver packages installed side by side is expected, not a smell. |
| `pwdlib[argon2]` | PyJWT ≥2.13.0 | No direct interaction between the two, but both are the two halves of "FastAPI's currently-recommended auth stack" — pin both together so a security audit of one naturally covers the other. |

## Sources

- [openWakeWord GitHub](https://github.com/dscripka/openWakeWord) / [Releases](https://github.com/dscripka/openWakeWord/releases) — version, model format, CPU cost, maintenance timeline (websearch, cross-checked)
- [alphacep/vosk-api](https://github.com/alphacep/vosk-api) / [vosk PyPI](https://pypi.org/project/vosk/) — version, activity timeline (websearch, cross-checked)
- [Picovoice Porcupine docs/pricing](https://picovoice.ai/docs/porcupine/) — licensing/account requirement (websearch, single-pass)
- [ESPHome micro_wake_word](https://esphome.io/components/micro_wake_word/) — architecture confirming microcontroller-only scope (websearch)
- [PyAV PyPI](https://pypi.org/project/av/) — version, Python/FFmpeg requirements (webfetch, direct)
- [xAI Grok Speech-to-Text docs](https://docs.x.ai/developers/model-capabilities/audio/speech-to-text) and [Text-to-Speech docs](https://docs.x.ai/developers/model-capabilities/audio/text-to-speech) — A-law/8kHz support, websocket protocol, parameters (webfetch, direct)
- [xai-sdk-python](https://github.com/xai-org/xai-sdk-python) and OpenAI-client-compatibility guides — client library options (websearch, cross-checked)
- [MCP Python SDK "What's new in v2"](https://py.sdk.modelcontextprotocol.io/whats-new/) and [repo](https://github.com/modelcontextprotocol/python-sdk) — v2 changes, protocol rev (webfetch, direct)
- Aggregated 2026 comparison posts on FastAPI/SQLAlchemy/asyncpg/Alembic version pairings (websearch, cross-checked across ~4 independent sources — treat version pins as "current as of mid-2026," re-verify at implementation time)
- [Procrastinate PyPI](https://pypi.org/project/procrastinate/) / [GitHub](https://github.com/procrastinate-org/procrastinate) — version, Postgres-native design (websearch, cross-checked)
- Aggregated 2026 posts on `arq` maintenance status and `StreAQ` succession (websearch, cross-checked across 2 sources)
- [Piper successor `OHF-Voice/piper1-gpl`](https://github.com/OHF-Voice/piper1-gpl) and archived [`rhasspy/piper`](https://github.com/rhasspy/piper) — licensing change, current version (websearch, cross-checked)
- FastAPI's own JWT/password-hashing tutorial migration discussion (`fastapi/fastapi` GitHub Discussion #11345) and PyJWT 2.13.0 CVE advisories — auth stack recommendation (websearch, cross-checked)
- 2026 React/bun/Vite ecosystem roundups — frontend version/tooling recommendations (websearch, cross-checked across ~3 sources)
- shadcn/ui vs Mantine vs Chakra UI comparison posts — component library tradeoffs (websearch, cross-checked)

<!-- GSD:stack-end -->

<!-- GSD:conventions-start source:CONVENTIONS.md -->

## Conventions

Conventions not yet established. Will populate as patterns emerge during development.
<!-- GSD:conventions-end -->

<!-- GSD:architecture-start source:ARCHITECTURE.md -->

## Architecture

Architecture not yet mapped. Follow existing patterns found in the codebase.
<!-- GSD:architecture-end -->

<!-- GSD:skills-start source:skills/ -->

## Project Skills

No project skills found. Add skills to any of: `.claude/skills/`, `.agents/skills/`, `.cursor/skills/`, `.github/skills/`, or `.codex/skills/` with a `SKILL.md` index file.
<!-- GSD:skills-end -->

<!-- GSD:workflow-start source:GSD defaults -->

## GSD Workflow Enforcement

Before using Edit, Write, or other file-changing tools, start work through a GSD command so planning artifacts and execution context stay in sync.

Use these entry points:

- `/gsd-quick` for small fixes, doc updates, and ad-hoc tasks
- `/gsd-debug` for investigation and bug fixing
- `/gsd-execute-phase` for planned phase work

Do not make direct repo edits outside a GSD workflow unless the user explicitly asks to bypass it.
<!-- GSD:workflow-end -->

<!-- GSD:profile-start -->

## Developer Profile

> Profile not yet configured. Run `/gsd-profile-user` to generate your developer profile.
> This section is managed by `generate-claude-profile` -- do not edit manually.
<!-- GSD:profile-end -->
