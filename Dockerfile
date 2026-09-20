# syntax=docker/dockerfile:1
#
# One multi-stage image (D-13). Stage "frontend" builds the React admin
# webapp with bun. Stage "runtime" serves it beside the FastAPI backend on
# python:3.12-slim -- not the -alpine variant, because musl breaks the
# prebuilt wheels for this project's native-extension dependencies (av,
# onnxruntime, ctranslate2), and building them from source in an image is
# a long, fragile detour for no benefit.
#
# D-13 names python:3.12-slim as the locked Python floor, even though this
# project's own development interpreter is 3.14. Stage "test" below is the
# proof that this project's full dependency set actually installs and runs
# on 3.12 -- not an assumption carried over from a wheel-tag listing.

### Stage: frontend ###########################################################
# web/vite.config.ts's own comment names build.outDir="dist" as a path a
# rename would break -- the runtime stage copies from that exact directory.
FROM oven/bun:1 AS frontend

WORKDIR /web
COPY web/package.json web/bun.lock ./
RUN bun install --frozen-lockfile
COPY web/ ./
RUN bun run build

### Stage: python-base #########################################################
# The dependency set both the test stage and the runtime stage need. Kept
# as its own stage so neither one repeats the pip install.
FROM python:3.12-slim AS python-base

WORKDIR /app

# libatomic1: vosk's own compiled libvosk.so links against it, and
# python:3.12-slim does not ship it (confirmed by a real `import vosk`
# failing with "libatomic.so.1: cannot open shared object file"
# otherwise). libgomp1: onnxruntime/ctranslate2's own OpenMP-parallel
# native code needs it for the same reason. Neither is a Python
# dependency pip can install -- both are system shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libatomic1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# alembic.ini's own script_location is %(here)s/alembic -- relative to
# alembic.ini's own directory -- and db/engine.py::run_migrations opens
# "alembic.ini" as a relative path. Both require the process's working
# directory to hold this exact layout, the same one the repository itself
# uses. The application package stays importable from this source
# directory too (never relocated into site-packages): FRONTEND_DIR and
# MCP_ROOT (app.py) are both computed from the application module's own
# file location, the same PYTHONPATH-style contract scripts/dev-run.sh
# already uses for the same two paths.
#
# mcp/spire_mcp is a second top-level source directory, never installed
# by pip (pytest.ini's own pythonpath=["src", "mcp"] and dev-run.sh's
# PYTHONPATH=src:mcp already treat it this way) -- db/repository.py
# imports it directly (`from spire_mcp.safety import Policy`), so the
# main process needs it on PYTHONPATH too, not only the MCP child
# processes app.py spawns with their own explicit env.
ENV PYTHONPATH=/app/mcp

COPY pyproject.toml ./
COPY src/ ./src/
COPY mcp/ ./mcp/
COPY alembic.ini ./
COPY alembic/ ./alembic/
COPY config/ ./config/
# scripts/fetch_models.py and friends: operator-run steps, never called by
# the application itself, but a container is the natural place to run
# `docker compose exec app python scripts/fetch_models.py` from. The test
# stage below also imports these by path (tests/test_fetch_models.py and
# its siblings), so they need to exist here, not only in the test stage.
COPY scripts/ ./scripts/
# docs/runbooks/*.md: tests/test_score_wake_engines.py reads the wake
# corpus runbook by path and checks it names real, existing scripts.
COPY docs/ ./docs/

# A non-root user, owning the writable mounts config.example.yaml already
# names (/data for sessions/tts-cache/calibration, /models for the local
# provider set) and the application directory. Shared by the test stage
# below (so a filesystem-permission test behaves the way it does on a
# real deployment, not the way it behaves under a root build user) and
# the runtime stage.
RUN groupadd --system spire \
    && useradd --system --gid spire --home-dir /app --no-create-home spire \
    && mkdir -p /data /models \
    && chown -R spire:spire /app /data /models

RUN pip install --no-cache-dir --upgrade pip

# openwakeword==0.6.0's own PyPI metadata makes tflite-runtime an
# unconditional dependency on Linux, and tflite-runtime has never
# published a wheel newer than cp311, for any platform -- confirmed
# directly against PyPI's own file listing this session. Left as an
# ordinary dependency, `pip install -e .` cannot complete on this image's
# Python at all, on any Linux host, independent of the cp312-vs-cp314
# question above.
#
# This project's own wake engine (wake/openwakeword_engine.py) always
# passes inference_framework="onnx" and never loads a .tflite model.
# openwakeword's real code (checked directly against the installed
# 0.6.0 wheel) only imports tflite_runtime inside a try/except, at the
# point a .tflite model is actually loaded -- never at module import
# time. Excluding tflite-runtime removes no capability this codebase
# uses. openwakeword installs with --no-deps below; its own real,
# working dependencies (every one except tflite-runtime) install by
# name on the next line.
RUN python3 -c "import re, tomllib; deps=[d for d in tomllib.load(open('pyproject.toml','rb'))['project']['dependencies'] if not re.match(r'^openwakeword\b', d)]; open('/tmp/requirements-no-oww.txt', 'w').write('\n'.join(deps) + '\n')"
RUN pip install --no-cache-dir -r /tmp/requirements-no-oww.txt \
    && pip install --no-cache-dir --no-deps openwakeword==0.6.0 \
    && pip install --no-cache-dir tqdm scipy scikit-learn requests \
    && pip install --no-cache-dir --no-deps -e . \
    && rm -f /tmp/requirements-no-oww.txt

### Stage: test ################################################################
# The actual proof this Dockerfile's own docstring above promises: install
# the dev extra and run the real backend suite on the Python this image
# ships, not the Python this project develops on. A database-backed test
# skips itself when SPIRE_TEST_DATABASE_URL is unset (see
# tests/test_db_migrations.py's own skip_without_postgres) -- this build
# has no Postgres to reach, so those tests skip here and run for real in
# CI/dev against a live database instead. Everything else -- every import,
# every unit test across the whole dependency set this image installs --
# runs for real, right here, at build time.
FROM python-base AS test

# pytest/pytest-asyncio directly, not `-e ".[dev]"` -- the extras syntax
# re-resolves spire-voice's whole dependency graph, including
# openwakeword, undoing the --no-deps workaround python-base already
# applied above.
RUN pip install --no-cache-dir pytest pytest-asyncio
COPY --chown=spire:spire tests/ ./tests/
# Non-root, matching the runtime stage: a test asserting that an
# unwritable directory is actually unwritable is meaningless as root,
# since root ignores ordinary permission bits.
USER spire
RUN python -m pytest -q

### Stage: runtime #############################################################
FROM python-base AS runtime

COPY --from=frontend --chown=spire:spire /web/dist ./web/dist

USER spire

EXPOSE 8080

# D-13 names /healthz. The route this application actually serves is
# /health (app.py) -- its own docstring says it carries no role
# requirement and is exempt from the setup gate by name, exactly what a
# container health check needs. This points at the route that exists.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=3).read()" || exit 1

CMD ["python", "-m", "spire_voice.app"]
