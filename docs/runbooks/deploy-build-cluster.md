# Deploying the build-cluster pipeline

This runbook is for the jt-spire build cluster. On this cluster, a push to
`main` builds the image and rolls it out. You do not run `docker build` or
`helm upgrade` by hand. This runbook lists only the prerequisites you apply
by hand on that cluster. CI and Argo CD do everything else once these
prerequisites exist.

If you deploy somewhere else, see [`deploy-helm.md`](deploy-helm.md) or
[`deploy-compose.md`](deploy-compose.md) instead. This runbook assumes the
same Helm chart those two documents cover, plus the automation layer on top
of it.

You run every step below once, by hand. CI and Argo CD do not run any of
these steps. The three manifests in `config/argo/` (applied in the last
step) run the rest automatically, forever after.

## 1. Harbor: create the project and a robot account

Create a Harbor project named `spire-voice`. Create a robot account scoped
to that project with push and pull permission. The build pushes to this
project. No other part of this pipeline needs a different registry.

## 2. Registry credentials: `regcred-spire-voice`, twice

Create a Kubernetes `dockerconfigjson` Secret named `regcred-spire-voice`
from that robot account's credentials. Create it in both of these
namespaces:

- `argo` — the CI build (`buildctl`, inside the WorkflowTemplate) pushes
  with it.
- `argocd` — `argocd-image-updater` pulls image metadata with it, through
  the `pullsecret:argocd/regcred-spire-voice` annotation on the Application.

```bash
kubectl create secret docker-registry regcred-spire-voice \
  --docker-server=harbor.tail41bc66.ts.net \
  --docker-username='<robot account name>' \
  --docker-password '<robot account token>' \
  --namespace argo

kubectl create secret docker-registry regcred-spire-voice \
  --docker-server=harbor.tail41bc66.ts.net \
  --docker-username='<robot account name>' \
  --docker-password '<robot account token>' \
  --namespace argocd
```

## 3. The runtime Secret, in the `home` namespace

Create the Secret the application runs against. Name it to match
`values-build.yaml`'s `secretName: spire-voice`. Create it in the `home`
namespace. This is the hand-applied Secret D-2 describes. This repository
carries no default for any key below. None of these values belong in git.
This repository is public.

The shipped configuration needs every key below.

| Key | Notes |
|---|---|
| `XAI_API_KEY` | xAI credential for speech-to-text, the language model, and text-to-speech. |
| `TAPO_CLOUD_PASSWORD` | The Tapo cloud-account password. The application reads this only when `speaker.backend` is `tapo_talk`. |
| `SPEAKER_ENSURE_URL` | The go2rtc backchannel-producer URL. It holds the camera's own credentials. |
| `CAMERA_RTSP_URL` | The whole authenticated camera RTSP URL. This value never appears in this repository. This Secret is the one place it lives. |
| `SPEAKER_BACKEND` | `go2rtc` or `tapo_talk`. |
| `BIND_HOST` | `0.0.0.0`. This matches `values-build.yaml`'s `config.bindHost`. |
| `COOKIE_SECURE` | `false`. This matches `values-build.yaml`'s `config.cookieSecure`, because no ingress or TLS sits in front of this release yet. |
| `DATABASE_URL` | See the pairing rule below. |
| `POSTGRES_PASSWORD` | See the pairing rule below. |
| `SPIRE_SECRET_KEY` | See the generation command below. |

Two keys are optional legacy holdovers. Keep them only if your own config
still builds the camera URL from them the old way:

- `TAPO_USER`
- `TAPO_PASSWORD`

**`HA_URL` and `HA_TOKEN` do NOT belong in this Secret.** Home Assistant is
a plugin now, not a `config.yaml` block. Its credentials live in the
plugins table (seeded by `alembic/versions/0008_plugin_tables.py`). You
enter them through the admin webapp after the assistant is already
running. The application does not read them from the environment at boot.

**`DATABASE_URL` and `POSTGRES_PASSWORD` are one coupled pair, not two
independent values.** The bundled Postgres StatefulSet reads
`POSTGRES_PASSWORD` from this same Secret at `initdb` time. This happens
once, the first time its volume is empty. `DATABASE_URL` must embed that
exact same password. It must also point at this release's own headless
Postgres Service:

```text
postgresql+asyncpg://spire:<the same POSTGRES_PASSWORD value>@<release-name>-postgres:5432/spire
```

Generate `SPIRE_SECRET_KEY` with this command:

```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Delete this Secret and the Postgres PersistentVolumeClaim together, or
not at all.** The templated-secret `helm.sh/resource-policy: keep`
annotation that `deploy-helm.md` describes does not apply to a hand-applied
Secret. You own its lifecycle entirely. If you delete this Secret while the
database volume survives, Kubernetes regenerates `POSTGRES_PASSWORD`. The
already-initialized database rejects that new password. A new
`SPIRE_SECRET_KEY` also makes every credential the database already holds
permanently unreadable. Treat the Secret and the volume as one unit
whenever you delete either.

## 4. GitHub webhook

Add a webhook on the `thejoshtaylor/spire-voice` repository. Point it at
the cluster's shared `git` EventSource — the same one the Sensor manifest
under `config/argo/` declares a dependency on. Send push events. The
Sensor's own filters (`body.repository.full_name`, `body.ref`) narrow this
down to a push on this repository's `main` branch. The webhook itself can
stay a generic push hook.

## 5. Image-updater registration: read, modify, write

`argocd-image-updater` writes digests only for applications named in the
cluster's one shared `ImageUpdater` object, `perpetuity`, in the `argocd`
namespace. Add an entry for `spire-voice` to it.

Read the live object first:

```bash
kubectl -n argocd get imageupdater perpetuity -o yaml > /tmp/perpetuity.yaml
```

Edit `/tmp/perpetuity.yaml`. Add this entry alongside whatever
`applicationRefs` entries are already there:

```yaml
- namePattern: spire-voice
  useAnnotations: true
```

Apply the edited file:

```bash
kubectl apply -f /tmp/perpetuity.yaml
```

**Do not apply a freshly authored file for this object without reading the
live object first.** This object is shared across every project on this
cluster that uses image-updater. A file that names only `spire-voice`
replaces every other project's entry. Always read the live object first,
fold your entry into it, and apply the result.

## 6. Apply the three manifests

```bash
kubectl apply -f config/argo/
```

This directory holds exactly three manifests. Each one names its own
namespace, so this one command places each in the right place: the Sensor
goes into `argo-events`, the WorkflowTemplate goes into `argo`, and the
Application goes into `argocd`.

## The seam: what CI does automatically, versus what you just did by hand

Once every step above is done, this sequence runs on every push to `main`,
with no further action from you.

1. The GitHub webhook fires. The Sensor's filters accept it (this
   repository, this branch) and submit a Workflow from the
   `spire-voice-ci` WorkflowTemplate.
2. The Workflow clones the repository over plain HTTPS. This step needs no
   credential, because the repository is public. The Workflow then builds
   the repo-root `Dockerfile`'s `runtime` stage with `buildctl`. It pushes
   `harbor.tail41bc66.ts.net/spire-voice/app` at both `:latest` and `:sha`
   to Harbor.
3. `argocd-image-updater` notices the new digest on `:latest`. It writes
   that digest into the `spire-voice` Application's `image.ref` Helm
   value.
4. Argo CD's `syncPolicy.automated` (with `prune: true` and `selfHeal:
   true`) applies that change to the cluster. The running pod is replaced
   with the freshly built image, in the `home` namespace. No one runs
   `helm upgrade` by hand.

Steps 1 through 6 above (the Harbor project, the registry credentials, the
runtime Secret, the GitHub webhook, the image-updater registration, and
applying the three manifests) are the one-time, hand-run setup. This setup
makes the automatic sequence above possible. None of it repeats on a later
push.

## Public-repo safety note

This repository is public. The real camera URL, the speaker backend value,
and every credential the assistant needs live only in the hand-applied
Secret from step 3 above. None of them live in git. The committed
configuration (`config/config.example.yaml`, shipped unchanged inside the
chart) reads `${CAMERA_RTSP_URL}` and `${SPEAKER_BACKEND}` placeholders
instead of a real value, the same way it already reads `${XAI_API_KEY}`
and the rest.
