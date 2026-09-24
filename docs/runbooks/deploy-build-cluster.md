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
these steps. The four manifests in `config/argo/` (applied in the last
step) run the rest automatically, forever after.

## 1. Harbor: create the project and a robot account

Create a Harbor project named `atlas`. Create a robot account scoped
to that project with push and pull permission. The build pushes to this
project. No other part of this pipeline needs a different registry.

## 2. Registry credentials: `regcred-atlas`, in three namespaces

Create a Kubernetes `dockerconfigjson` Secret named `regcred-atlas`
from that robot account's credentials. Create it in all three of these
namespaces:

- `argo` — the CI build (`buildctl`, inside the WorkflowTemplate) pushes
  with it.
- `argocd` — `argocd-image-updater` pulls image metadata with it, through
  the `pullsecret:argocd/regcred-atlas` annotation on the Application.
- `home` — the running pod pulls the image itself with it. The Harbor
  `atlas` project is private, so without this Secret (and the
  ServiceAccount patch below) the pod fails to start with
  `ImagePullBackOff: no basic auth credentials`.

```bash
kubectl create secret docker-registry regcred-atlas \
  --docker-server=harbor.tail41bc66.ts.net \
  --docker-username='<robot account name>' \
  --docker-password '<robot account token>' \
  --namespace argo

kubectl create secret docker-registry regcred-atlas \
  --docker-server=harbor.tail41bc66.ts.net \
  --docker-username='<robot account name>' \
  --docker-password '<robot account token>' \
  --namespace argocd

kubectl create secret docker-registry regcred-atlas \
  --docker-server=harbor.tail41bc66.ts.net \
  --docker-username='<robot account name>' \
  --docker-password '<robot account token>' \
  --namespace home
```

The Secret alone is not enough. The pod pulls under the `home` namespace's
default ServiceAccount, and nothing points that ServiceAccount at the new
Secret until you patch it:

```bash
kubectl -n home patch serviceaccount default \
  -p '{"imagePullSecrets":[{"name":"regcred-atlas"}]}'
```

Apply this patch by hand. Do not add it to the chart or to an Argo CD
manifest. Argo CD's `syncPolicy.automated.selfHeal` reverts any field a
synced manifest does not declare back to that manifest's own state on
every reconcile. A hand-applied patch to an object Argo CD does not manage
survives that reconcile; a patch folded into a synced manifest would not.
This mirrors the pattern chat-manager already uses: a `regcred` Secret
referenced from its own namespace's default ServiceAccount, applied by
hand for the same reason.

## 3. The runtime Secret, in the `home` namespace

Create the Secret the application runs against. Name it to match
`values-build.yaml`'s `secretName: atlas`. Create it in the `home`
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
| `COOKIE_SECURE` | Set `COOKIE_SECURE=true` once the public ingress at `atlas.jtlabs.co` is enabled (section 7 below) — public HTTPS in front of this release means the session cookie must be Secure. `false` is only correct for a deploy with no ingress and no TLS in front of it at all. |
| `DATABASE_URL` | See the pairing rule below. |
| `POSTGRES_PASSWORD` | See the pairing rule below. |
| `ATLAS_SECRET_KEY` | See the generation command below. |

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
postgresql+asyncpg://atlas:<the same POSTGRES_PASSWORD value>@<release-name>-postgres:5432/atlas
```

Generate `ATLAS_SECRET_KEY` with this command:

```bash
python3 -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

**Delete this Secret and the Postgres PersistentVolumeClaim together, or
not at all.** The templated-secret `helm.sh/resource-policy: keep`
annotation that `deploy-helm.md` describes does not apply to a hand-applied
Secret. You own its lifecycle entirely. If you delete this Secret while the
database volume survives, Kubernetes regenerates `POSTGRES_PASSWORD`. The
already-initialized database rejects that new password. A new
`ATLAS_SECRET_KEY` also makes every credential the database already holds
permanently unreadable. Treat the Secret and the volume as one unit
whenever you delete either.

## 4. GitHub webhook

Add a webhook on the `thejoshtaylor/atlas` repository. Point it at
the cluster's shared `git` EventSource — the same one the Sensor manifest
under `config/argo/` declares a dependency on. Send push events. The
Sensor's own filters (`body.repository.full_name`, `body.ref`) narrow this
down to a push on this repository's `main` branch. The webhook itself can
stay a generic push hook.

## 5. Image-updater registration

`argocd-image-updater` writes digests only for applications that an
`ImageUpdater` object names. The Application's annotations alone do
nothing. Without this object, a push to `main` builds and pushes an image,
but the running pod never changes.

`config/argo/atlas-image-updater.yaml` is this project's own
`ImageUpdater` object. Step 6 applies it with the other manifests. Do not
add `atlas` to another project's shared `ImageUpdater` object.

To make sure that the updater found the application:

```bash
kubectl -n argocd get imageupdater atlas
```

The `APPS` column must show `1`.

## 6. Apply the four manifests

```bash
kubectl apply -f config/argo/
```

This directory holds exactly four manifests. Each one names its own
namespace, so this one command places each in the right place: the Sensor
goes into `argo-events`, the WorkflowTemplate goes into `argo`, and the
Application and the ImageUpdater go into `argocd`.

## 7. The public ingress at atlas.jtlabs.co, and the cookie it requires

`values-build.yaml` enables the chart's ingress. Once it is applied, this
webapp is reachable at `https://atlas.jtlabs.co` over public HTTPS
(Cloudflare-terminated, in "Full" mode). A session cookie served over
public HTTPS must be Secure, so the hand-applied `home/atlas`
Secret must set `COOKIE_SECURE=true` — see the table in section 3.

Set it there, not in `values-build.yaml`. `config.cookieSecure` in that
file is inert once `secretName` is set (the chart then renders no
Secret of its own); the value that reaches the running pod is the one
in the hand-applied Secret.

Flipping a live Secret's key does not, by itself, restart the pod that
reads it. Restart the pod after you change `COOKIE_SECURE`, or the
running process keeps the old value until its next restart for any
other reason.

This runbook documents the required value and the restart it needs. It
does not flip the live Secret for you — that edit is an operator
hand-action, run once when you first enable the ingress.

## The seam: what CI does automatically, versus what you just did by hand

Once every step above is done, this sequence runs on every push to `main`,
with no further action from you.

1. The GitHub webhook fires. The Sensor's filters accept it (this
   repository, this branch) and submit a Workflow from the
   `atlas-ci` WorkflowTemplate.
2. The Workflow clones the repository over plain HTTPS. This step needs no
   credential, because the repository is public. The Workflow then builds
   the repo-root `Dockerfile`'s `runtime` stage with `buildctl`. It pushes
   `harbor.tail41bc66.ts.net/atlas/app` at both `:latest` and `:sha`
   to Harbor.
3. `argocd-image-updater` notices the new digest on `:latest`. It writes
   that digest into the `atlas` Application's `image.ref` Helm
   value.
4. Argo CD's `syncPolicy.automated` (with `prune: true` and `selfHeal:
   true`) applies that change to the cluster. The running pod is replaced
   with the freshly built image, in the `home` namespace. No one runs
   `helm upgrade` by hand.

Steps 1 through 7 above (the Harbor project, the registry credentials, the
runtime Secret, the GitHub webhook, the image-updater registration,
applying the four manifests, and the public ingress) are the one-time,
hand-run setup. This setup makes the automatic sequence above possible.
None of it repeats on a later push.

## Public-repo safety note

This repository is public. The real camera URL, the speaker backend value,
and every credential the assistant needs live only in the hand-applied
Secret from step 3 above. None of them live in git. The committed
configuration (`config/config.example.yaml`, shipped unchanged inside the
chart) reads `${CAMERA_RTSP_URL}` and `${SPEAKER_BACKEND}` placeholders
instead of a real value, the same way it already reads `${XAI_API_KEY}`
and the rest.
