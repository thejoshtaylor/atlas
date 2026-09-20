# Deploying with Helm

This is the full walk from a chart in this repository to a running assistant on
your own Kubernetes cluster. Every value below is invented for illustration --
substitute your own release name, namespace, host name, and image reference.

## What you need first

- A Kubernetes cluster you can reach with `kubectl`, and `helm` (this chart uses
  Chart API v2, no chart dependencies).
- Your own image, pushed somewhere your cluster can pull from. This chart names
  no default registry on purpose (`charts/spire-voice/values.yaml`'s own
  comment): you build the image from this repository's `Dockerfile` and push it,
  rather than pulling a build this project would have to maintain and version
  independently of the chart.

  ```bash
  docker build -t <your-registry>/spire-voice:<tag> .
  docker push <your-registry>/spire-voice:<tag>
  ```

- An Ingress controller, if you want a URL rather than a `kubectl port-forward`.
  This chart's Ingress is off by default (`ingress.enabled: false`).

## Install it

```bash
helm install spire-voice charts/spire-voice \
  --set image.repository=<your-registry>/spire-voice \
  --set image.tag=<tag>
```

This renders a Secret (the generated session/credential-encryption key and the
bundled database's password), a ConfigMap carrying the application's own
configuration file unchanged, the application Deployment and its two
PersistentVolumeClaims (`data`, `models`), and the bundled Postgres
StatefulSet and headless Service. No file to edit -- every operator-specific
value is a `--set` flag or a values file, never a template you hand-modify.

## The wake-word model -- one job, before the application pod can become healthy

The application's default configuration listens for the phrase "hey spire"
using Vosk, a wake-word engine that needs a model file on disk. Nothing in
this project downloads that model automatically. The chart gives the
application pod a `models` volume, but nothing provisions the model file
into it by itself -- run this once, after `helm install`, before the
application pod's first successful start:

```bash
kubectl run spire-voice-fetch-wake-model --rm -i --restart=Never \
  --image=<your-registry>/spire-voice:<tag> \
  --overrides='
{
  "spec": {
    "containers": [{
      "name": "fetch-wake-model",
      "image": "<your-registry>/spire-voice:<tag>",
      "command": ["python", "scripts/fetch_wake_model.py", "--config", "config/config.example.yaml"],
      "volumeMounts": [{"name": "models", "mountPath": "/models"}]
    }],
    "volumes": [{
      "name": "models",
      "persistentVolumeClaim": {"claimName": "spire-voice-models"}
    }],
    "restartPolicy": "Never"
  }
}'
```

`spire-voice-models` above is `{release name}-models` -- substitute your own
release name if you installed under a different one. Until this job runs, the
application pod refuses to start with a message naming the missing model
directory, by design, rather than pretending to listen for a wake word it
cannot detect; once the model is provisioned, Kubernetes' own restart policy
brings the application pod up on its next attempt with no further action from
you.

## What this deployment actually does, stated plainly

**This chart assumes an Ingress (or some other front door) terminates TLS
before traffic reaches it.** `values.yaml`'s `config.cookieSecure` defaults to
`"true"` on that assumption -- the standard shape for a Kubernetes-fronted
deployment. **If you are not terminating TLS in front of this release**, set
`--set config.cookieSecure=false` explicitly; leaving the default in place
without TLS means the session cookie claims a security property the connection
does not actually have. Setting it to `false` brings back this project's own
insecure-cookie/reachable-bind startup warning as the backstop for exactly this
case.

**The bundled database needs no external Postgres, and no default password ships
anywhere in this chart.** `templates/secret.yaml` generates the database
password (and the session/credential-encryption key) once, at install time, and
`templates/secret.yaml`'s own `lookup` guard reads the release's existing Secret
before ever generating again -- an upgrade or a `helm template` re-render never
regenerates either value. Nothing about the database password needs your
attention; you never see it unless you go looking with `kubectl get secret`.

**The generated key is minted exactly once, and survives an upgrade byte-for-
byte -- proven, not assumed, this session.** `scripts/verify-helm-deploy.sh`
installs the chart into a throwaway namespace, upgrades the release, and checks
the key on both sides of that upgrade -- see that script's own comments for what
it proves and how to run it yourself against your cluster. This is the same
value `docs/runbooks/deploy-compose.md` describes for the Compose path: back up
whatever this Secret's data ultimately rests on according to your own cluster's
backup practice, since losing it makes every stored provider credential
permanently unreadable and invalidates every signed-in session at once.

## Providers, credentials, and restarts

Provider credentials (a speech-to-text key, a language-model key, a
text-to-speech key, a Home Assistant token) are entered from the webapp's
Providers and Settings screens once you have signed in -- never from a values
file or a template you edit. **Changing which provider a slot uses needs a
restart of the application pod**, and the screen says so: your choice is saved
immediately, but the slot keeps running whichever provider it started with
until the pod restarts (`kubectl rollout restart deployment/spire-voice`,
substituting your own release name).

## The local provider set

A fully local provider set needs no cloud account at all -- see
[`docs/runbooks/local-providers.md`](local-providers.md) for what it is, how to
provision its own model files (the same `scripts/fetch_models.py` step, run the
same way the wake-model job above runs, against the `models` volume), and the
measured latency you should expect on a machine with no GPU: it is slower than
the cloud set, published as a real number in that runbook, not a target this
project claims to meet. If you select Piper for text-to-speech, read that
runbook's licence note first -- it is an optional package under a different
licence than the rest of this project.

## Upgrading

```bash
helm upgrade spire-voice charts/spire-voice \
  --set image.repository=<your-registry>/spire-voice \
  --set image.tag=<new-tag>
```

Migrations run automatically at application startup (DEP-04); the bundled
database's own data survives an upgrade because it lives on a
PersistentVolumeClaim, never `emptyDir`.

## Uninstalling, and what a clean reinstall really means

Two things this release creates outlive `helm uninstall`, on purpose, and they
must be deleted together or not at all:

- the Secret (`<release>-spire-voice-secret`), which carries the generated
  `SPIRE_SECRET_KEY` and the bundled database's password;
- the database's PersistentVolumeClaim (`postgres-data-<release>-spire-voice-postgres-0`).

Kubernetes never deletes a StatefulSet's PVC, and this chart marks the Secret
`helm.sh/resource-policy: keep` so that Helm does not delete it either. Those
two facts exist together for one reason. The database image reads
`POSTGRES_PASSWORD` only when its data directory is empty, so a database that
survives on its PVC keeps the password it was initialized with; and
`SPIRE_SECRET_KEY` is what every provider credential in that database is
encrypted with. Delete the Secret while keeping the volume and you get a
database that rejects the new password and a set of credentials nobody can ever
decrypt again -- silently, from two ordinary commands run in order.

To uninstall and reinstall the same deployment, keeping your data:

```bash
helm uninstall spire-voice
helm install spire-voice charts/spire-voice --set image.repository=... --set image.tag=...
```

Use the same release name and the same namespace. The reinstall adopts the kept
Secret and reuses the surviving volume. A different release name gets a
different Secret and a different volume -- a genuinely new deployment, not this
one.

To start genuinely fresh, delete both, and accept that every stored credential
and every signed-in session goes with them:

```bash
helm uninstall spire-voice
kubectl delete secret spire-voice-secret
kubectl delete pvc postgres-data-spire-voice-postgres-0
```

(Substitute your own release name in all three.) `scripts/verify-helm-deploy.sh`
runs the uninstall-then-reinstall path for real on every invocation and fails if
the key does not survive it, or if the surviving database will not accept the
reinstalled password.

## Verifying this document is true

`scripts/verify-helm-deploy.sh` is the real proof behind the claims above: it
installs this chart into a throwaway namespace it creates, waits for the
application pod's init containers to run to completion, upgrades the release,
and checks the generated key and a marker row written to the bundled database
-- all of it survives, or the script says exactly which claim failed. Run it
yourself against a cluster you can reach:

```bash
./scripts/verify-helm-deploy.sh
```

Add `--image <your-registry>/spire-voice:<tag> --wait-for-app` to also prove
the application pod itself becomes healthy; without a pullable image the
script honestly reports that claim as "not attempted" rather than assuming it.
That flag also puts `--wait` on the `helm install` and `helm upgrade` calls,
so a release that never becomes ready fails the helm command itself.

The init-container claim runs on every invocation, with or without an image,
because it finishes before the application image is ever needed. It is there
because an earlier version of this chart could not install on any cluster --
the `wait-for-postgres` init container was refused by the kubelet -- while
this script still printed "every attempted claim was proved".
