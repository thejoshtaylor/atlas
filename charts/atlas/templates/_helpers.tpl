{{/*
Chart name, truncated and DNS-1123-safe.
*/}}
{{- define "atlas.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name -- combines the release name with the chart name,
matching the pattern every generated Helm chart uses so this one composes
predictably alongside other charts on the same cluster.
*/}}
{{- define "atlas.fullname" -}}
{{- if .Values.fullnameOverride -}}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- $name := default .Chart.Name .Values.nameOverride -}}
{{- if contains $name .Release.Name -}}
{{- .Release.Name | trunc 63 | trimSuffix "-" -}}
{{- else -}}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" -}}
{{- end -}}
{{- end -}}
{{- end -}}

{{/*
Common labels applied to every resource this chart renders.
*/}}
{{- define "atlas.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "atlas.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels -- the stable subset also used to match Pods to their owner.
*/}}
{{- define "atlas.selectorLabels" -}}
app.kubernetes.io/name: {{ include "atlas.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
The Secret name every template that reads env vars from it shares -- one
name, defined once, so the Deployment's envFrom and the Secret's own
metadata.name can never drift apart.

260922-cmo (D-2): when .Values.secretName is set, this returns that name
verbatim -- the operator's own hand-applied Secret, holding every
house-specific credential this public repo never carries. secret.yaml
itself renders nothing in that case (its own top-level guard), so the
only Secret the Deployment's envFrom and the Postgres StatefulSet's
POSTGRES_PASSWORD secretKeyRef ever see is the hand-applied one. Left
empty, the chart falls back to the templated Secret exactly as before.
*/}}
{{- define "atlas.secretName" -}}
{{- if .Values.secretName -}}
{{- .Values.secretName -}}
{{- else -}}
{{- printf "%s-secret" (include "atlas.fullname" .) -}}
{{- end -}}
{{- end -}}

{{/*
The bundled Postgres's own name and selector labels -- distinct from the
application's so the two StatefulSet/Deployment pods are never confused,
while still sharing the release's common labels.
*/}}
{{- define "atlas.postgres.fullname" -}}
{{- printf "%s-postgres" (include "atlas.fullname" .) -}}
{{- end -}}

{{- define "atlas.postgres.selectorLabels" -}}
app.kubernetes.io/name: {{ include "atlas.name" . }}-postgres
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "atlas.postgres.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "atlas.postgres.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
