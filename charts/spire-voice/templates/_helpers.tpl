{{/*
Chart name, truncated and DNS-1123-safe.
*/}}
{{- define "spire-voice.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" -}}
{{- end -}}

{{/*
Fully qualified app name -- combines the release name with the chart name,
matching the pattern every generated Helm chart uses so this one composes
predictably alongside other charts on the same cluster.
*/}}
{{- define "spire-voice.fullname" -}}
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
{{- define "spire-voice.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "spire-voice.selectorLabels" . }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}

{{/*
Selector labels -- the stable subset also used to match Pods to their owner.
*/}}
{{- define "spire-voice.selectorLabels" -}}
app.kubernetes.io/name: {{ include "spire-voice.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{/*
The Secret name every template that reads env vars from it shares -- one
name, defined once, so the Deployment's envFrom and the Secret's own
metadata.name can never drift apart.
*/}}
{{- define "spire-voice.secretName" -}}
{{- printf "%s-secret" (include "spire-voice.fullname" .) -}}
{{- end -}}

{{/*
The bundled Postgres's own name and selector labels -- distinct from the
application's so the two StatefulSet/Deployment pods are never confused,
while still sharing the release's common labels.
*/}}
{{- define "spire-voice.postgres.fullname" -}}
{{- printf "%s-postgres" (include "spire-voice.fullname" .) -}}
{{- end -}}

{{- define "spire-voice.postgres.selectorLabels" -}}
app.kubernetes.io/name: {{ include "spire-voice.name" . }}-postgres
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end -}}

{{- define "spire-voice.postgres.labels" -}}
helm.sh/chart: {{ printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" }}
{{ include "spire-voice.postgres.selectorLabels" . }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end -}}
