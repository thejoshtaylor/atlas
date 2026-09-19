import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Link, useNavigate, useParams } from "react-router-dom"
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
} from "@/components/ui/alert-dialog"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { RadioGroup, RadioGroupItem } from "@/components/ui/radio-group"
import { EmptyState } from "@/components/state/EmptyState"
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import { SubmitButton } from "@/components/state/SubmitButton"
import { ApiError } from "@/lib/api"
import {
  PLUGIN_CATALOG_QUERY_KEY,
  deletePluginMutationOptions,
  fetchPlugin,
  fetchPluginCatalog,
  installPluginMutationOptions,
  pluginQueryKey,
  savePluginConfigMutationOptions,
  setPluginEnabledMutationOptions,
  type Plugin,
} from "@/lib/plugins"
import {
  BLANK_CONFIG_KEY_NAME_REASON,
  draftConfigValuesToInput,
  saveBlockedByBlankConfigKeyName,
  usePluginDraftStore,
} from "@/stores/pluginDraftStore"
import {
  BLANK_DISPLAY_NAME_REASON,
  deriveCatalogPaneState,
  derivePluginEditorState,
  formatConfigKeyCount,
  remoteConfigCaption,
  saveBlockedByBlankDisplayName,
  saveBlockedByMissingCustomSource,
  saveBlockedByNoCatalogSelection,
  toolCollisionText,
} from "./derivePluginEditorState"
import { formatToolCount, runtimeStatusDisplay } from "./derivePluginsScreenState"

// PLUG-03, PLUG-04, PLUG-08, D-14, D-16 -- the same editor component
// serves `/plugins/new` and `/plugins/:id`, branching on whether an id is
// present, the exact shape `MacroEditorRoute.tsx` already ships (06-UI-
// SPEC.md's own opening line). The install flow is one scrolling screen,
// never a multi-step wizard; the runtime status block leads on an
// existing plugin, because an admin who opens an already-installed
// plugin's page is most often asking why it is not working
// (06-UI-SPEC.md's Focal Point row).

function installFailureMessage(error: unknown): string {
  const reason = error instanceof ApiError ? error.message : "Try again"
  return `Couldn't install this plugin. ${reason}.`
}

export function PluginEditorRoute() {
  const params = useParams<{ id: string }>()
  const navigate = useNavigate()
  const pluginId = params.id ? Number(params.id) : null

  const pluginQuery = useQuery({
    queryKey: pluginId !== null ? pluginQueryKey(pluginId) : ["plugins", "new"],
    queryFn: () => fetchPlugin(pluginId as number),
    enabled: pluginId !== null,
  })
  const screen = derivePluginEditorState(pluginId === null ? null : pluginQuery)
  const recordLoading = screen.kind !== "ready"

  const catalogQuery = useQuery({
    queryKey: PLUGIN_CATALOG_QUERY_KEY,
    queryFn: fetchPluginCatalog,
    enabled: pluginId === null,
  })
  const catalogScreen = deriveCatalogPaneState(catalogQuery)

  const installMode = usePluginDraftStore((state) => state.installMode)
  const selectedCatalogEntry = usePluginDraftStore((state) => state.selectedCatalogEntry)
  const displayName = usePluginDraftStore((state) => state.displayName)
  const customTransport = usePluginDraftStore((state) => state.customTransport)
  const command = usePluginDraftStore((state) => state.command)
  const url = usePluginDraftStore((state) => state.url)
  const timeoutMs = usePluginDraftStore((state) => state.timeoutMs)
  const configValues = usePluginDraftStore((state) => state.configValues)
  const newKeyName = usePluginDraftStore((state) => state.newKeyName)
  const newKeyValue = usePluginDraftStore((state) => state.newKeyValue)
  const newKeySecret = usePluginDraftStore((state) => state.newKeySecret)

  const setInstallMode = usePluginDraftStore((state) => state.setInstallMode)
  const selectCatalogEntry = usePluginDraftStore((state) => state.selectCatalogEntry)
  const clearCatalogSelection = usePluginDraftStore((state) => state.clearCatalogSelection)
  const setDisplayName = usePluginDraftStore((state) => state.setDisplayName)
  const setCustomTransport = usePluginDraftStore((state) => state.setCustomTransport)
  const setCommand = usePluginDraftStore((state) => state.setCommand)
  const setUrl = usePluginDraftStore((state) => state.setUrl)
  const setTimeoutMs = usePluginDraftStore((state) => state.setTimeoutMs)
  const setConfigValue = usePluginDraftStore((state) => state.setConfigValue)
  const startEditingSecret = usePluginDraftStore((state) => state.startEditingSecret)
  const revertSecretEdit = usePluginDraftStore((state) => state.revertSecretEdit)
  const addConfigKey = usePluginDraftStore((state) => state.addConfigKey)
  const setNewKeyName = usePluginDraftStore((state) => state.setNewKeyName)
  const setNewKeyValue = usePluginDraftStore((state) => state.setNewKeyValue)
  const setNewKeySecret = usePluginDraftStore((state) => state.setNewKeySecret)
  const loadPlugin = usePluginDraftStore((state) => state.loadPlugin)
  const loadBlank = usePluginDraftStore((state) => state.loadBlank)

  // The draft only ever loads from the server (or resets blank) once per
  // route identity -- keyed on the plugin id (or "new") so a background
  // refetch of the same plugin never clobbers an admin's unsaved typing;
  // this component's own save handlers call `loadPlugin` directly to
  // resync after a save it made itself (`MacroEditorRoute.tsx`'s own
  // `updated_at`-keyed effect solves the identical problem -- `Plugin`
  // carries no such field, so a direct post-save call stands in for it).
  const loadedKeyRef = React.useRef<string | null>(null)
  React.useEffect(() => {
    if (screen.kind !== "ready") return
    const key = screen.plugin ? String(screen.plugin.id) : "new"
    if (loadedKeyRef.current === key) return
    loadedKeyRef.current = key
    if (screen.plugin) loadPlugin(screen.plugin)
    else loadBlank()
  }, [screen, loadPlugin, loadBlank])

  const [installError, setInstallError] = React.useState<string | null>(null)
  const [saveConfigError, setSaveConfigError] = React.useState<string | null>(null)
  const [deleteConfirmOpen, setDeleteConfirmOpen] = React.useState(false)
  const [deleteError, setDeleteError] = React.useState<string | null>(null)

  const installPlugin = useMutation(installPluginMutationOptions)
  const saveConfig = useMutation(savePluginConfigMutationOptions)
  const setEnabled = useMutation(setPluginEnabledMutationOptions)
  const retryPlugin = useMutation(setPluginEnabledMutationOptions)
  const deletePlugin = useMutation(deletePluginMutationOptions)

  const selectedEntry =
    catalogScreen.kind === "ready" ? (catalogScreen.entries.find((entry) => entry.name === selectedCatalogEntry) ?? null) : null

  function configKeyLabel(key: string): string {
    return selectedEntry?.config_keys.find((configKey) => configKey.key === key)?.label ?? key
  }

  const missingSourceReason =
    installMode === "catalog"
      ? saveBlockedByNoCatalogSelection(selectedCatalogEntry)
        ? "Select a plugin from the catalog before installing."
        : null
      : saveBlockedByMissingCustomSource(customTransport, command, url)
  const blankNameReason = saveBlockedByBlankDisplayName(displayName) ? BLANK_DISPLAY_NAME_REASON : null
  const installBlockedReason = missingSourceReason ?? blankNameReason
  const installDisabled = installBlockedReason !== null

  const handleInstall = async () => {
    setInstallError(null)
    try {
      const result =
        installMode === "catalog"
          ? await installPlugin.mutateAsync({
              catalog_entry: selectedCatalogEntry,
              display_name: displayName,
              timeout_ms: timeoutMs,
              config_values: draftConfigValuesToInput(configValues),
            })
          : await installPlugin.mutateAsync({
              display_name: displayName,
              transport: customTransport,
              command: customTransport === "command" ? command : null,
              url: customTransport === "url" ? url : null,
              timeout_ms: timeoutMs,
              config_values: draftConfigValuesToInput(configValues),
            })
      navigate(`/plugins/${result.id}`, { replace: true })
    } catch (err) {
      setInstallError(installFailureMessage(err))
    }
  }

  const handleSaveConfig = async () => {
    if (screen.kind !== "ready" || !screen.plugin) return
    setSaveConfigError(null)
    const editedSecretKeys = Object.entries(configValues)
      .filter(([, draft]) => draft.secret && draft.editing)
      .map(([key]) => key)
    try {
      const result = await saveConfig.mutateAsync({
        pluginId: screen.plugin.id,
        values: draftConfigValuesToInput(configValues),
      })
      loadPlugin(result)
    } catch {
      if (editedSecretKeys.length > 0) {
        for (const key of editedSecretKeys) revertSecretEdit(key)
        setSaveConfigError("Nothing was stored. Try again.")
      } else {
        setSaveConfigError("Couldn't save this configuration. Try again.")
      }
    }
  }

  const handleAddConfigKey = () => {
    if (saveBlockedByBlankConfigKeyName(newKeyName)) return
    addConfigKey(newKeyName.trim(), newKeyValue, newKeySecret)
  }

  const handleDelete = async () => {
    if (screen.kind !== "ready" || !screen.plugin) return
    setDeleteError(null)
    try {
      await deletePlugin.mutateAsync({ pluginId: screen.plugin.id })
      setDeleteConfirmOpen(false)
      navigate("/plugins")
    } catch {
      setDeleteError("Couldn't delete this plugin. Try again.")
    }
  }

  const plugin: Plugin | null = screen.kind === "ready" ? screen.plugin : null
  const title = plugin ? plugin.display_name : "New plugin"
  const configKeyEntries = Object.entries(configValues)

  return (
    <div className="flex flex-col gap-6">
      <h1 className="truncate text-display font-semibold">{title}</h1>

      {screen.kind === "loading" ? (
        <div
          role="status"
          aria-live="polite"
          className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border p-8 text-center"
        >
          <p className="text-body text-foreground">Loading plugin…</p>
        </div>
      ) : null}

      {screen.kind === "error" ? (
        <div className="flex flex-col gap-3">
          <ErrorState message={screen.message} onRetry={() => void pluginQuery.refetch()} />
          <Link to="/plugins" className="touch-target flex items-center text-body text-primary underline-offset-4 hover:underline">
            Back to plugins
          </Link>
        </div>
      ) : null}

      {screen.kind === "ready" && plugin === null ? (
        <>
          <RadioGroup value={installMode} onValueChange={(value) => setInstallMode(value as typeof installMode)} aria-label="Install source">
            <div className="touch-target flex items-center gap-2">
              <RadioGroupItem value="catalog" id="install-mode-catalog" />
              <Label htmlFor="install-mode-catalog">From catalog</Label>
            </div>
            <div className="touch-target flex items-center gap-2">
              <RadioGroupItem value="custom" id="install-mode-custom" />
              <Label htmlFor="install-mode-custom">Custom</Label>
            </div>
          </RadioGroup>

          {installMode === "catalog" ? (
            selectedCatalogEntry === null ? (
              <>
                {catalogScreen.kind === "loading" ? <SkeletonList rows={2} /> : null}
                {catalogScreen.kind === "error" ? (
                  <ErrorState message={catalogScreen.message} onRetry={() => void catalogQuery.refetch()} />
                ) : null}
                {catalogScreen.kind === "empty" ? (
                  <EmptyState
                    heading="No catalog plugins yet."
                    body="Add a plugin by command or URL instead."
                    action={
                      <Button type="button" variant="outline" onClick={() => setInstallMode("custom")}>
                        Custom
                      </Button>
                    }
                  />
                ) : null}
                {catalogScreen.kind === "ready" ? (
                  <ul className="flex flex-col gap-2">
                    {catalogScreen.entries.map((entry) => (
                      <li key={entry.name} className="flex items-center justify-between gap-3 rounded-lg border border-border bg-card p-4">
                        <div className="flex min-w-0 flex-1 flex-col gap-1">
                          <span className="truncate text-body font-medium text-foreground">{entry.name}</span>
                          <span className="truncate text-label text-muted-foreground">{entry.description}</span>
                        </div>
                        <Button
                          type="button"
                          variant="outline"
                          size="sm"
                          className="touch-target shrink-0"
                          onClick={() => selectCatalogEntry(entry)}
                        >
                          Select
                        </Button>
                      </li>
                    ))}
                  </ul>
                ) : null}
              </>
            ) : (
              <div className="flex flex-col gap-3 rounded-lg border border-border bg-card p-4">
                <div className="flex items-center justify-between gap-2">
                  <span className="text-body font-medium text-foreground">{selectedCatalogEntry}</span>
                  <button
                    type="button"
                    className="touch-target text-body text-primary underline-offset-4 hover:underline"
                    onClick={clearCatalogSelection}
                  >
                    Change selection
                  </button>
                </div>
              </div>
            )
          ) : (
            <>
              <RadioGroup
                value={customTransport}
                onValueChange={(value) => setCustomTransport(value as typeof customTransport)}
                aria-label="Plugin transport"
              >
                <div className="touch-target flex items-center gap-2">
                  <RadioGroupItem value="command" id="custom-transport-command" />
                  <Label htmlFor="custom-transport-command">Command</Label>
                </div>
                <div className="touch-target flex items-center gap-2">
                  <RadioGroupItem value="url" id="custom-transport-url" />
                  <Label htmlFor="custom-transport-url">URL</Label>
                </div>
              </RadioGroup>

              {customTransport === "command" ? (
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="plugin-command">Command</Label>
                  <Input id="plugin-command" value={command} onChange={(event) => setCommand(event.target.value)} />
                  <p className="text-label text-muted-foreground">
                    The command that starts this plugin, with any arguments it needs.
                  </p>
                </div>
              ) : (
                <div className="flex flex-col gap-1.5">
                  <Label htmlFor="plugin-url">Server URL</Label>
                  <Input id="plugin-url" className="scroll-field font-mono" value={url} onChange={(event) => setUrl(event.target.value)} />
                </div>
              )}
            </>
          )}

          {installMode === "custom" || selectedCatalogEntry !== null ? (
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="plugin-name">Name</Label>
              <Input id="plugin-name" value={displayName} onChange={(event) => setDisplayName(event.target.value)} />
              {blankNameReason ? <p className="text-body text-destructive">{blankNameReason}</p> : null}
            </div>
          ) : null}

          {configKeyEntries.length > 0 ? (
            <div className="flex flex-col gap-2">
              <p className="text-heading font-semibold text-foreground">Configuration</p>
              {configKeyEntries.map(([key, draft]) => (
                <div key={key} className="flex flex-col gap-1.5 rounded-lg border border-border bg-card p-4">
                  <Label htmlFor={`config-${key}`}>{configKeyLabel(key)}</Label>
                  <Input
                    id={`config-${key}`}
                    type={draft.secret ? "password" : "text"}
                    className={draft.secret ? "scroll-field font-mono" : "scroll-field"}
                    placeholder={draft.secret ? "Not set" : undefined}
                    value={draft.value}
                    onChange={(event) => setConfigValue(key, event.target.value)}
                  />
                </div>
              ))}
            </div>
          ) : null}

          <div className="flex flex-col gap-2 rounded-lg border border-dashed border-border p-4">
            <p className="text-heading font-semibold text-foreground">Add configuration key</p>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="new-key-name">Key</Label>
              <Input id="new-key-name" value={newKeyName} onChange={(event) => setNewKeyName(event.target.value)} />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label htmlFor="new-key-value">Value</Label>
              <Input id="new-key-value" value={newKeyValue} onChange={(event) => setNewKeyValue(event.target.value)} />
            </div>
            <RadioGroup
              value={newKeySecret ? "secret" : "plain"}
              onValueChange={(value) => setNewKeySecret(value === "secret")}
              aria-label="Configuration key kind"
            >
              <div className="touch-target flex items-center gap-2">
                <RadioGroupItem value="plain" id="new-key-plain" />
                <Label htmlFor="new-key-plain">Plain</Label>
              </div>
              <div className="touch-target flex items-center gap-2">
                <RadioGroupItem value="secret" id="new-key-secret" />
                <Label htmlFor="new-key-secret">Secret</Label>
              </div>
            </RadioGroup>
            {saveBlockedByBlankConfigKeyName(newKeyName) && newKeyName !== "" ? (
              <p className="text-body text-destructive">{BLANK_CONFIG_KEY_NAME_REASON}</p>
            ) : null}
            <Button type="button" variant="outline" onClick={handleAddConfigKey} disabled={saveBlockedByBlankConfigKeyName(newKeyName)}>
              Add key
            </Button>
          </div>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="plugin-timeout">Timeout (ms)</Label>
            <Input
              id="plugin-timeout"
              type="number"
              value={timeoutMs}
              onChange={(event) => setTimeoutMs(Number(event.target.value))}
            />
            <p className="text-label text-muted-foreground">
              How long one tool call may run before it fails alone, without ending the turn.
            </p>
          </div>

          {installBlockedReason ? <p className="text-body text-destructive">{installBlockedReason}</p> : null}
          {installError ? <p className="text-body text-destructive">{installError}</p> : null}
          <SubmitButton onSubmit={handleInstall} disabled={installDisabled}>
            Install plugin
          </SubmitButton>
        </>
      ) : null}

      {screen.kind === "ready" && plugin !== null ? (
        <>
          {(() => {
            const status = runtimeStatusDisplay(plugin)
            return (
              <div className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
                <div className="flex items-center gap-2">
                  <Badge variant={status.badgeVariant}>{status.badgeText}</Badge>
                </div>
                {status.caption ? <p className="text-body text-muted-foreground">{status.caption}</p> : null}
                {status.showRetry ? (
                  <Button
                    type="button"
                    variant="outline"
                    className="w-fit"
                    disabled={retryPlugin.isPending}
                    onClick={() => void retryPlugin.mutateAsync({ pluginId: plugin.id, enabled: true })}
                  >
                    Retry now
                  </Button>
                ) : null}
              </div>
            )
          })()}

          <div className="flex flex-col gap-2">
            <p className="text-heading font-semibold text-foreground">{formatToolCount(plugin.tools.length)}</p>
            {plugin.tools.length === 0 ? (
              <p className="text-body text-muted-foreground">No tools right now.</p>
            ) : (
              <ul className="flex flex-col gap-2">
                {plugin.tools.map((tool) => {
                  const collisionText = toolCollisionText(tool.collides_with)
                  return (
                    <li key={tool.name} className="flex flex-col gap-1 rounded-lg border border-border bg-card p-4">
                      <span className="truncate text-body font-medium text-foreground">{tool.name}</span>
                      <span className="text-label text-muted-foreground">{tool.description}</span>
                      {collisionText ? (
                        <Badge variant="denied" className="w-fit">
                          {collisionText}
                        </Badge>
                      ) : null}
                    </li>
                  )
                })}
              </ul>
            )}
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-heading font-semibold text-foreground">{formatConfigKeyCount(configKeyEntries.length)}</p>
            {remoteConfigCaption(plugin.transport) ? (
              <p className="text-label text-muted-foreground">{remoteConfigCaption(plugin.transport)}</p>
            ) : null}
            {configKeyEntries.length === 0 ? <p className="text-body text-muted-foreground">No configuration.</p> : null}
            {configKeyEntries.map(([key, draft]) =>
              draft.secret && !draft.editing ? (
                <div key={key} className="flex flex-col gap-1.5 rounded-lg border border-border bg-card p-4">
                  <Label>{key}</Label>
                  <div className="flex items-center justify-between gap-2">
                    <Badge variant="outline">Set</Badge>
                    <Button type="button" variant="outline" size="sm" onClick={() => startEditingSecret(key)}>
                      Update
                    </Button>
                  </div>
                </div>
              ) : (
                <div key={key} className="flex flex-col gap-1.5 rounded-lg border border-border bg-card p-4">
                  <Label htmlFor={`config-${key}`}>{key}</Label>
                  <Input
                    id={`config-${key}`}
                    type={draft.secret ? "password" : "text"}
                    className={draft.secret ? "scroll-field font-mono" : "scroll-field"}
                    placeholder={draft.secret ? "Not set" : undefined}
                    value={draft.value}
                    onChange={(event) => setConfigValue(key, event.target.value)}
                  />
                </div>
              ),
            )}

            <div className="flex flex-col gap-2 rounded-lg border border-dashed border-border p-4">
              <p className="text-heading font-semibold text-foreground">Add configuration key</p>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="new-key-name">Key</Label>
                <Input id="new-key-name" value={newKeyName} onChange={(event) => setNewKeyName(event.target.value)} />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="new-key-value">Value</Label>
                <Input id="new-key-value" value={newKeyValue} onChange={(event) => setNewKeyValue(event.target.value)} />
              </div>
              <RadioGroup
                value={newKeySecret ? "secret" : "plain"}
                onValueChange={(value) => setNewKeySecret(value === "secret")}
                aria-label="Configuration key kind"
              >
                <div className="touch-target flex items-center gap-2">
                  <RadioGroupItem value="plain" id="new-key-plain" />
                  <Label htmlFor="new-key-plain">Plain</Label>
                </div>
                <div className="touch-target flex items-center gap-2">
                  <RadioGroupItem value="secret" id="new-key-secret" />
                  <Label htmlFor="new-key-secret">Secret</Label>
                </div>
              </RadioGroup>
              {saveBlockedByBlankConfigKeyName(newKeyName) && newKeyName !== "" ? (
                <p className="text-body text-destructive">{BLANK_CONFIG_KEY_NAME_REASON}</p>
              ) : null}
              <Button
                type="button"
                variant="outline"
                onClick={handleAddConfigKey}
                disabled={saveBlockedByBlankConfigKeyName(newKeyName)}
              >
                Add key
              </Button>
            </div>
          </div>

          {saveConfigError ? <p className="text-body text-destructive">{saveConfigError}</p> : null}
          <SubmitButton onSubmit={handleSaveConfig} disabled={recordLoading}>
            Save configuration
          </SubmitButton>

          <div className="flex items-center gap-2 border-t border-border pt-4">
            <Button
              type="button"
              variant="outline"
              disabled={setEnabled.isPending}
              onClick={() => setEnabled.mutate({ pluginId: plugin.id, enabled: !plugin.enabled })}
            >
              {plugin.enabled ? "Disable" : "Enable"}
            </Button>
          </div>

          {plugin.builtin ? (
            <p className="text-label text-muted-foreground">
              Built-in plugins can&apos;t be deleted — only disabled and reconfigured.
            </p>
          ) : (
            <div className="flex flex-col gap-2">
              <AlertDialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
                <Button type="button" variant="destructive" onClick={() => setDeleteConfirmOpen(true)}>
                  Delete plugin
                </Button>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete plugin</AlertDialogTitle>
                    <AlertDialogDescription>
                      {`Delete "${plugin.display_name}"? Its tools will no longer reach the assistant, and any macro or workflow step that used one will be flagged as unresolved.`}
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  {deleteError ? <p className="text-body text-destructive">{deleteError}</p> : null}
                  <AlertDialogFooter>
                    <AlertDialogCancel onClick={() => setDeleteConfirmOpen(false)}>Cancel</AlertDialogCancel>
                    <AlertDialogAction variant="destructive" disabled={deletePlugin.isPending} onClick={() => void handleDelete()}>
                      Delete plugin
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          )}
        </>
      ) : null}
    </div>
  )
}
