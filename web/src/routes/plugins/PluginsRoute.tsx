import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { Link } from "react-router-dom"
import { Badge } from "@/components/ui/badge"
import { Button } from "@/components/ui/button"
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
import { ErrorState } from "@/components/state/ErrorState"
import { SkeletonList } from "@/components/state/SkeletonList"
import {
  PLUGINS_QUERY_KEY,
  deletePluginMutationOptions,
  fetchPlugins,
  setPluginEnabledMutationOptions,
  type Plugin,
} from "@/lib/plugins"
import {
  collidingToolCount,
  derivePluginsScreenState,
  formatCollisionSummary,
  runtimeStatusDisplay,
  showDeleteControl,
} from "./derivePluginsScreenState"

// PLUG-03, PLUG-08, D-14, 06-UI-SPEC.md's Focal Point row: "the card list
// itself ... an admin comes here to see what is running and what needs
// attention before deciding what to install or turn off." Built from
// MacrosRoute.tsx's own row/list shape, with two deltas 06-UI-SPEC.md
// names explicitly: no zero-plugins branch renders at all (the list is
// never empty -- Home Assistant and weather are always-seeded builtin
// rows), and a builtin row renders no Delete control (absent, never
// disabled).

function PluginRow({ plugin, disabled }: { plugin: Plugin; disabled: boolean }) {
  const [confirmOpen, setConfirmOpen] = React.useState(false)
  const setEnabled = useMutation(setPluginEnabledMutationOptions)
  const remove = useMutation(deletePluginMutationOptions)
  const status = runtimeStatusDisplay(plugin)
  const collisions = collidingToolCount(plugin)
  const rowControlsDisabled = disabled || setEnabled.isPending || remove.isPending

  return (
    <li className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
      <div className="flex items-center justify-between gap-3">
        <Link to={`/plugins/${plugin.id}`} className="min-w-0 flex-1 truncate text-body font-medium text-foreground">
          {plugin.display_name}
        </Link>
        <Badge variant={status.badgeVariant} className="shrink-0">
          {status.badgeText}
        </Badge>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <span className="text-label text-muted-foreground">{plugin.transport}</span>
        {plugin.builtin ? <Badge variant="outline">Built-in</Badge> : null}
        {collisions > 0 ? <Badge variant="denied">{formatCollisionSummary(collisions)}</Badge> : null}
      </div>

      <div className="flex items-center justify-end gap-2">
        <Button
          type="button"
          variant="outline"
          size="sm"
          className="touch-target"
          disabled={rowControlsDisabled}
          onClick={() => setEnabled.mutate({ pluginId: plugin.id, enabled: !plugin.enabled })}
        >
          {plugin.enabled ? "Disable" : "Enable"}
        </Button>
        {showDeleteControl(plugin) ? (
          <AlertDialog open={confirmOpen} onOpenChange={setConfirmOpen}>
            <Button
              type="button"
              variant="outline"
              size="sm"
              className="touch-target shrink-0"
              disabled={rowControlsDisabled}
              onClick={() => setConfirmOpen(true)}
            >
              Delete
            </Button>
            <AlertDialogContent>
              <AlertDialogHeader>
                <AlertDialogTitle>Delete plugin</AlertDialogTitle>
                <AlertDialogDescription>
                  {`Delete "${plugin.display_name}"? Its tools will no longer reach the assistant, and any macro or workflow step that used one will be flagged as unresolved.`}
                </AlertDialogDescription>
              </AlertDialogHeader>
              <AlertDialogFooter>
                <AlertDialogCancel onClick={() => setConfirmOpen(false)}>Cancel</AlertDialogCancel>
                <AlertDialogAction
                  variant="destructive"
                  onClick={() => {
                    setConfirmOpen(false)
                    void remove.mutateAsync({ pluginId: plugin.id })
                  }}
                >
                  Delete plugin
                </AlertDialogAction>
              </AlertDialogFooter>
            </AlertDialogContent>
          </AlertDialog>
        ) : null}
      </div>
    </li>
  )
}

export function PluginsRoute() {
  const query = useQuery({ queryKey: PLUGINS_QUERY_KEY, queryFn: fetchPlugins })
  const screen = derivePluginsScreenState(query)
  const controlsDisabled = screen.kind !== "ready"

  return (
    <div className="flex flex-col gap-6">
      <div className="flex items-center justify-between gap-3">
        <h1 className="text-display font-semibold">Plugins</h1>
        <Button asChild disabled={controlsDisabled}>
          <Link to="/plugins/new">New plugin</Link>
        </Button>
      </div>

      {screen.kind === "loading" ? <SkeletonList rows={3} /> : null}

      {screen.kind === "error" ? (
        <ErrorState message={screen.message} onRetry={() => void query.refetch()} />
      ) : null}

      {screen.kind === "ready" ? (
        <ul className="flex flex-col gap-2">
          {screen.plugins.map((plugin) => (
            <PluginRow key={plugin.id} plugin={plugin} disabled={controlsDisabled} />
          ))}
        </ul>
      ) : null}
    </div>
  )
}
