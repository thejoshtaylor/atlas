import * as React from "react"
import { useMutation, useQuery } from "@tanstack/react-query"
import { ChevronDown, ChevronUp } from "lucide-react"
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
import { ErrorState } from "@/components/state/ErrorState"
import { SubmitButton } from "@/components/state/SubmitButton"
import {
  MACROS_QUERY_KEY,
  createMacroMutationOptions,
  deleteMacroMutationOptions,
  fetchMacro,
  fetchMacros,
  macroQueryKey,
  updateMacroMutationOptions,
  type ConflictAnnotation,
} from "@/lib/macros"
import {
  draftActionsToInput,
  isFirstAction,
  isLastAction,
  saveBlockedByEmptyActions,
  useMacroDraftStore,
} from "@/stores/macroDraftStore"
import {
  actionConflictDisplay,
  deriveMacroEditorState,
  deriveReplyPrecacheState,
  DUPLICATE_PHRASE_MESSAGE,
  isDuplicatePhrase,
} from "./deriveMacroEditorState"

// MACRO-03, D-11, D-12, T-04-37: every rule this screen enforces (zero
// actions, a duplicate phrase) is enforced again on the route in plan
// 04-06 -- these blocks are for the operator's benefit, not the
// boundary's. 04-UI-SPEC.md's Focal Point row: "the phrase field and the
// ordered action list together ... the reply field and its precache
// state sit below, deliberately quieter." No free-text box parses
// configuration here (D-11) -- an action's domain/service/entity id are
// three structured `Input` fields, folded into the exact
// `ha_call_service` shape `routes/macros.py`'s own conflict check
// understands, never a markup-language box handed a parser.

export function MacroEditorRoute() {
  const params = useParams<{ id: string }>()
  const navigate = useNavigate()
  const macroId = params.id ? Number(params.id) : null

  const macroQuery = useQuery({
    queryKey: macroId !== null ? macroQueryKey(macroId) : ["macros", "new"],
    queryFn: () => fetchMacro(macroId as number),
    enabled: macroId !== null,
  })
  const screen = deriveMacroEditorState(macroId === null ? null : macroQuery)

  // For the duplicate-phrase check only -- an advisory, client-side
  // courtesy (T-04-37); the real check re-runs on the route regardless.
  const macrosListQuery = useQuery({ queryKey: MACROS_QUERY_KEY, queryFn: fetchMacros })

  const phrase = useMacroDraftStore((state) => state.phrase)
  const reply = useMacroDraftStore((state) => state.reply)
  const actions = useMacroDraftStore((state) => state.actions)
  const aliases = useMacroDraftStore((state) => state.aliases)
  const setPhrase = useMacroDraftStore((state) => state.setPhrase)
  const setReply = useMacroDraftStore((state) => state.setReply)
  const addAction = useMacroDraftStore((state) => state.addAction)
  const removeAction = useMacroDraftStore((state) => state.removeAction)
  const moveActionUp = useMacroDraftStore((state) => state.moveActionUp)
  const moveActionDown = useMacroDraftStore((state) => state.moveActionDown)
  const updateAction = useMacroDraftStore((state) => state.updateAction)
  const addAlias = useMacroDraftStore((state) => state.addAlias)
  const removeAlias = useMacroDraftStore((state) => state.removeAlias)
  const loadMacro = useMacroDraftStore((state) => state.loadMacro)
  const loadBlank = useMacroDraftStore((state) => state.loadBlank)

  // The draft only ever loads from the server once per record version --
  // keyed on `id:updated_at` so a save's own fresh response (real ids,
  // a new `updated_at`) re-syncs the draft, but a background refetch of
  // the *same* version never clobbers an operator's unsaved typing.
  const loadedKeyRef = React.useRef<string | null>(null)
  React.useEffect(() => {
    if (screen.kind !== "ready") return
    const key = screen.macro ? `${screen.macro.id}:${screen.macro.updated_at}` : "new"
    if (loadedKeyRef.current === key) return
    loadedKeyRef.current = key
    if (screen.macro) loadMacro(screen.macro)
    else loadBlank()
  }, [screen, loadMacro, loadBlank])

  const [synthesisFailedMessage, setSynthesisFailedMessage] = React.useState<string | null>(null)
  const [saveError, setSaveError] = React.useState<string | null>(null)
  const [newAliasValue, setNewAliasValue] = React.useState("")
  const [newDomain, setNewDomain] = React.useState("")
  const [newService, setNewService] = React.useState("")
  const [newEntityId, setNewEntityId] = React.useState("")
  const [deleteConfirmOpen, setDeleteConfirmOpen] = React.useState(false)

  const createMacro = useMutation(createMacroMutationOptions)
  const updateMacro = useMutation(updateMacroMutationOptions)
  const deleteMacro = useMutation(deleteMacroMutationOptions)

  const duplicate = macrosListQuery.data ? isDuplicatePhrase(phrase, macrosListQuery.data, macroId) : false
  const zeroActionsReason = saveBlockedByEmptyActions(actions)
  const recordLoading = screen.kind !== "ready"
  const saveDisabled = recordLoading || actions.length === 0 || duplicate

  const canAddAction = newDomain.trim() !== "" && newService.trim() !== "" && newEntityId.trim() !== ""

  const handleAddAction = () => {
    if (!canAddAction) return
    addAction()
    const current = useMacroDraftStore.getState().actions
    const key = current[current.length - 1]!.key
    updateAction(key, { domain: newDomain.trim(), service: newService.trim(), entityId: newEntityId.trim() })
    setNewDomain("")
    setNewService("")
    setNewEntityId("")
  }

  const handleAddAlias = () => {
    if (!newAliasValue.trim()) return
    addAlias(newAliasValue)
    setNewAliasValue("")
  }

  const handleSave = async () => {
    setSaveError(null)
    const input = { phrase, aliases, reply, actions: draftActionsToInput(actions) }
    try {
      const result =
        macroId === null ? await createMacro.mutateAsync(input) : await updateMacro.mutateAsync({ macroId, ...input })
      setSynthesisFailedMessage(result.reply_synthesis_degraded ? result.reply_synthesis_message : null)
      if (macroId === null) {
        navigate(`/macros/${result.id}`, { replace: true })
      }
    } catch {
      setSaveError("Couldn't save this macro. Try again.")
    }
  }

  const handleDelete = async () => {
    if (macroId === null) return
    await deleteMacro.mutateAsync({ macroId })
    navigate("/macros")
  }

  const precache = deriveReplyPrecacheState(screen.kind === "ready" ? screen.macro : null, synthesisFailedMessage)
  const conflictByKey = new Map<string, ConflictAnnotation>(
    screen.kind === "ready" && screen.macro ? screen.macro.actions.map((action) => [String(action.id), action.conflict]) : [],
  )

  const title = screen.kind === "ready" && screen.macro ? screen.macro.phrase : "New macro"

  return (
    <div className="flex flex-col gap-6">
      <h1 className="truncate text-display font-semibold">{title}</h1>

      {screen.kind === "loading" ? (
        <div
          role="status"
          aria-live="polite"
          className="flex flex-col items-center gap-2 rounded-lg border border-dashed border-border p-8 text-center"
        >
          <p className="text-body text-foreground">Loading macro…</p>
        </div>
      ) : null}

      {screen.kind === "error" ? (
        <div className="flex flex-col gap-3">
          <ErrorState message={screen.message} onRetry={() => void macroQuery.refetch()} />
          <Link to="/macros" className="touch-target flex items-center text-body text-primary underline-offset-4 hover:underline">
            Back to macros
          </Link>
        </div>
      ) : null}

      {screen.kind === "ready" ? (
        <>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="macro-phrase">Phrase</Label>
            <Input id="macro-phrase" value={phrase} onChange={(event) => setPhrase(event.target.value)} />
            {duplicate ? <p className="text-body text-destructive">{DUPLICATE_PHRASE_MESSAGE}</p> : null}
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-heading font-semibold text-foreground">Actions</p>
            {zeroActionsReason ? <p className="text-body text-destructive">{zeroActionsReason}</p> : null}

            <ol className="flex flex-col gap-2">
              {actions.map((action, index) => {
                const conflict = actionConflictDisplay(conflictByKey.get(action.key) ?? "ok")
                return (
                  <li key={action.key} className="flex flex-col gap-2 rounded-lg border border-border bg-card p-4">
                    <div className="flex items-center justify-between gap-2">
                      <div className="flex min-w-0 flex-1 items-baseline gap-2">
                        <span className="text-label text-muted-foreground">{index + 1}.</span>
                        <span className="truncate text-body font-medium text-foreground">
                          {`${action.domain}.${action.service} → ${action.entityId}`}
                        </span>
                      </div>
                      <Button
                        type="button"
                        variant="outline"
                        size="sm"
                        className="shrink-0"
                        onClick={() => removeAction(action.key)}
                      >
                        Remove
                      </Button>
                    </div>
                    {conflict.kind === "denied" || conflict.kind === "not_found" ? (
                      <Badge variant="denied" className="w-fit">
                        {conflict.text}
                      </Badge>
                    ) : null}
                    {conflict.kind === "unknown" ? <p className="text-label text-muted-foreground">{conflict.text}</p> : null}
                    <div className="flex justify-end gap-2">
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-sm"
                        aria-label="Move up"
                        disabled={isFirstAction(actions, action.key)}
                        onClick={() => moveActionUp(action.key)}
                      >
                        <ChevronUp className="size-4" />
                      </Button>
                      <Button
                        type="button"
                        variant="outline"
                        size="icon-sm"
                        aria-label="Move down"
                        disabled={isLastAction(actions, action.key)}
                        onClick={() => moveActionDown(action.key)}
                      >
                        <ChevronDown className="size-4" />
                      </Button>
                    </div>
                  </li>
                )
              })}
            </ol>

            <div className="flex flex-col gap-2 rounded-lg border border-dashed border-border p-4">
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="new-action-domain">Domain</Label>
                <Input id="new-action-domain" placeholder="light" value={newDomain} onChange={(event) => setNewDomain(event.target.value)} />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="new-action-service">Service</Label>
                <Input
                  id="new-action-service"
                  placeholder="turn_off"
                  value={newService}
                  onChange={(event) => setNewService(event.target.value)}
                />
              </div>
              <div className="flex flex-col gap-1.5">
                <Label htmlFor="new-action-entity">Entity id</Label>
                <Input
                  id="new-action-entity"
                  placeholder="light.bedroom"
                  value={newEntityId}
                  onChange={(event) => setNewEntityId(event.target.value)}
                />
              </div>
              <Button type="button" variant="outline" onClick={handleAddAction} disabled={!canAddAction}>
                Add action
              </Button>
            </div>
          </div>

          <div className="flex flex-col gap-2">
            <p className="text-heading font-semibold text-foreground">Aliases</p>
            {aliases.length > 0 ? (
              <ul className="flex flex-col gap-2">
                {aliases.map((alias) => (
                  <li key={alias} className="flex items-center justify-between gap-2 rounded-lg border border-border bg-card p-4">
                    <span className="truncate text-body text-foreground">{alias}</span>
                    <Button type="button" variant="outline" size="sm" onClick={() => removeAlias(alias)}>
                      Remove
                    </Button>
                  </li>
                ))}
              </ul>
            ) : null}
            <div className="flex gap-2">
              <Input
                placeholder="Add an alias"
                value={newAliasValue}
                onChange={(event) => setNewAliasValue(event.target.value)}
              />
              <Button type="button" variant="outline" onClick={handleAddAlias}>
                Add alias
              </Button>
            </div>
          </div>

          <div className="flex flex-col gap-1.5">
            <div className="flex items-center gap-2">
              <Label htmlFor="macro-reply">Reply</Label>
              {precache === "cached" ? <Badge variant="secondary">Reply cached · ready</Badge> : null}
            </div>
            <Input id="macro-reply" className="scroll-field" value={reply} onChange={(event) => setReply(event.target.value)} />
            {precache === "failed" ? (
              <div className="flex flex-col gap-2">
                <p className="text-body text-destructive">{synthesisFailedMessage}</p>
                <Button type="button" variant="outline" onClick={() => void handleSave()}>
                  Retry
                </Button>
              </div>
            ) : null}
          </div>

          {saveError ? <p className="text-body text-destructive">{saveError}</p> : null}
          <SubmitButton onSubmit={handleSave} disabled={saveDisabled}>
            Save macro
          </SubmitButton>

          {macroId !== null ? (
            <div className="flex flex-col gap-2 border-t border-border pt-4">
              <AlertDialog open={deleteConfirmOpen} onOpenChange={setDeleteConfirmOpen}>
                <Button type="button" variant="destructive" onClick={() => setDeleteConfirmOpen(true)}>
                  Delete macro
                </Button>
                <AlertDialogContent>
                  <AlertDialogHeader>
                    <AlertDialogTitle>Delete macro</AlertDialogTitle>
                    <AlertDialogDescription>
                      {`Delete "${phrase}"? This macro will no longer run, and its cached reply is discarded.`}
                    </AlertDialogDescription>
                  </AlertDialogHeader>
                  <AlertDialogFooter>
                    <AlertDialogCancel onClick={() => setDeleteConfirmOpen(false)}>Cancel</AlertDialogCancel>
                    <AlertDialogAction
                      variant="destructive"
                      onClick={() => {
                        setDeleteConfirmOpen(false)
                        void handleDelete()
                      }}
                    >
                      Delete macro
                    </AlertDialogAction>
                  </AlertDialogFooter>
                </AlertDialogContent>
              </AlertDialog>
            </div>
          ) : null}
        </>
      ) : null}
    </div>
  )
}
