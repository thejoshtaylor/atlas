import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "PluginEditorRoute.tsx"), "utf-8")
const APP_SOURCE = readFileSync(join(import.meta.dir, "..", "..", "App.tsx"), "utf-8")
const SHELL_SOURCE = readFileSync(join(import.meta.dir, "..", "..", "components", "layout", "AppShell.tsx"), "utf-8")

describe("PluginEditorRoute -- one component serves both /plugins/new and /plugins/:id", () => {
  test("branches on whether an id is present, the same shape MacroEditorRoute uses", () => {
    expect(SOURCE).toMatch(/const pluginId = params\.id \? Number\(params\.id\) : null/)
    expect(SOURCE).toMatch(/enabled: pluginId !== null/)
  })

  test("the title is the plugin's own display name, or \"New plugin\" for an unsaved one", () => {
    expect(SOURCE).toMatch(/const title = plugin \? plugin\.display_name : "New plugin"/)
  })
})

describe("PluginEditorRoute -- the install flow is a picker between a curated catalog and a hand-entered plugin", () => {
  test("the install-mode RadioGroup offers From catalog / Custom, both touch-target", () => {
    expect(SOURCE).toMatch(/aria-label="Install source"/)
    expect(SOURCE).toMatch(/<Label htmlFor="install-mode-catalog">From catalog<\/Label>/)
    expect(SOURCE).toMatch(/<Label htmlFor="install-mode-custom">Custom<\/Label>/)
    const matches = SOURCE.match(/touch-target/g) ?? []
    expect(matches.length).toBeGreaterThanOrEqual(6)
  })

  test("selecting a catalog entry collapses the list in place with a Change selection link", () => {
    expect(SOURCE).toMatch(/onClick=\{\(\) => selectCatalogEntry\(entry\)\}/)
    expect(SOURCE).toMatch(/onClick=\{clearCatalogSelection\}/)
    expect(SOURCE).toMatch(/Change selection/)
  })

  test("the empty catalog state offers an action that switches to the Custom picker", () => {
    expect(SOURCE).toMatch(/heading="No catalog plugins yet\."/)
    expect(SOURCE).toMatch(/body="Add a plugin by command or URL instead\."/)
    expect(SOURCE).toMatch(/onClick=\{\(\) => setInstallMode\("custom"\)\}/)
  })

  test("custom mode offers a Command/URL transport picker, one field at a time", () => {
    expect(SOURCE).toMatch(/aria-label="Plugin transport"/)
    expect(SOURCE).toMatch(/customTransport === "command" \? \(/)
    expect(SOURCE).toMatch(/id="plugin-url" className="scroll-field font-mono"/)
  })
})

describe("PluginEditorRoute -- install is blocked, with a named reason, until a source is chosen", () => {
  test("saveBlockedByNoCatalogSelection / saveBlockedByMissingCustomSource / saveBlockedByBlankDisplayName all feed installDisabled", () => {
    expect(SOURCE).toMatch(/saveBlockedByNoCatalogSelection\(selectedCatalogEntry\)/)
    expect(SOURCE).toMatch(/saveBlockedByMissingCustomSource\(customTransport, command, url\)/)
    expect(SOURCE).toMatch(/saveBlockedByBlankDisplayName\(displayName\)/)
    expect(SOURCE).toMatch(/const installDisabled = installBlockedReason !== null/)
  })

  test("install failure is inline, under Install plugin, and never clears the typed form", () => {
    expect(SOURCE).toMatch(/setInstallError\(installFailureMessage\(err\)\)/)
    expect(SOURCE).not.toMatch(/catch \(err\) \{[\s\S]{0,80}loadBlank\(\)/)
  })
})

describe("PluginEditorRoute -- on an existing plugin, the runtime status block leads", () => {
  test("runtimeStatusDisplay drives the badge, and the caption is the server's own reason, unmodified", () => {
    expect(SOURCE).toMatch(/const status = runtimeStatusDisplay\(plugin\)/)
    expect(SOURCE).toMatch(/status\.caption \? <p className="text-body text-muted-foreground">\{status\.caption\}<\/p> : null/)
  })

  test("Retry now only renders when showRetry is true, and calls set-enabled true (the live-reconcile primitive)", () => {
    expect(SOURCE).toMatch(/status\.showRetry \? \(/)
    expect(SOURCE).toMatch(/retryPlugin\.mutate\(\s*\{ pluginId: plugin\.id, enabled: true \}/)
  })

  // IN-02 (code review): the one control whose entire purpose is a plugin
  // that is already failing said nothing when the retry failed too.
  test("a retry that fails again says so, rather than leaving an unhandled rejection", () => {
    expect(SOURCE).toMatch(/onError: \(\) => setRetryError\(RETRY_FAILED\)/)
    expect(SOURCE).toMatch(/\{retryError \? <p className="text-body text-destructive">\{retryError\}<\/p> : null\}/)
  })
})

describe("PluginEditorRoute -- a plugin with zero tools shows a caption, not a generic empty panel", () => {
  test("\"No tools right now.\" renders beneath the tools heading, no EmptyState import", () => {
    expect(SOURCE).toMatch(/plugin\.tools\.length === 0 \? \(\s*<p className="text-body text-muted-foreground">No tools right now\.<\/p>/)
  })

  test("a colliding tool shows its per-tool collision text via toolCollisionText", () => {
    expect(SOURCE).toMatch(/const collisionText = toolCollisionText\(tool\.collides_with\)/)
  })
})

describe("PluginEditorRoute -- a secret configuration value is never pre-filled", () => {
  test("an already-set secret renders the Set badge and an Update control that opens a blank input", () => {
    expect(SOURCE).toMatch(/draft\.secret && !draft\.editing \? \(/)
    expect(SOURCE).toMatch(/<Badge variant="outline">Set<\/Badge>/)
    expect(SOURCE).toMatch(/onClick=\{\(\) => startEditingSecret\(key\)\}/)
  })

  test("an unset or actively-edited secret is a blank password input placeholdered Not set, never pre-filled with value", () => {
    expect(SOURCE).toMatch(/placeholder=\{draft\.secret \? "Not set" : undefined\}/)
  })
})

describe("PluginEditorRoute -- a failed secret save returns the field to its prior state and stores nothing", () => {
  test("editedSecretKeys are reverted and the exact SettingsRoute.tsx failure copy is shown", () => {
    expect(SOURCE).toMatch(/for \(const key of editedSecretKeys\) revertSecretEdit\(key\)/)
    expect(SOURCE).toMatch(/setSaveConfigError\("Nothing was stored\. Try again\."\)/)
  })

  test("a save failure with no secret edits shows the general config-failure copy instead", () => {
    expect(SOURCE).toMatch(/setSaveConfigError\("Couldn't save this configuration\. Try again\."\)/)
  })
})

describe("PluginEditorRoute -- a builtin plugin has no delete control, only an explanatory caption", () => {
  test("plugin.builtin renders the caption, never a disabled button", () => {
    expect(SOURCE).toMatch(/plugin\.builtin \? \(/)
    expect(SOURCE).toMatch(/Built-in plugins can&apos;t be deleted — only disabled and reconfigured\./)
  })
})

describe("PluginEditorRoute -- a failed delete leaves the dialog open with the failure inside it", () => {
  test("the dialog is not closed on catch, and the failure renders inside AlertDialogContent", () => {
    expect(SOURCE).toMatch(/catch \{\s*setDeleteError\("Couldn't delete this plugin\. Try again\."\)/)
    expect(SOURCE).toMatch(/\{deleteError \? <p className="text-body text-destructive">\{deleteError\}<\/p> : null\}/)
  })

  test("the confirm button is disabled while the delete is pending, so it visibly returns from its pending state", () => {
    expect(SOURCE).toMatch(/disabled=\{deletePlugin\.isPending\}/)
  })
})

describe("PluginEditorRoute -- disable/enable act with no confirmation", () => {
  test("the editor's own Disable/Enable button calls the mutation directly, no AlertDialog wraps it", () => {
    expect(SOURCE).toMatch(/onClick=\{\(\) => setEnabled\.mutate\(\{ pluginId: plugin\.id, enabled: !plugin\.enabled \}\)\}/)
  })
})

describe("App.tsx -- all three plugin paths sit inside the admin-gated route block", () => {
  test("PluginsRoute and PluginEditorRoute are imported and mounted at /plugins, /plugins/new, /plugins/:id", () => {
    expect(APP_SOURCE).toMatch(/import \{ PluginEditorRoute \} from "@\/routes\/plugins\/PluginEditorRoute"/)
    expect(APP_SOURCE).toMatch(/import \{ PluginsRoute \} from "@\/routes\/plugins\/PluginsRoute"/)
    expect(APP_SOURCE).toMatch(/<Route path="\/plugins" element=\{<PluginsRoute \/>\} \/>/)
    expect(APP_SOURCE).toMatch(/<Route path="\/plugins\/new" element=\{<PluginEditorRoute \/>\} \/>/)
    expect(APP_SOURCE).toMatch(/<Route path="\/plugins\/:id" element=\{<PluginEditorRoute \/>\} \/>/)
  })

  test("all three routes sit inside the admin-gated block, alongside /accounts and /settings, not the operator-gated one", () => {
    const adminBlock = APP_SOURCE.match(
      /<Route element=\{<RequireRole minimum="admin" \/>\}>[\s\S]*?<\/Route>\s*<Route path="\/calibration"/,
    )?.[0]
    expect(adminBlock).toBeDefined()
    expect(adminBlock).toMatch(/path="\/accounts"/)
    expect(adminBlock).toMatch(/path="\/plugins"/)
    expect(adminBlock).toMatch(/path="\/plugins\/new"/)
    expect(adminBlock).toMatch(/path="\/plugins\/:id"/)
  })
})

describe("AppShell.tsx -- a Plugins navigation entry exists at admin level", () => {
  test("nav item points at /plugins with minimumRole admin", () => {
    expect(SHELL_SOURCE).toMatch(/\{ to: "\/plugins", label: "Plugins", minimumRole: "admin" \}/)
  })
})

describe("PluginEditorRoute -- IN-04: the per-plugin deadline is editable after install", () => {
  test("the edit screen renders its own Timeout (ms) field, with the contract's helper text", () => {
    expect(SOURCE).toMatch(/id="plugin-timeout-edit"/)
    expect(SOURCE).toMatch(
      /How long one tool call may run before it fails alone, without ending the turn\./,
    )
  })

  test("saving the configuration sends the timeout with it, rather than loading a value it cannot save", () => {
    expect(SOURCE).toMatch(/values: draftConfigValuesToInput\(configValues\),\s*\n\s*timeout_ms: timeoutMs,/)
  })
})
