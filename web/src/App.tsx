import { QueryClientProvider } from "@tanstack/react-query"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { Toaster } from "@/components/ui/sonner"
import { AppShell } from "@/components/layout/AppShell"
import { AuthGuard } from "@/components/layout/AuthGuard"
import { RequireRole } from "@/components/layout/RequireRole"
import { SetupGuard } from "@/components/layout/SetupGuard"
import { queryClient } from "@/lib/queryClient"
import { AcceptInviteRoute } from "@/routes/auth/AcceptInviteRoute"
import { SignInRoute } from "@/routes/auth/SignInRoute"
import { AccountsRoute } from "@/routes/accounts/AccountsRoute"
import { CalibrationRoute } from "@/routes/calibration/CalibrationRoute"
import { DevMicRoute } from "@/routes/dev-mic/DevMicRoute"
import { HomeRoute } from "@/routes/HomeRoute"
import { MacroEditorRoute } from "@/routes/macros/MacroEditorRoute"
import { MacrosRoute } from "@/routes/macros/MacrosRoute"
import { PluginEditorRoute } from "@/routes/plugins/PluginEditorRoute"
import { PluginsRoute } from "@/routes/plugins/PluginsRoute"
import { PolicyRoute } from "@/routes/policy/PolicyRoute"
import { SettingsRoute } from "@/routes/settings/SettingsRoute"
import { WorkflowEditorRoute } from "@/routes/workflows/WorkflowEditorRoute"
import { WorkflowsRoute } from "@/routes/workflows/WorkflowsRoute"
import { AudioSourceStep } from "@/routes/wizard/AudioSourceStep"
import { CreateAdminStep } from "@/routes/wizard/CreateAdminStep"
import { HubStep } from "@/routes/wizard/HubStep"
import { ProviderSetStep } from "@/routes/wizard/ProviderSetStep"
import { RoomStep } from "@/routes/wizard/RoomStep"
import { WizardRoute } from "@/routes/wizard/WizardRoute"

/**
 * The route tree (CD-1, executor's discretion within the phone-first
 * constraint): a public sign-in route, the first-run wizard (plan
 * 03-10), and an authenticated shell holding every other screen.
 * `SetupGuard` wraps everything -- the setup-incomplete gate is a
 * product-wide state, not something only authenticated screens can hit.
 *
 * The wizard's own steps sit outside `AppShell` (no site nav during
 * first-run setup, matching the standalone treatment sign-in already
 * uses) -- `admin_account` needs no session (there is no session to have
 * yet), every other step is behind `AuthGuard` + `RequireRole
 * minimum="admin"` (every wizard route, `routes/wizard.py`, requires
 * `Role.ADMIN`).
 *
 * `/calibration` (plan 03-06) stays reachable standalone, inside
 * `AppShell`, for an operator revisiting the test outside the wizard;
 * `RoomStep` (`/setup/room`) mounts the same component, never a second
 * implementation.
 */
function AppRoutes() {
  return (
    <Routes>
      <Route path="/sign-in" element={<SignInRoute />} />
      <Route path="/invite/:token" element={<AcceptInviteRoute />} />
      <Route path="/setup" element={<WizardRoute />} />
      <Route path="/setup/admin_account" element={<CreateAdminStep />} />
      <Route element={<AuthGuard />}>
        <Route element={<RequireRole minimum="admin" />}>
          <Route path="/setup/hub" element={<HubStep />} />
          <Route path="/setup/provider_set" element={<ProviderSetStep />} />
          <Route path="/setup/audio_source" element={<AudioSourceStep />} />
          <Route path="/setup/room" element={<RoomStep />} />
        </Route>
        <Route element={<AppShell />}>
          <Route index element={<HomeRoute />} />
          <Route element={<RequireRole minimum="operator" />}>
            <Route path="/policy" element={<PolicyRoute />} />
            <Route path="/macros" element={<MacrosRoute />} />
            <Route path="/macros/new" element={<MacroEditorRoute />} />
            <Route path="/macros/:id" element={<MacroEditorRoute />} />
            <Route path="/workflows" element={<WorkflowsRoute />} />
            <Route path="/workflows/new" element={<WorkflowEditorRoute />} />
            <Route path="/workflows/:id" element={<WorkflowEditorRoute />} />
            <Route path="/dev-mic" element={<DevMicRoute />} />
          </Route>
          <Route element={<RequireRole minimum="admin" />}>
            <Route path="/accounts" element={<AccountsRoute />} />
            <Route path="/settings" element={<SettingsRoute />} />
            <Route path="/plugins" element={<PluginsRoute />} />
            <Route path="/plugins/new" element={<PluginEditorRoute />} />
            <Route path="/plugins/:id" element={<PluginEditorRoute />} />
          </Route>
          <Route path="/calibration" element={<CalibrationRoute />} />
        </Route>
      </Route>
      <Route path="*" element={<Navigate to="/" replace />} />
    </Routes>
  )
}

function App() {
  return (
    <QueryClientProvider client={queryClient}>
      <BrowserRouter>
        <SetupGuard>
          <AppRoutes />
        </SetupGuard>
      </BrowserRouter>
      <Toaster />
    </QueryClientProvider>
  )
}

export default App
