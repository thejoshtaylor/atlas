import { QueryClientProvider } from "@tanstack/react-query"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { Toaster } from "@/components/ui/sonner"
import { AppShell } from "@/components/layout/AppShell"
import { AuthGuard } from "@/components/layout/AuthGuard"
import { SetupGuard } from "@/components/layout/SetupGuard"
import { queryClient } from "@/lib/queryClient"
import { CalibrationRoute } from "@/routes/calibration/CalibrationRoute"
import { HomeRoute } from "@/routes/HomeRoute"
import { PlaceholderRoute } from "@/routes/PlaceholderRoute"
import { SetupRoute } from "@/routes/SetupRoute"
import { SignInRoute } from "@/routes/SignInRoute"

/**
 * The route tree (CD-1, executor's discretion within the phone-first
 * constraint): a public sign-in route, a setup route, and an
 * authenticated shell holding the screens later plans fill.
 * `SetupGuard` wraps everything -- the setup-incomplete gate is a
 * product-wide state, not something only authenticated screens can hit.
 *
 * `/calibration` (plan 03-06) is not on `AppShell`'s nav -- it is the
 * component plan 03-10 mounts as one step of the setup wizard, reachable
 * here by direct URL only so it has a real address to test against
 * before that wizard exists.
 */
function AppRoutes() {
  return (
    <Routes>
      <Route path="/sign-in" element={<SignInRoute />} />
      <Route path="/setup" element={<SetupRoute />} />
      <Route element={<AuthGuard />}>
        <Route element={<AppShell />}>
          <Route index element={<HomeRoute />} />
          <Route path="/policy" element={<PlaceholderRoute title="Safety policy" />} />
          <Route path="/accounts" element={<PlaceholderRoute title="Accounts" />} />
          <Route path="/settings" element={<PlaceholderRoute title="Settings" />} />
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
