/**
 * The authenticated shell's index -- a real, reachable placeholder until
 * a later plan gives it substance. `AppShell` (the header, the
 * navigation, and `<Outlet />`) is what this mounts into.
 */
export function HomeRoute() {
  return (
    <div className="flex flex-col gap-2">
      <h1 className="text-display font-semibold">spire-voice</h1>
      <p className="text-body text-muted-foreground">
        Signed in. Use the navigation to reach policy, accounts, and settings.
      </p>
    </div>
  )
}
