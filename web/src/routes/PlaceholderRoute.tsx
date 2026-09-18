/**
 * A real, reachable stand-in for a screen a later plan in this phase
 * builds (the safety policy editor -- 03-07/03-08; accounts and invites
 * -- 03-06; settings -- 03-08). `AppShell`'s navigation already points
 * at these paths so the shell is fully wired from this plan forward,
 * even though the content behind each link arrives incrementally.
 */
export function PlaceholderRoute({ title }: { title: string }) {
  return (
    <div className="flex flex-col gap-2">
      <h1 className="text-display font-semibold">{title}</h1>
      <p className="text-body text-muted-foreground">This screen arrives in a later plan.</p>
    </div>
  )
}
