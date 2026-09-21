import * as React from "react"
import { Menu, X } from "lucide-react"
import { NavLink, Outlet } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { useSession } from "@/hooks/useSession"
import { cn } from "@/lib/utils"

const NAV_ITEMS = [
  { to: "/", label: "Home", minimumRole: "viewer" },
  { to: "/policy", label: "Safety policy", minimumRole: "operator" },
  { to: "/macros", label: "Macros", minimumRole: "operator" },
  { to: "/workflows", label: "Scheduled", minimumRole: "operator" },
  { to: "/accounts", label: "Accounts", minimumRole: "admin" },
  { to: "/settings", label: "Settings", minimumRole: "admin" },
  { to: "/plugins", label: "Plugins", minimumRole: "admin" },
  { to: "/providers", label: "Providers", minimumRole: "admin" },
  // The developer microphone page (plan 03-10, D-19) -- a genuine nav
  // entry, not reachable by direct URL alone the way /calibration is
  // (that page is a wizard step first; this one is a standalone
  // diagnostic tool an operator navigates to on purpose).
  { to: "/dev-mic", label: "Developer mic", minimumRole: "operator" },
  // Phase 8, WEB-06/D-05: the live observer feed.
  { to: "/live", label: "Live", minimumRole: "operator" },
  // Phase 8, WEB-07/D-01: past recorded turns. `/sessions/:id` has no nav
  // entry of its own -- reached by a link from this list, matching how
  // `/plugins/:id` has no separate nav entry today.
  { to: "/sessions", label: "Sessions", minimumRole: "operator" },
] as const

const ROLE_RANK: Record<string, number> = { viewer: 0, operator: 1, admin: 2 }

/**
 * Phone-first from the first commit (WEB-08, D-17): single column at
 * narrow width, navigation collapsed into the header rather than a
 * permanent sidebar, and it widens from here rather than the reverse.
 * The screens later plans mount render through `<Outlet />`.
 */
export function AppShell() {
  const [navOpen, setNavOpen] = React.useState(false)
  const session = useSession()
  // Presentation only, T-03-47: a viewer simply never sees a link to a
  // screen their role cannot use. This holds no line by itself -- the
  // route itself refuses a role that cannot use it
  // (`RequireRole`/`require_role`), regardless of what this nav shows or
  // hides. A hidden link is convenience, not a permission boundary.
  const role = session.data?.role ?? "viewer"
  const visibleNavItems = NAV_ITEMS.filter((item) => ROLE_RANK[role] >= ROLE_RANK[item.minimumRole])

  return (
    <div className="flex min-h-svh flex-col bg-background text-foreground">
      <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-background px-4 py-3">
        <span className="text-heading font-semibold">spire-voice</span>
        <Button
          type="button"
          variant="ghost"
          size="icon"
          aria-label={navOpen ? "Close navigation" : "Open navigation"}
          aria-expanded={navOpen}
          onClick={() => setNavOpen((open) => !open)}
        >
          {navOpen ? <X className="size-5" /> : <Menu className="size-5" />}
        </Button>
      </header>

      {navOpen ? (
        <nav aria-label="Primary" className="border-b border-border">
          <ul className="flex flex-col">
            {visibleNavItems.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.to === "/"}
                  onClick={() => setNavOpen(false)}
                  className={({ isActive }) =>
                    cn(
                      "touch-target flex items-center px-4 text-body",
                      // Accent (10%) is reserved for, among other things,
                      // "the active nav item" -- 03-UI-SPEC.md's Color
                      // section. Nothing else in this list uses it.
                      isActive ? "font-semibold text-primary" : "text-foreground",
                    )
                  }
                >
                  {item.label}
                </NavLink>
              </li>
            ))}
          </ul>
          <Separator />
        </nav>
      ) : null}

      <main className="flex-1 p-4">
        <Outlet />
      </main>
    </div>
  )
}
