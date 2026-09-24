import * as React from "react"
import {
  Blocks,
  CalendarClock,
  Cpu,
  Gauge,
  History,
  House,
  Menu,
  Mic,
  Radio,
  Settings,
  ShieldCheck,
  Users,
  Workflow,
  X,
  type LucideIcon,
} from "lucide-react"
import { NavLink, Outlet } from "react-router-dom"
import { AtlasGlobe } from "@/components/brand/AtlasGlobe"
import { Button } from "@/components/ui/button"
import { useSession } from "@/hooks/useSession"
import { cn } from "@/lib/utils"

type Role = "viewer" | "operator" | "admin"

interface NavItem {
  to: string
  label: string
  minimumRole: Role
  icon: LucideIcon
}

// Grouped by what the operator is doing: watching the assistant,
// configuring what it may do, and diagnosing how it hears.
const NAV_GROUPS: { title: string; items: NavItem[] }[] = [
  {
    title: "Observe",
    items: [
      { to: "/", label: "Home", minimumRole: "viewer", icon: House },
      // Phase 8, WEB-06/D-05: the live observer feed.
      { to: "/live", label: "Live", minimumRole: "operator", icon: Radio },
      // Phase 8, WEB-07/D-01: past recorded turns. `/sessions/:id` is
      // reached by a link from this list, never from the nav.
      { to: "/sessions", label: "Sessions", minimumRole: "operator", icon: History },
    ],
  },
  {
    title: "Configure",
    items: [
      { to: "/policy", label: "Safety policy", minimumRole: "operator", icon: ShieldCheck },
      { to: "/macros", label: "Macros", minimumRole: "operator", icon: Workflow },
      { to: "/workflows", label: "Scheduled", minimumRole: "operator", icon: CalendarClock },
      { to: "/plugins", label: "Plugins", minimumRole: "admin", icon: Blocks },
      { to: "/providers", label: "Providers", minimumRole: "admin", icon: Cpu },
      { to: "/accounts", label: "Accounts", minimumRole: "admin", icon: Users },
      { to: "/settings", label: "Settings", minimumRole: "admin", icon: Settings },
    ],
  },
  {
    title: "Diagnose",
    items: [
      // The developer microphone page (plan 03-10, D-19) -- a standalone
      // diagnostic tool an operator navigates to on purpose.
      { to: "/dev-mic", label: "Developer mic", minimumRole: "operator", icon: Mic },
      // Phase 8, DBG-05: the wake-threshold tuning screen.
      { to: "/wake-tuning", label: "Wake threshold", minimumRole: "operator", icon: Gauge },
    ],
  },
]

const ROLE_RANK: Record<string, number> = { viewer: 0, operator: 1, admin: 2 }

function Wordmark() {
  return (
    <span className="flex items-center gap-2.5">
      <AtlasGlobe className="size-7 text-foreground" />
      <span className="text-body font-semibold tracking-[0.28em]">ATLAS</span>
    </span>
  )
}

function NavList({ role, onNavigate }: { role: string; onNavigate?: () => void }) {
  return (
    <div className="flex flex-col gap-5 lg:gap-4">
      {NAV_GROUPS.map((group) => {
        const items = group.items.filter((item) => ROLE_RANK[role] >= ROLE_RANK[item.minimumRole])
        if (items.length === 0) return null
        return (
          <div key={group.title} className="flex flex-col gap-0.5">
            <p aria-hidden className="caps-label px-3 pb-1.5 text-muted-foreground/80">
              {group.title}
            </p>
            <ul className="flex flex-col gap-0.5">
              {items.map((item) => (
                <li key={item.to}>
                  <NavLink
                    to={item.to}
                    end={item.to === "/"}
                    onClick={onNavigate}
                    className={({ isActive }) =>
                      cn(
                        "touch-target flex items-center gap-3 rounded-md px-3 text-label transition-colors lg:min-h-9",
                        // Accent is reserved for, among other things, the
                        // active nav item. Nothing else in this list uses it.
                        isActive
                          ? "bg-primary/10 font-semibold text-primary"
                          : "text-muted-foreground hover:bg-accent hover:text-foreground",
                      )
                    }
                  >
                    <item.icon className="size-4 shrink-0" aria-hidden />
                    {item.label}
                  </NavLink>
                </li>
              ))}
            </ul>
          </div>
        )
      })}
    </div>
  )
}

function Identity({ name, role }: { name?: string; role: string }) {
  if (!name) return null
  return (
    <div className="flex items-center gap-3 border-t border-border px-3 pt-4">
      <span className="flex size-8 items-center justify-center rounded-full bg-secondary text-label font-semibold">
        {name.charAt(0).toUpperCase()}
      </span>
      <span className="flex min-w-0 flex-col">
        <span className="truncate text-label text-foreground">{name}</span>
        <span className="caps-label text-muted-foreground">{role}</span>
      </span>
    </div>
  )
}

/**
 * Phone first (WEB-08, D-17): at narrow widths navigation collapses into
 * a header toggle. From `lg` up the same list is a permanent sidebar.
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
  const name = session.data?.display_name || session.data?.email

  return (
    <div className="min-h-svh bg-background text-foreground lg:grid lg:grid-cols-[16rem_1fr]">
      <aside className="sticky top-0 hidden h-svh flex-col gap-6 overflow-y-auto [scrollbar-width:thin] border-r border-border bg-card/40 px-3 py-5 lg:flex">
        <div className="px-3">
          <Wordmark />
        </div>
        <nav aria-label="Main" className="flex-1">
          <NavList role={role} />
        </nav>
        <Identity name={name} role={role} />
      </aside>

      <div className="flex min-w-0 flex-col">
        <header className="sticky top-0 z-10 flex items-center justify-between border-b border-border bg-background/95 px-4 py-2 backdrop-blur lg:hidden">
          <Wordmark />
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
          <nav aria-label="Primary" className="border-b border-border bg-card/60 px-3 py-4 lg:hidden">
            <NavList role={role} onNavigate={() => setNavOpen(false)} />
          </nav>
        ) : null}

        <main className="w-full max-w-5xl flex-1 px-4 py-6 sm:px-8 lg:px-12 lg:py-10">
          <Outlet />
        </main>
      </div>
    </div>
  )
}

