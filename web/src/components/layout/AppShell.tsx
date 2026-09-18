import * as React from "react"
import { Menu, X } from "lucide-react"
import { NavLink, Outlet } from "react-router-dom"
import { Button } from "@/components/ui/button"
import { Separator } from "@/components/ui/separator"
import { cn } from "@/lib/utils"

const NAV_ITEMS = [
  { to: "/", label: "Home" },
  { to: "/policy", label: "Safety policy" },
  { to: "/accounts", label: "Accounts" },
  { to: "/settings", label: "Settings" },
] as const

/**
 * Phone-first from the first commit (WEB-08, D-17): single column at
 * narrow width, navigation collapsed into the header rather than a
 * permanent sidebar, and it widens from here rather than the reverse.
 * The screens later plans mount render through `<Outlet />`.
 */
export function AppShell() {
  const [navOpen, setNavOpen] = React.useState(false)

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
            {NAV_ITEMS.map((item) => (
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
