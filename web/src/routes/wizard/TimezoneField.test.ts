import { afterEach, describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"
import { queryClient } from "../../lib/queryClient"
import { initialTimezoneValue, setTimezoneMutationOptions, type TimezoneStatus } from "../../lib/wizard"

// `web/` has no rendered-DOM test infrastructure -- this file asserts the
// source text carries the wiring the plan's own behavior list requires
// (the same source-regex convention `AudioSourceStep.test.ts` and
// `HubStep.test.ts` already use), plus a lib part exercising the pure
// `initialTimezoneValue` function and the mutation's own request shape
// with `global.fetch` mocked, in the style of `src/lib/wizard.test.ts`.
const FIELD_SOURCE = readFileSync(join(import.meta.dir, "TimezoneField.tsx"), "utf-8")
const HUB_SOURCE = readFileSync(join(import.meta.dir, "HubStep.tsx"), "utf-8")
const SETTINGS_SOURCE = readFileSync(
  join(import.meta.dir, "..", "settings", "SettingsRoute.tsx"),
  "utf-8",
)

describe("TimezoneField -- the warning, badge, save path, and zone choices come from the server/lib, never re-derived", () => {
  test("the warning renders verbatim from the server's own field, with role=\"alert\"", () => {
    expect(FIELD_SOURCE).toMatch(/\.warning/)
    expect(FIELD_SOURCE).toMatch(/role="alert"/)
  })

  test("the badge reads applies_live, never a hardcoded value", () => {
    expect(FIELD_SOURCE).toMatch(/applies_live \? "Live" : "Needs restart"/)
  })

  test("the save goes through setTimezoneMutationOptions, no second write path", () => {
    expect(FIELD_SOURCE).toMatch(/setTimezoneMutationOptions/)
  })

  test("the choices come from Intl.supportedValuesOf(\"timeZone\") in a datalist", () => {
    expect(FIELD_SOURCE).toMatch(/Intl\.supportedValuesOf\("timeZone"\)/)
    expect(FIELD_SOURCE).toMatch(/<datalist/)
  })

  test("the browser zone comes from Intl.DateTimeFormat().resolvedOptions().timeZone", () => {
    expect(FIELD_SOURCE).toMatch(/Intl\.DateTimeFormat\(\)\.resolvedOptions\(\)\.timeZone/)
  })
})

describe("HubStep and SettingsRoute both render the shared TimezoneField, with no props", () => {
  test("HubStep renders <TimezoneField", () => {
    expect(HUB_SOURCE).toMatch(/<TimezoneField\b/)
  })

  test("SettingsRoute renders <TimezoneField", () => {
    expect(SETTINGS_SOURCE).toMatch(/<TimezoneField\b/)
  })
})

describe("initialTimezoneValue -- the field's starting value, one branch per source", () => {
  const base: TimezoneStatus = {
    zone: "Europe/Berlin",
    resolved_from: "home_assistant",
    stored: null,
    warning: null,
    applies_live: false,
  }

  test("a stored zone wins, regardless of resolved_from", () => {
    expect(initialTimezoneValue({ ...base, stored: "Asia/Tokyo" }, "America/Chicago")).toBe("Asia/Tokyo")
  })

  test("with nothing stored, a non-process zone (config or Home Assistant) is used", () => {
    expect(initialTimezoneValue(base, "America/Chicago")).toBe("Europe/Berlin")
  })

  test("with nothing stored and a process fallback, the browser's own zone is offered", () => {
    const processStatus: TimezoneStatus = { ...base, resolved_from: "process", zone: "the local zone" }
    expect(initialTimezoneValue(processStatus, "America/Chicago")).toBe("America/Chicago")
  })
})

const originalFetch = global.fetch
afterEach(() => {
  global.fetch = originalFetch
  queryClient.clear()
})

describe("setTimezoneMutationOptions -- PUT /api/wizard/timezone", () => {
  test("mutationFn sends the zone as the request body", async () => {
    let calledPath: string | undefined
    let calledMethod: string | undefined
    let sentBody: unknown
    global.fetch = (async (url: string, init?: RequestInit) => {
      calledPath = url
      calledMethod = init?.method
      sentBody = init?.body ? JSON.parse(init.body as string) : undefined
      return new Response(
        JSON.stringify({
          zone: "America/New_York",
          resolved_from: "database",
          stored: "America/New_York",
          warning: null,
          applies_live: false,
        }),
        { status: 200, headers: { "Content-Type": "application/json" } },
      )
    }) as typeof fetch

    await setTimezoneMutationOptions.mutationFn!({ zone: "America/New_York" }, {} as never)
    expect(calledPath).toBe("/api/wizard/timezone")
    expect(calledMethod).toBe("PUT")
    expect(sentBody).toEqual({ zone: "America/New_York" })
  })
})
