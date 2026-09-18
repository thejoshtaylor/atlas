import { describe, expect, test } from "bun:test"
import { readFileSync } from "node:fs"
import { join } from "node:path"

const SOURCE = readFileSync(join(import.meta.dir, "SetupGuard.tsx"), "utf-8")

describe("SetupGuard -- the terminal gate, and the wizard's own path never blocked by it", () => {
  test("reads the dedicated, always-answering setup-status query, not an incidental 401/503 on another route", () => {
    expect(SOURCE).toMatch(/setupStatusQueryOptions/)
  })

  test("the terminal gate keys on admin_account specifically, not full wizard completion -- WEB-02's never-a-lockout", () => {
    expect(SOURCE).toMatch(/steps\.find\(\(step\) => step\.name === "admin_account"\)/)
  })

  test("/setup and /sign-in are exempt from the terminal gate -- both must be reachable before an admin exists", () => {
    expect(SOURCE).toMatch(/pathname\.startsWith\("\/setup"\)/)
    expect(SOURCE).toMatch(/pathname\.startsWith\("\/sign-in"\)/)
  })
})
