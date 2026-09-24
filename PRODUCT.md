# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Users

- **Operator/admin of a home install.** Usually the person who deployed ATLAS. They configure providers, plugins, the safety policy, macros, scheduled workflows, and accounts. They also debug: live turn feed, recorded sessions, wake-threshold tuning, calibration.
- **Viewer/operator household members.** Fewer screens, same shell.
- Used on desktop and phone about equally. A phone is often in hand next to the camera while testing a spoken command.

## Product Purpose

ATLAS is a self-hosted home voice assistant. A local wake word starts it; speech, reasoning, and speech synthesis run through operator-chosen providers (cloud or local); every capability is an MCP server. The admin webapp configures and observes all of it. Success: a spoken sentence makes the house do the right thing fast enough to feel like a reply, and the assistant never touches what the operator marked off limits.

## Positioning

Self-hosted, provider-agnostic, safety-policy-first. The operator sees measured latency and every recorded turn, not a black box.

## Operating Context

- Audio source is an 8 kHz narrowband camera mic; latency figures (ms end to end) are first-class data the UI shows.
- Deployed via Helm or Docker Compose; set up in the browser via a first-run wizard.
- Public repo: strangers deploy it; no house-specific data in UI defaults.

## Capabilities and Constraints

- Roles: viewer < operator < admin; nav hides what a role cannot use (convenience only).
- Screens: sign-in, invite, setup wizard, home, safety policy, macros (+editor), scheduled workflows (+editor), accounts, settings, plugins (+editor), providers, developer mic, live feed, sessions (+detail), wake threshold, calibration.
- Stack: React 19, Vite, Tailwind v4, shadcn/Radix components, TanStack Query, bun.
- All transcribed text is untrusted input.

## Brand Commitments

- Name: **ATLAS** (Assistant for Tasks, Logistics, Automation, and Scheduling). Wake phrase "hey atlas".
- Personality (user-confirmed 2026-09-23): futuristic, in the "sci-fi AI companion" sense — calm, present, restrained. Not gamer-RGB, no neon/glow overload.
- Logo direction (user-confirmed): an atlas globe / meridian mark.

## Evidence on Hand

Real data in the live install: recorded sessions with transcripts and end-to-end ms, wake attempts, provider latency measurements, denylist entries. No testimonials or marketing claims exist; none should be invented.

## Product Principles

1. The house's state and the assistant's behaviour are visible, never hidden.
2. Safety policy reads as calm and legible, not alarming.
3. Measured numbers beat adjectives.
4. Works the same on a phone beside the camera as on a desk.

## Accessibility & Inclusion

WCAG AA contrast; 44px minimum touch targets (existing `touch-target` utility) kept everywhere.
