# Design

The ATLAS admin webapp is a calm celestial instrument. A living meridian globe is the assistant's presence. Every screen around the globe is quiet: ink surfaces, hairline rules, porcelain text, and one gold accent.

Source of truth: `web/src/index.css` (tokens and utilities) and `web/src/components/brand/AtlasGlobe.tsx` (the mark).

## Theme

Dark only. The operator uses the app at night beside a camera, or at a desk. Toasts are pinned to dark and do not follow the OS.

| Role | Token | Value | Use |
|---|---|---|---|
| Ground | `--background` | `oklch(0.165 0.022 258)` | Page. Blue-black ink, never pure black. |
| Surface | `--card` | `oklch(0.2 0.024 258)` | Panels, list rows, sidebar (at 40% opacity). |
| Raised | `--secondary` / `--muted` | `oklch(0.25 0.024 258)` | Secondary buttons, skeletons, avatar. |
| Rule | `--border` | `oklch(0.3 0.022 258)` | Hairlines between rows and around panels. |
| Text | `--foreground` | `oklch(0.95 0.008 85)` | Porcelain. |
| Quiet text | `--muted-foreground` | `oklch(0.74 0.02 250)` | Descriptions, readout labels. |
| Accent | `--primary` / `--ring` | `oklch(0.84 0.12 84)` | Ecliptic gold. Primary actions, the active nav item, focus, links, the globe's ecliptic ring, histogram bars that clear. Nothing else. |
| Denied | `--denied-bg` / `--denied-fg` | lilac | "Denied for control" badges. A calm, safe state. Not red. |
| Destructive | `--destructive` | `oklch(0.58 0.19 25)` | Confirmations for actions you cannot undo. |

Do not use glow halos, neon, gradient text, or glass.

## Type

- **Manrope Variable** (`--font-sans`) for all UI text. Self-hosted through `@fontsource-variable/manrope`.
- **JetBrains Mono Variable** (`--font-mono`) only through the `readout` utility. Use it for measured values: ms, timestamps, gain, entity IDs, transports. Do not use it for prose.
- There are four sizes: `text-label` (14px), `text-body` (16px), `text-heading` (20px), and `text-display` (28px). In Tailwind v4 these must be `--text-*` tokens. The earlier `--font-size-*` names generated no utilities, so every heading rendered at body size.
- `caps-label`: 11px, tracked 0.14em, uppercase. Use it only for sidebar group names and the role under the user's name.
- Headings use balanced wrapping and tracking of -0.02em.

## Layout

- **Shell** (`AppShell.tsx`): from `lg` up, a 16rem sidebar with the globe and wordmark. Navigation is grouped under Observe, Configure, and Diagnose, and the user's identity sits at the bottom. Below `lg`, a sticky header holds a toggle that opens a drawer with the same list. The content area is at most `max-w-5xl`.
- **Lists** use the `panel-list` utility on the `<ul>`. It draws one bordered panel with hairline-divided rows. Rows are `px-4 py-3`. Do not stack separate cards.
- **Primary actions** (`SubmitButton`) span the full width on a phone. From `sm` up they take their own width and sit at the start of the form.
- **Touch targets:** the 44px minimum (`touch-target`, `h-11`) applies everywhere. Only desktop sidebar rows drop to 36px (`lg:min-h-9`).

## The globe

`<AtlasGlobe spinning? label? />` is a wireframe sphere tilted -18°. It has five parallels and six meridians, and a gold ecliptic ring tilted 8° across it. When `spinning` is set, each meridian's x-scale runs through cos(t) (easeInOutSine, 12s per turn, staggered by phase). The ecliptic breathes between 100% and 55% opacity. Both stop under `prefers-reduced-motion`.

Where the globe appears:

- Sidebar and header wordmark: static, 28px.
- Sign-in page and wizard step header: spinning.
- Home: large and spinning.
- Live idle state: spinning while the observer socket is connected, static while it is not.
- Listen dial: see below.

## The listen dial

`ListenDial` (`routes/listen/`) puts the globe inside a graduated ring of 120 ticks, with a longer tick every tenth. Two waveforms pulse round the ring, each drawn from its own live spectrum and mirrored left to right:

- **Gold** (`--primary`) is incoming audio: the microphone, while the server reads it (armed at half reach, hearing at full reach).
- **Voice blue** (`--voice`, `oklch(0.78 0.1 230)`) is outgoing audio: ATLAS's reply and the wake chime playing on this device. This token is used only on the dial and the shell's listening dot.

| State | Dial | Globe |
|---|---|---|
| Off | Porcelain ticks, faint, at rest | Static |
| Armed | Gold pulse at half reach, following the room | Spinning |
| Hearing | Gold pulse at full reach, following the voice | Spinning, ecliptic held at full strength |
| Thinking | No audio. A porcelain sweep runs round the dial once every 1.4 s | Spinning four times faster |
| Speaking | Blue pulse, following the reply audio | Spinning |

Under `prefers-reduced-motion`, the sweep becomes a steady raised porcelain ring, and the pulses still follow the audio level but settle slower.

While the listener is on, the shell shows it on every page: a dot (gold, or voice blue while replying) and the state word, in the sidebar above the user's identity and in the phone header. The dot breathes slowly and stops under reduced motion.

`web/public/favicon.svg` is the static mark on an ink tile.

## Browser surfaces

- Selection is gold at 35% opacity.
- Scrollbars are thin and use the input color.
- Range inputs use the gold `accent-color`.
- The page declares `color-scheme: dark`, so native controls render dark.
