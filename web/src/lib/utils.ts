// shadcn's cn() utility. The current shadcn CLI (4.21.0) generates
// components that import `cn` directly from the official `cn` npm package
// (github.com/shadcn-ui/cn -- a drop-in clsx+tailwind-merge replacement
// published by the shadcn team itself), rather than writing a local
// implementation here the way older shadcn versions did. This file exists
// so `@/lib/utils` -- the import path 03-UI-SPEC.md and this plan's own
// artifact list expect -- still resolves, and so hand-written code in this
// project has one canonical import path regardless of which convention a
// given shadcn CLI version happens to generate against.
export { cn } from "cn"
