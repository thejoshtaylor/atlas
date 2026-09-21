import { GlobalRegistrator } from "@happy-dom/global-registrator"

GlobalRegistrator.register()

// React 19's `act(...)` (used by `LiveRoute.test.tsx` to flush a state
// update that fires outside any React-tracked event, the same way a real
// `WebSocket.onmessage` handler would) checks this global before running --
// unset, it warns on every call even though the flush itself works
// correctly. Set once, here, for the whole suite, matching React's own
// testing documentation for a custom DOM environment.
;(globalThis as { IS_REACT_ACT_ENVIRONMENT?: boolean }).IS_REACT_ACT_ENVIRONMENT = true
