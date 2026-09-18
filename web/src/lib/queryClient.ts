import { QueryClient } from "@tanstack/react-query"

// This is a configuration console, not a feed -- refetch-on-window-focus
// would re-run every visible query each time the operator alt-tabs back
// to the tab, for no benefit. Mutation retries are off for a stronger
// reason than convenience: a credential write or an invite send is not
// safe to silently repeat (T-03-14), so a failed mutation must surface as
// a failure the caller decides how to handle, never a background retry
// the operator never sees.
export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      refetchOnWindowFocus: false,
      retry: false,
    },
    mutations: {
      retry: false,
    },
  },
})
