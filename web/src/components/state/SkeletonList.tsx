import { Skeleton } from "@/components/ui/skeleton"

export interface SkeletonListProps {
  rows?: number
}

/**
 * The loading state for every card list in this application (the policy
 * list, the accounts list). A list that is loading must never render as
 * visually empty -- an empty-looking flash on a safety-boundary list is
 * a defect, not a loading nicety (03-UI-SPEC.md's UI Considerations
 * table) -- so this renders placeholder rows rather than nothing while
 * the real data is in flight.
 */
export function SkeletonList({ rows = 3 }: SkeletonListProps) {
  return (
    <div className="flex flex-col gap-2" role="status" aria-label="Loading">
      {Array.from({ length: rows }, (_, index) => (
        <Skeleton key={index} className="h-16 w-full rounded-lg" />
      ))}
    </div>
  )
}
