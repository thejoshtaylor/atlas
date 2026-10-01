import Foundation

/// Lets at most 20 partial updates through each second (research pitfall 17).
/// The newest partial wins: one that arrives inside the window is held, a later
/// one replaces it, and `flush` hands out the winner when the window ends.
/// The clock is an argument, so a test controls it.
public struct PartialCoalescer: Sendable, Equatable {
    // Dates are doubles, and a date near 2026 has a granularity of about 2e-7
    // seconds. This margin makes a call exactly at `flushAt` count as due.
    private static let tolerance: TimeInterval = 0.000_001

    private var lastDelivery: Date?
    private var held: TranscriptPartial?

    public init() {}

    /// The partial to apply now, or `nil` when it is held for `flush`.
    public mutating func offer(_ partial: TranscriptPartial, now: Date) -> TranscriptPartial? {
        if isDue(now) {
            held = nil
            lastDelivery = now
            return partial
        }
        held = partial
        return nil
    }

    /// The held partial once its window has ended, else `nil`.
    public mutating func flush(now: Date) -> TranscriptPartial? {
        guard let partial = held, isDue(now) else { return nil }
        held = nil
        lastDelivery = now
        return partial
    }

    /// When a held partial is due. `nil` with nothing held.
    public var flushAt: Date? {
        guard held != nil, let last = lastDelivery else { return nil }
        return last.addingTimeInterval(PanelTiming.partialMinIntervalS)
    }

    /// Forget the held partial and the last delivery. A new turn starts clean.
    public mutating func reset() {
        held = nil
        lastDelivery = nil
    }

    private func isDue(_ now: Date) -> Bool {
        guard let last = lastDelivery else { return true }
        return now.timeIntervalSince(last) >= PanelTiming.partialMinIntervalS - Self.tolerance
    }
}
