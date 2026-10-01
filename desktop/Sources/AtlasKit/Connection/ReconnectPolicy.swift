import Foundation

/// How long the Mac waits before its next dial (D-05).
///
/// The delay for attempt `n` is `min(cap, initial * factor^n)`, spread by a
/// random factor of plus or minus `jitter`. The attempt count goes back to 0
/// only after a connection that reached `hello.ack` stayed up for `stableAfter`
/// seconds. The edge Pi uses the same shape in `atlas_edge/client.py`.
public struct ReconnectPolicy: Sendable, Equatable {
    public var initial: TimeInterval = 1.0
    public var factor: Double = 2.0
    public var cap: TimeInterval = 60.0
    public var jitter: Double = 0.2
    public var stableAfter: TimeInterval = 30.0
    /// The wait after a wake or a returning network path. An attempt inside the
    /// wake callback often fails because the network is not up yet.
    public var wakeDelay: TimeInterval = 1.0

    public init() {}

    /// `random` is a value from 0 up to (not including) 1. 0.5 means no jitter.
    public func delay(attempt: Int, random: Double) -> TimeInterval {
        let steps = Double(max(0, attempt))
        let base = min(cap, initial * pow(factor, steps))
        let unit = min(max(random, 0), 1.0.nextDown)
        return base * (1 + jitter * (2 * unit - 1))
    }
}
