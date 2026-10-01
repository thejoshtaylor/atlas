import Foundation

/// Time for the connection actor. Tests inject a clock they advance by hand.
public protocol ConnectionClock: Sendable {
    func now() -> Date
    /// Throws `CancellationError` when the calling task is cancelled.
    func sleep(for seconds: TimeInterval) async throws
}

public struct SystemConnectionClock: ConnectionClock {
    public init() {}

    public func now() -> Date { Date() }

    public func sleep(for seconds: TimeInterval) async throws {
        try await Task.sleep(for: .seconds(max(0, seconds)))
    }
}
