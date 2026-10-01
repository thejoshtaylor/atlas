import Foundation

/// The id of the timer that rings now, for the menu bar item "Stop Ringing"
/// (UI-SPEC "Keyboard and accessibility").
///
/// It follows the server frames, not the panel. The operator can close the
/// panel while the ring keeps sounding, and the menu item must stay.
public struct RingTracker: Sendable, Equatable {
    public private(set) var timerId: Int?

    public init() {}

    /// What "Stop Ringing" does.
    public enum StopAction: Sendable, Equatable {
        /// The panel shows this ring with an idle Stop button. Click it, so the
        /// reducer sends the stop and shows "Stopping".
        case clickInPanel
        /// The panel is closed, or its button already waits. Send the stop at once.
        case send(timerId: Int)
    }

    /// `timer.ringing` sets the id. `timer.stopped` for the same id and a lost
    /// connection clear it. After a reconnect the server sends the ring again.
    public mutating func apply(_ event: PanelEvent) {
        switch event {
        case .timerRinging(let message):
            timerId = message.timerId
        case .timerStopped(let message):
            if message.timerId == timerId { timerId = nil }
        case .connectionLost:
            timerId = nil
        default:
            break
        }
    }

    /// `nil` when no timer rings.
    public func stopAction(panelRing: PanelRing?) -> StopAction? {
        guard let timerId else { return nil }
        if let panelRing, panelRing.timerId == timerId, panelRing.stop == .idle { return .clickInPanel }
        return .send(timerId: timerId)
    }
}
