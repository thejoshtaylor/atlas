import Foundation

/// The id of the timer that rings now, for the menu bar item "Stop Ringing"
/// (UI-SPEC "Keyboard and accessibility").
///
/// It follows the server frames, not the panel. The operator can close the
/// panel while the ring keeps sounding, and the menu item must stay. It has its
/// own cap, the same as the panel's ring view: a lost `timer.stopped` would
/// otherwise leave the item in the menu until the next reconnect.
public struct RingTracker: Sendable, Equatable {
    public private(set) var timerId: Int?
    /// When the shell must call `expire`. `nil` when no timer rings.
    public private(set) var capAt: Date?

    public init() {}

    /// What "Stop Ringing" does.
    public enum StopAction: Sendable, Equatable {
        /// The panel shows this ring with an idle Stop button. Click it, so the
        /// reducer sends the stop and shows "Stopping".
        case clickInPanel
        /// The panel is closed, or its button already waits. Send the stop at once.
        case send(timerId: Int)
    }

    /// `timer.ringing` sets the id and starts the cap. The same id again (a
    /// replay) keeps the first cap, as the panel's ring view does. `timer.stopped`
    /// for the same id and a lost connection clear it. After a reconnect the
    /// server sends the ring again.
    public mutating func apply(_ event: PanelEvent, now: Date = Date()) {
        switch event {
        case .timerRinging(let message):
            if message.timerId != timerId || capAt == nil {
                capAt = now.addingTimeInterval(PanelTiming.ringCapS)
            }
            timerId = message.timerId
        case .timerStopped(let message):
            if message.timerId == timerId { clear() }
        case .connectionLost:
            clear()
        default:
            break
        }
    }

    /// The cap passed with no `timer.stopped`: the ring is over.
    public mutating func expire(now: Date = Date()) {
        guard let capAt, capAt <= now else { return }
        clear()
    }

    private mutating func clear() {
        timerId = nil
        capAt = nil
    }

    /// `nil` when no timer rings.
    public func stopAction(panelRing: PanelRing?) -> StopAction? {
        guard let timerId else { return nil }
        if let panelRing, panelRing.timerId == timerId, panelRing.stop == .idle { return .clickInPanel }
        return .send(timerId: timerId)
    }
}
