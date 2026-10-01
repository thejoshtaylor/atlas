import Foundation

/// Every timing value and cap of the panel (Phase 15). The values come from the
/// UI-SPEC "Interaction and Timing Contract" table and from the discretion items
/// of 15-CONTEXT. The reducer reads them here so a test can name each one, and no
/// other file holds a number of its own.
public enum PanelTiming {
    /// Hide delay after `turn.ended` when no follow-up window is open (D-06).
    public static let hideAfterTurnS: TimeInterval = 4
    /// Hide delay after `timer.stopped` when no turn is active (D-16).
    public static let hideAfterStopS: TimeInterval = 4
    /// Silence that hides an open turn panel (D-08).
    public static let watchdogS: TimeInterval = 30
    /// Backstop for a lost `timer.stopped`. The server ring stops at 120 s.
    public static let ringCapS: TimeInterval = 130
    /// A Stop click with no `timer.stopped` after this long is enabled again.
    public static let stopRetryS: TimeInterval = 3
    /// The least time a hide timer or watchdog has left after a hover ends (D-04).
    public static let hoverResumeMinS: TimeInterval = 2
    /// 20 partial updates each second (research pitfall 17).
    public static let partialMinIntervalS: TimeInterval = 0.05
    /// Fade durations the shell uses. Reduce Motion turns both off.
    public static let fadeInS: TimeInterval = 0.12
    public static let fadeOutS: TimeInterval = 0.2
    /// The Mac does not trust server durations: each one is clamped to this.
    public static let maxServerWindowS: TimeInterval = 120
    /// The transcript keeps its newest characters, counted as unicode scalars.
    public static let transcriptKeepScalars = 500
    public static let cardMaxScalars = 600
    public static let labelMaxScalars = 120
    /// How many replaced, hidden or closed turn ids the panel remembers.
    public static let retiredTurnMemory = 8
}
