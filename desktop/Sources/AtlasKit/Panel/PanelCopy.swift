import Foundation

/// Every fixed string and symbol name of the panel (UI-SPEC "Copywriting
/// Contract"). The text is written in code and never comes from the model
/// (D-09). New copy uses no contractions.
public enum PanelCopy {
    public static let panelLabel = "ATLAS"
    public static let listeningAnnouncement = "ATLAS is listening"
    public static let emptyHint = "Go ahead"
    public static let closeLabel = "Close panel"
    public static let stopTitle = "Stop"
    /// "Stopping" and U+2026.
    public static let stopPendingTitle = "Stopping\u{2026}"
    public static let stoppedLine = "Stopped"
    /// The keyboard path: a menu bar item that shows only while a timer rings.
    public static let stopRingingMenuItem = "Stop Ringing"
    public static let emptyTimerLabel = "Time is up"
    public static let emptyAlarmLabel = "Alarm is ringing"
    public static let bellSymbol = "bell.fill"
    public static let closeSymbol = "xmark"

    public static func stateWord(_ word: PanelStateWord) -> String {
        switch word {
        case .listening: "Listening"
        case .thinking: "Thinking"
        case .speaking: "Speaking"
        case .done: "Done"
        }
    }

    public static func stateSymbol(_ word: PanelStateWord) -> String {
        switch word {
        case .listening: "waveform"
        case .thinking: "ellipsis"
        case .speaking: "speaker.wave.2.fill"
        case .done: "checkmark.circle.fill"
        }
    }

    public static func outcomeLine(_ outcome: PanelOutcomeLine) -> String {
        switch outcome {
        case .noSpeech: "Did not catch that"
        case .stopped: "Stopped"
        case .failed: "Could not answer. Try again."
        }
    }

    public static func ringKindWord(_ kind: RingKind) -> String {
        switch kind {
        case .timer: "Timer"
        case .alarm: "Alarm"
        }
    }

    /// What the ring view shows when the sanitized label is empty.
    public static func emptyRingLabel(_ kind: RingKind) -> String {
        switch kind {
        case .timer: emptyTimerLabel
        case .alarm: emptyAlarmLabel
        }
    }

    /// VoiceOver label of the Stop button.
    public static func stopAccessibilityLabel(label: String, kind: RingKind) -> String {
        if label.isEmpty { return "Stop \(ringKindWord(kind).lowercased())" }
        return "Stop \(label)"
    }

    /// The announcement when a ring starts. An empty label reads as the empty-label text.
    public static func ringAnnouncement(kind: RingKind, label: String) -> String {
        "\(ringKindWord(kind)) ringing, \(label.isEmpty ? emptyRingLabel(kind) : label)"
    }
}
