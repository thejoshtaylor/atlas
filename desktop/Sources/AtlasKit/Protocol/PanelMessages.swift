import Foundation

// The panel frames of the `/ws/desktop` protocol (Phase 15, D-10, D-11). They
// mirror the Python models in `src/atlas/desktop/protocol.py` and
// `src/atlas/desktop/cards.py`, and `FixtureDecodingTests` reads the same
// fixture files that pytest reads.
//
// A word field (a state, an outcome, a card kind, a timer kind) accepts any
// word of 1 to 32 characters. A newer server can add a word, and the Mac must
// keep working. The Python models are stricter on purpose, because the server
// only sends words it knows.

/// Caps shared by the panel frames. They match the Python models and
/// `desktop/protocol/v1/panel_limits.json`.
public enum PanelLimits {
    public static let idMax = 64
    public static let wordMax = 32
    public static let transcriptTextMax = 1000
    public static let cardTextMax = 600
    public static let labelMax = 120
    public static let msMax = 600_000
    public static let timerIdMax = 2_147_483_647
}

/// A string of 0...max unicode scalars. Python counts code points.
func boundedText<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, _ key: K, max: Int
) throws -> String {
    let value = try container.decode(String.self, forKey: key)
    guard value.unicodeScalars.count <= max else {
        throw DecodingError.dataCorruptedError(
            forKey: key, in: container, debugDescription: "length must be at most \(max)")
    }
    return value
}

/// A strict integer (no bool, no string, no fraction) inside a closed range.
func boundedInt<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, _ key: K, in range: ClosedRange<Int>
) throws -> Int {
    let value = try strictInt(container, key)
    guard range.contains(value) else {
        throw DecodingError.dataCorruptedError(
            forKey: key, in: container, debugDescription: "must be \(range)")
    }
    return value
}

/// The server confirmed a wake for this turn. The panel opens.
public struct WakeConfirmed: Codable, Sendable, Equatable {
    public let turnId: String

    public init(turnId: String) { self.turnId = turnId }

    private enum CodingKeys: String, CodingKey {
        case type
        case turnId = "turn_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.wakeConfirmed)
        turnId = try shortText(c, .turnId, max: PanelLimits.idMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.wakeConfirmed, forKey: .type)
        try c.encode(turnId, forKey: .turnId)
    }
}

/// The turn moved to a new state. `state` is a word: listening, thinking or
/// speaking today, and any other word is kept as it is.
public struct TurnStateMessage: Codable, Sendable, Equatable {
    public let turnId: String
    public let state: String

    public init(turnId: String, state: String) {
        self.turnId = turnId
        self.state = state
    }

    private enum CodingKeys: String, CodingKey {
        case type, state
        case turnId = "turn_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.state)
        turnId = try shortText(c, .turnId, max: PanelLimits.idMax)
        state = try shortText(c, .state, max: PanelLimits.wordMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.state, forKey: .type)
        try c.encode(turnId, forKey: .turnId)
        try c.encode(state, forKey: .state)
    }
}

/// A live partial of the transcript. The next partial replaces it.
public struct TranscriptPartial: Codable, Sendable, Equatable {
    public let turnId: String
    public let text: String

    public init(turnId: String, text: String) {
        self.turnId = turnId
        self.text = text
    }

    private enum CodingKeys: String, CodingKey {
        case type, text
        case turnId = "turn_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.transcriptPartial)
        turnId = try shortText(c, .turnId, max: PanelLimits.idMax)
        text = try boundedText(c, .text, max: PanelLimits.transcriptTextMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.transcriptPartial, forKey: .type)
        try c.encode(turnId, forKey: .turnId)
        try c.encode(text, forKey: .text)
    }
}

/// The final transcript of the turn, with the wake phrase removed.
public struct TranscriptFinal: Codable, Sendable, Equatable {
    public let turnId: String
    public let text: String

    public init(turnId: String, text: String) {
        self.turnId = turnId
        self.text = text
    }

    private enum CodingKeys: String, CodingKey {
        case type, text
        case turnId = "turn_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.transcriptFinal)
        turnId = try shortText(c, .turnId, max: PanelLimits.idMax)
        text = try boundedText(c, .text, max: PanelLimits.transcriptTextMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.transcriptFinal, forKey: .type)
        try c.encode(turnId, forKey: .turnId)
        try c.encode(text, forKey: .text)
    }
}

/// The data of a `text` card.
public struct TextCardData: Codable, Sendable, Equatable {
    public let text: String

    public init(text: String) { self.text = text }

    private enum CodingKeys: String, CodingKey { case text }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        text = try shortText(c, .text, max: PanelLimits.cardTextMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(text, forKey: .text)
    }
}

/// The data of a `timer` card.
public struct TimerCardData: Codable, Sendable, Equatable {
    public let timerId: Int
    public let timerKind: String
    public let label: String

    public init(timerId: Int, timerKind: String, label: String) {
        self.timerId = timerId
        self.timerKind = timerKind
        self.label = label
    }

    private enum CodingKeys: String, CodingKey {
        case label
        case timerId = "timer_id"
        case timerKind = "timer_kind"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        timerId = try boundedInt(c, .timerId, in: 0...PanelLimits.timerIdMax)
        timerKind = try shortText(c, .timerKind, max: PanelLimits.wordMax)
        label = try boundedText(c, .label, max: PanelLimits.labelMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(timerId, forKey: .timerId)
        try c.encode(timerKind, forKey: .timerKind)
        try c.encode(label, forKey: .label)
    }
}

/// What a card carries. A kind this build does not know is `.unknown`, and the
/// card keeps its `fallbackText`, so a newer server never blanks the panel.
public enum CardData: Sendable, Equatable {
    case text(TextCardData)
    case timer(TimerCardData)
    case unknown
}

/// One card of a turn.
public struct CardMessage: Codable, Sendable, Equatable {
    public let turnId: String
    public let cardId: String
    public let kind: String
    public let fallbackText: String
    public let data: CardData

    public init(turnId: String, cardId: String, kind: String, fallbackText: String, data: CardData) {
        self.turnId = turnId
        self.cardId = cardId
        self.kind = kind
        self.fallbackText = fallbackText
        self.data = data
    }

    private enum CodingKeys: String, CodingKey {
        case type, kind, data
        case turnId = "turn_id"
        case cardId = "card_id"
        case fallbackText = "fallback_text"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.card)
        turnId = try shortText(c, .turnId, max: PanelLimits.idMax)
        cardId = try shortText(c, .cardId, max: PanelLimits.idMax)
        kind = try shortText(c, .kind, max: PanelLimits.wordMax)
        fallbackText = try boundedText(c, .fallbackText, max: PanelLimits.cardTextMax)
        switch kind {
        case "text": data = .text(try c.decode(TextCardData.self, forKey: .data))
        case "timer": data = .timer(try c.decode(TimerCardData.self, forKey: .data))
        default: data = .unknown
        }
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.card, forKey: .type)
        try c.encode(turnId, forKey: .turnId)
        try c.encode(cardId, forKey: .cardId)
        try c.encode(kind, forKey: .kind)
        try c.encode(fallbackText, forKey: .fallbackText)
        switch data {
        case .text(let value): try c.encode(value, forKey: .data)
        case .timer(let value): try c.encode(value, forKey: .data)
        case .unknown: break
        }
    }

    /// The text to show for this card: the data text of a text card, else the
    /// fallback text. `nil` when that is empty.
    public var displayText: String? {
        let text: String
        if case .text(let value) = data { text = value.text } else { text = fallbackText }
        return text.isEmpty ? nil : text
    }
}

/// The turn is over. `outcome` is a word: completed, no_speech, stopped or
/// failed today. The panel hides after the playback hint and any follow-up
/// window.
public struct TurnEnded: Codable, Sendable, Equatable {
    public let turnId: String
    public let outcome: String
    public let followUpWindowMs: Int
    public let playbackMsLeft: Int

    public init(turnId: String, outcome: String, followUpWindowMs: Int, playbackMsLeft: Int) {
        self.turnId = turnId
        self.outcome = outcome
        self.followUpWindowMs = followUpWindowMs
        self.playbackMsLeft = playbackMsLeft
    }

    private enum CodingKeys: String, CodingKey {
        case type, outcome
        case turnId = "turn_id"
        case followUpWindowMs = "follow_up_window_ms"
        case playbackMsLeft = "playback_ms_left"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.turnEnded)
        turnId = try shortText(c, .turnId, max: PanelLimits.idMax)
        outcome = try shortText(c, .outcome, max: PanelLimits.wordMax)
        followUpWindowMs = try boundedInt(c, .followUpWindowMs, in: 0...PanelLimits.msMax)
        playbackMsLeft = try boundedInt(c, .playbackMsLeft, in: 0...PanelLimits.msMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.turnEnded, forKey: .type)
        try c.encode(turnId, forKey: .turnId)
        try c.encode(outcome, forKey: .outcome)
        try c.encode(followUpWindowMs, forKey: .followUpWindowMs)
        try c.encode(playbackMsLeft, forKey: .playbackMsLeft)
    }
}

/// A timer or alarm is ringing. `kind` is a word: timer or alarm today.
public struct TimerRinging: Codable, Sendable, Equatable {
    public let timerId: Int
    public let kind: String
    public let label: String

    public init(timerId: Int, kind: String, label: String) {
        self.timerId = timerId
        self.kind = kind
        self.label = label
    }

    private enum CodingKeys: String, CodingKey {
        case type, kind, label
        case timerId = "timer_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.timerRinging)
        timerId = try boundedInt(c, .timerId, in: 0...PanelLimits.timerIdMax)
        kind = try shortText(c, .kind, max: PanelLimits.wordMax)
        label = try boundedText(c, .label, max: PanelLimits.labelMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.timerRinging, forKey: .type)
        try c.encode(timerId, forKey: .timerId)
        try c.encode(kind, forKey: .kind)
        try c.encode(label, forKey: .label)
    }
}

/// The timer stopped ringing. The ring view hides.
public struct TimerStopped: Codable, Sendable, Equatable {
    public let timerId: Int

    public init(timerId: Int) { self.timerId = timerId }

    private enum CodingKeys: String, CodingKey {
        case type
        case timerId = "timer_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.timerStopped)
        timerId = try boundedInt(c, .timerId, in: 0...PanelLimits.timerIdMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.timerStopped, forKey: .type)
        try c.encode(timerId, forKey: .timerId)
    }
}

/// The Mac asks the server to stop a ringing timer.
public struct TimerStop: Codable, Sendable, Equatable {
    public let timerId: Int

    public init(timerId: Int) { self.timerId = timerId }

    private enum CodingKeys: String, CodingKey {
        case type
        case timerId = "timer_id"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.timerStop)
        timerId = try boundedInt(c, .timerId, in: 0...PanelLimits.timerIdMax)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.timerStop, forKey: .type)
        try c.encode(timerId, forKey: .timerId)
    }
}
