import Foundation

// The panel frames of the `/ws/desktop` protocol (Phase 15, D-10). They mirror
// the Python models in `src/atlas/desktop/protocol.py`, and
// `FixtureDecodingTests` reads the same fixture files that pytest reads.

/// Caps shared by the panel frames. They match the Python models.
public enum PanelLimits {
    public static let idMax = 64
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
