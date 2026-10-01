/// The numbers of the `/ws/desktop` protocol, version 1.
///
/// Every value has a twin in `desktop/protocol/v1/constants.json` and in
/// `src/atlas/desktop/protocol.py`. `FixtureDecodingTests` fails if one drifts.
public enum ProtocolConstants {
    public static let protocolVersion = 1
    public static let pingIntervalS = 15
    public static let pongTimeoutS = 10
    public static let helloTimeoutS = 10
    public static let idleTimeoutS = 45
    public static let maxTextFrameBytes = 2048
    public static let maxInvalidMessages = 20
    public static let testTimeoutS = 5
}

/// What the app does when the server closes the socket with a given code.
public enum CloseAction: String, Sendable {
    case reconnect
    case retryAfterCap = "retry_after_cap"
    case unpair
    case stopUpdateApp = "stop_update_app"
}

/// The WebSocket close codes the server uses (D-21).
public enum CloseCode: Int, Sendable, CaseIterable {
    case goingAway = 1001
    case policyViolation = 1008
    /// Another socket for the same Mac replaced this one.
    case superseded = 4000
    /// The operator revoked this Mac. The app deletes its token and shows the
    /// Pair state (D-21).
    case revoked = 4001
    /// The server and the app speak different protocol versions. The app stops
    /// and tells the person to update (D-09).
    case protocolMismatch = 4002

    /// The snake_case name used in `close_codes.json`.
    public var name: String {
        switch self {
        case .goingAway: "going_away"
        case .policyViolation: "policy_violation"
        case .superseded: "superseded"
        case .revoked: "revoked"
        case .protocolMismatch: "protocol_mismatch"
        }
    }

    public var clientAction: CloseAction {
        switch self {
        case .goingAway, .policyViolation: .reconnect
        case .superseded: .retryAfterCap
        case .revoked: .unpair
        case .protocolMismatch: .stopUpdateApp
        }
    }
}
