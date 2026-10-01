import Foundation

// The frames of the `/ws/desktop` protocol (D-09). The caps mirror the Python
// models in `src/atlas/desktop/protocol.py`, and `FixtureDecodingTests` reads
// the same fixture files that pytest reads.

/// Wire name of every frame type.
enum WireType {
    static let hello = "hello"
    static let helloAck = "hello.ack"
    static let ping = "ping"
    static let pong = "pong"
    static let error = "error"
    static let wakeConfirmed = "wake.confirmed"
}

private enum Caps {
    static let textMax = 32
    static let capabilityMax = 64
    static let capabilitiesMax = 32
    static let idMax = 2_147_483_647
}

private enum TypeKey: String, CodingKey { case type }

/// Reads an integer and refuses a JSON bool, a string or a fraction. Python's
/// `StrictInt` does the same, so both sides drop the same frames.
func strictInt<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, _ key: K
) throws -> Int {
    if (try? container.decode(Bool.self, forKey: key)) != nil {
        throw DecodingError.typeMismatch(
            Int.self,
            .init(codingPath: container.codingPath + [key], debugDescription: "a bool is not an int"))
    }
    return try container.decode(Int.self, forKey: key)
}

func shortText<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, _ key: K, max: Int
) throws -> String {
    let value = try container.decode(String.self, forKey: key)
    // Python counts code points, so count unicode scalars here.
    let length = value.unicodeScalars.count
    guard length >= 1, length <= max else {
        throw DecodingError.dataCorruptedError(
            forKey: key, in: container, debugDescription: "length must be 1...\(max)")
    }
    return value
}

func requireType<K: CodingKey>(
    _ container: KeyedDecodingContainer<K>, _ key: K, equals expected: String
) throws {
    let actual = try container.decode(String.self, forKey: key)
    guard actual == expected else {
        throw DecodingError.dataCorruptedError(
            forKey: key, in: container, debugDescription: "type must be \(expected)")
    }
}

/// The first frame a Mac sends. It carries nothing about where the Mac is
/// (D-11): exactly the keys `type`, `protocol`, `app_version`, `os_version`
/// and `capabilities`.
public struct Hello: Codable, Sendable, Equatable {
    public let protocolVersion: Int
    public let appVersion: String
    public let osVersion: String
    public let capabilities: [String]

    public init(protocolVersion: Int, appVersion: String, osVersion: String, capabilities: [String]) {
        self.protocolVersion = protocolVersion
        self.appVersion = appVersion
        self.osVersion = osVersion
        self.capabilities = capabilities
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case protocolVersion = "protocol"
        case appVersion = "app_version"
        case osVersion = "os_version"
        case capabilities
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.hello)
        protocolVersion = try strictInt(c, .protocolVersion)
        appVersion = try shortText(c, .appVersion, max: Caps.textMax)
        osVersion = try shortText(c, .osVersion, max: Caps.textMax)
        let caps = try c.decode([String].self, forKey: .capabilities)
        guard caps.count <= Caps.capabilitiesMax,
            caps.allSatisfy({ (1...Caps.capabilityMax).contains($0.unicodeScalars.count) })
        else {
            throw DecodingError.dataCorruptedError(
                forKey: .capabilities, in: c, debugDescription: "capabilities out of range")
        }
        capabilities = caps
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.hello, forKey: .type)
        try c.encode(protocolVersion, forKey: .protocolVersion)
        try c.encode(appVersion, forKey: .appVersion)
        try c.encode(osVersion, forKey: .osVersion)
        try c.encode(capabilities, forKey: .capabilities)
    }
}

/// The server accepts the hello. Connected means this frame arrived.
public struct HelloAck: Codable, Sendable, Equatable {
    public let protocolVersion: Int
    public let deviceId: Int
    public let pingIntervalS: Int

    public init(protocolVersion: Int, deviceId: Int, pingIntervalS: Int) {
        self.protocolVersion = protocolVersion
        self.deviceId = deviceId
        self.pingIntervalS = pingIntervalS
    }

    private enum CodingKeys: String, CodingKey {
        case type
        case protocolVersion = "protocol"
        case deviceId = "device_id"
        case pingIntervalS = "ping_interval_s"
    }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.helloAck)
        protocolVersion = try strictInt(c, .protocolVersion)
        deviceId = try strictInt(c, .deviceId)
        pingIntervalS = try strictInt(c, .pingIntervalS)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.helloAck, forKey: .type)
        try c.encode(protocolVersion, forKey: .protocolVersion)
        try c.encode(deviceId, forKey: .deviceId)
        try c.encode(pingIntervalS, forKey: .pingIntervalS)
    }
}

private func pingId<K: CodingKey>(_ c: KeyedDecodingContainer<K>, _ key: K) throws -> Int {
    let id = try strictInt(c, key)
    guard (0...Caps.idMax).contains(id) else {
        throw DecodingError.dataCorruptedError(
            forKey: key, in: c, debugDescription: "id must be 0...\(Caps.idMax)")
    }
    return id
}

public struct Ping: Codable, Sendable, Equatable {
    public let id: Int

    public init(id: Int) { self.id = id }

    private enum CodingKeys: String, CodingKey { case type, id }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.ping)
        id = try pingId(c, .id)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.ping, forKey: .type)
        try c.encode(id, forKey: .id)
    }
}

public struct Pong: Codable, Sendable, Equatable {
    public let id: Int

    public init(id: Int) { self.id = id }

    private enum CodingKeys: String, CodingKey { case type, id }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.pong)
        id = try pingId(c, .id)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.pong, forKey: .type)
        try c.encode(id, forKey: .id)
    }
}

/// A named refusal. The server sends it before it closes with 4002 or 1008.
public struct ErrorMessage: Codable, Sendable, Equatable {
    public let code: String
    public let detail: String

    public init(code: String, detail: String) {
        self.code = code
        self.detail = detail
    }

    private enum CodingKeys: String, CodingKey { case type, code, detail }

    public init(from decoder: any Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        try requireType(c, .type, equals: WireType.error)
        code = try c.decode(String.self, forKey: .code)
        detail = try c.decode(String.self, forKey: .detail)
    }

    public func encode(to encoder: any Encoder) throws {
        var c = encoder.container(keyedBy: CodingKeys.self)
        try c.encode(WireType.error, forKey: .type)
        try c.encode(code, forKey: .code)
        try c.encode(detail, forKey: .detail)
    }
}

/// A frame the Mac sends. An unknown type decodes to `.unknown`, so a newer
/// build of this protocol never closes an older peer.
public enum ClientMessage: Codable, Sendable, Equatable {
    case hello(Hello)
    case ping(Ping)
    case pong(Pong)
    case unknown(type: String)

    public init(from decoder: any Decoder) throws {
        let type = try decoder.container(keyedBy: TypeKey.self).decode(String.self, forKey: .type)
        switch type {
        case WireType.hello: self = .hello(try Hello(from: decoder))
        case WireType.ping: self = .ping(try Ping(from: decoder))
        case WireType.pong: self = .pong(try Pong(from: decoder))
        default: self = .unknown(type: type)
        }
    }

    public func encode(to encoder: any Encoder) throws {
        switch self {
        case .hello(let value): try value.encode(to: encoder)
        case .ping(let value): try value.encode(to: encoder)
        case .pong(let value): try value.encode(to: encoder)
        case .unknown(let type):
            var c = encoder.container(keyedBy: TypeKey.self)
            try c.encode(type, forKey: .type)
        }
    }
}

/// A frame the server sends. An unknown type decodes to `.unknown` and the app
/// ignores it, so an old app keeps working against a newer server.
public enum ServerMessage: Decodable, Sendable, Equatable {
    case helloAck(HelloAck)
    case ping(Ping)
    case pong(Pong)
    case error(ErrorMessage)
    case wakeConfirmed(WakeConfirmed)
    case unknown(type: String)

    public init(from decoder: any Decoder) throws {
        let type = try decoder.container(keyedBy: TypeKey.self).decode(String.self, forKey: .type)
        switch type {
        case WireType.helloAck: self = .helloAck(try HelloAck(from: decoder))
        case WireType.ping: self = .ping(try Ping(from: decoder))
        case WireType.pong: self = .pong(try Pong(from: decoder))
        case WireType.error: self = .error(try ErrorMessage(from: decoder))
        case WireType.wakeConfirmed: self = .wakeConfirmed(try WakeConfirmed(from: decoder))
        default: self = .unknown(type: type)
        }
    }
}
